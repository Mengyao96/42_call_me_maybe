import sys #standard libraries
import os
import json

from pydantic import BaseModel #3rd party libraries to define data structure
from llm_sdk import Small_LLM_Model
import numpy as np

def build_tool_dict(schema):
    tool_dict = {}
    for unit in schema:
       tool_dict[unit["fn_name"]] = {
        "args_names": unit["args_names"],
        "args_types": unit["args_types"]
       }
    return tool_dict

def get_str_id(voca_map, target_str):
    if target_str in voca_map:
        return voca_map[target_str]
    for token, idx in voca_map.items():
        if token == target_str:
            return idx
    return None

def get_num_ids(voca_map):
    num_ids = []
    for token, idx in voca_map.items():
        if token.isdigit() or token == '.':
            num_ids.append(idx)
    return num_ids

def get_next_tokenid(llm, allowed_ids, input_ids):
    logits = llm.get_logits_from_input_ids(input_ids)
    filtered_logits = [-float('inf')] * len(logits)
    for idx in allowed_ids:
        if 0 <= idx < len(logits):
            filtered_logits[idx] = logits[idx]
    
    if max(filtered_logits) == -float('inf'):
        best_id = logits.index(max(logits))
    else:
        best_id = filtered_logits.index(max(filtered_logits))
    
    return best_id

def get_func_name(llm, system_prompt, current_json, tools_dict):
    """
    生成函数名并返回函数名和更新后的 JSON 字符串
    
    参数:
        llm: 语言模型实例
        system_prompt: 系统提示词
        current_json: 当前 JSON 字符串
        tools_dict: 工具字典
    
    返回:
        (func_name, current_json): 函数名和更新后的 JSON 字符串
    """
    # === 第一步：生成函数名 ===
    # 为每个函数名生成 token 序列
    func_name_tokens = {}
    for fn_name in tools_dict.keys():
        # 编码函数名，得到 token 列表
        tokens = llm._encode(fn_name)[0].tolist()
        func_name_tokens[fn_name] = tokens

    full_context = system_prompt + current_json
    input_ids = llm._encode(full_context)[0].tolist()
    
    generated_func_name = ""
    max_func_name_length = max(len(tokens) for tokens in func_name_tokens.values())

    for token_pos in range(max_func_name_length):
        # 找出在当前位置，哪些函数名还是候选
        allowed_token_ids = set()
        
        for fn_name, tokens in func_name_tokens.items():
            # 检查已生成的部分是否匹配这个函数名的开头
            if token_pos < len(tokens):
                # 检查前面的 token 是否匹配
                current_tokens = llm._encode(generated_func_name)[0].tolist() if generated_func_name else []
                expected_tokens = tokens[:token_pos]
                
                if current_tokens == expected_tokens or token_pos == 0:
                    # 这个函数名还是候选，允许它的下一个 token
                    allowed_token_ids.add(tokens[token_pos])
        
        if not allowed_token_ids:
            break
        
        # 从允许的 token 中选择最优的
        next_token_id = get_next_tokenid(llm, list(allowed_token_ids), input_ids)
        next_token_str = llm._decode([next_token_id])
        
        generated_func_name += next_token_str
        current_json += next_token_str  # 修改局部变量
        
        # 更新上下文
        full_context = system_prompt + current_json
        input_ids = llm._encode(full_context)[0].tolist()
        
        # 检查是否完整匹配了某个函数名
        if generated_func_name in tools_dict:
            break
    
    func_name = generated_func_name
    print(f"Predicted function name: {func_name}\n")
    
    # 🔑 关键：返回函数名和更新后的 current_json
    return func_name, current_json




def constrained_generation(llm, prompt, schema, voca_map):
    tools_dict = build_tool_dict(schema)

    # 构建完整的提示词，告诉模型要生成 JSON 格式的函数调用
    # 关键：让模型从 prompt 中提取参数值
    system_prompt = (
        f"Extract function call from query.\n"
        f"Query: {prompt}\n"
        f"Available functions: {list(tools_dict.keys())}\n"
        f"Extract exact values from the query and format as JSON.\n"
        f"JSON: "
    )
    
    current_json = '{"fn_name": "'

    # === 第一步：生成函数名 ===
    func_name, current_json = get_func_name(llm, system_prompt, current_json, tools_dict)

    # === 第二步：生成参数 ===
    transition_str = '", "args": {'
    current_json += transition_str
    
    full_context = system_prompt + current_json
    input_ids = llm._encode(full_context)[0].tolist()

    if func_name in tools_dict:
        args_names = tools_dict[func_name]["args_names"]
        args_types = tools_dict[func_name]["args_types"]

        for i, arg_name in enumerate(args_names):
            current_json += f'"{arg_name}": '
            
            # 更新完整上下文
            full_context = system_prompt + current_json
            input_ids = llm._encode(full_context)[0].tolist()

            arg_type = args_types.get(arg_name, "str")
            
            # 根据参数类型生成值
            if arg_type in ["float", "int"]:
                # 对于数字类型，需要逐个 token 生成完整的数字
                generated_value = ""
                max_number_tokens = 5  # 限制为 5 个 token（避免生成过长数字）
                
                for token_idx in range(max_number_tokens):
                    allowed_val_ids = get_num_ids(voca_map)
                    
                    if not allowed_val_ids:
                        break
                    
                    val_id = get_next_tokenid(llm, allowed_val_ids, input_ids)
                    val_token = llm._decode([val_id])
                    
                    # 只保留数字和小数点字符
                    clean_token = ''.join(c for c in val_token if c in '0123456789.')
                    
                    # 如果 token 为空或包含非数字字符，停止
                    if not clean_token:
                        break
                    
                    generated_value += clean_token
                    current_json += clean_token
                    
                    # 更新上下文继续生成
                    full_context = system_prompt + current_json
                    input_ids = llm._encode(full_context)[0].tolist()
                    
                    # 检查下一个最可能的 token
                    next_logits = llm.get_logits_from_input_ids(input_ids)
                    next_best_id = next_logits.index(max(next_logits))
                    next_token = llm._decode([next_best_id])
                    
                    # 如果下一个 token 不是数字，停止
                    if not any(c in next_token for c in '0123456789.'):
                        break
                
                # 如果没生成任何数字，使用默认值
                if not generated_value:
                    generated_value = "0"
                    current_json += "0"
                    
            else:
                # 对于字符串类型，生成带引号的字符串
                current_json += '"'
                full_context = system_prompt + current_json
                input_ids = llm._encode(full_context)[0].tolist()
                
                generated_value = ""
                max_string_tokens = 15  # 限制字符串长度
                
                for token_idx in range(max_string_tokens):
                    # 允许所有 token
                    allowed_val_ids = range(len(voca_map))
                    
                    val_id = get_next_tokenid(llm, allowed_val_ids, input_ids)
                    val_token = llm._decode([val_id])
                    
                    # 如果遇到引号或逗号，说明参数值结束
                    if '"' in val_token or ',' in val_token or '}' in val_token:
                        break
                    
                    generated_value += val_token
                    current_json += val_token
                    
                    # 更新上下文
                    full_context = system_prompt + current_json
                    input_ids = llm._encode(full_context)[0].tolist()
                
                current_json += '"'

            if i < len(args_names) - 1:
                current_json += ", "

        current_json += "}}"  # 关闭 args 和整个 JSON

    else:
        # 如果函数名不在字典中，关闭 JSON
        current_json += "}}"

    try:
        return json.loads(current_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Generated JSON string: {current_json}")
        return {"fn_name": func_name, "args": {}}


def is_json_complete(text):
    if not text:
        return False
    
    open_braces = text.count('{')
    close_braces = text.count('}')
    if open_braces > 0 and open_braces == close_braces:
        return True
    return False

def main():
    llm = Small_LLM_Model()  # 修复：直接实例化，不要调用 __init__()

    with open("data/exercise_input/function_calling_tests.json", "r") as f:
        prompts = json.load(f)
    with open("data/exercise_input/functions_definition.json", "r") as f:
        schema = json.load(f)

    voca_path = llm.get_path_to_vocabulary_json()
    with open(voca_path, "r") as f:
        voca_map = json.load(f)

    outputs = []
    
    # 确保输出目录存在
    output_dir = "data/exercise_output"
    os.makedirs(output_dir, exist_ok=True)

    for prompt in prompts:
        prompt_str = prompt["prompt"]
        print(f"prompt: {prompt_str}")

        res_json = constrained_generation(llm, prompt_str, schema, voca_map)
        output_entry = {
            "prompt": prompt_str,
            "fn_name": res_json.get("fn_name", ""),
            "args": res_json.get("args", {})
        }
        outputs.append(output_entry)

    output_path = os.path.join(output_dir, "function_calling_results.json")
    with open(output_path, "w") as output_file:
        json.dump(outputs, output_file, indent=4)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"📊 Processed {len(outputs)} prompts")


if __name__ == "__main__":
    main()