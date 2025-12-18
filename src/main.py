import sys #standard libraries
import os
import json

from pydantic import BaseModel #3rd party libraries to define data structure
from llm_sdk import Small_LLM_Model
import numpy as np

# def main():
#     llm = Small_LLM_Model()  # 修复：直接实例化，不要调用 __init__()
#     prompt = "What is 2 + 2?"
#     tokens_tensor = llm._encode(prompt)

#     logits = llm.get_logits_from_input_ids(tokens_tensor[0].tolist())
#     print(f"Encoded tokens: {tokens_tensor}")
#     print(f"Logits for next token: {logits}")

#     highest_logit_index = logits.index(max(logits))
#     next_token = llm._decode([highest_logit_index])
#     print(f"Predicted next token: {next_token}")


# def handle_each_prompt(prompt):
    
#     tokens_tensor = llm._encode(prompt)

#     logits = llm.get_logits_from_input_ids(tokens_tensor[0].tolist())
#     while max(logits) < 0:
#         next_token = verify_and_get_next_token(llm, logits)
#         prompt += next_token
#         tokens_tensor = llm._encode(prompt)
#         logits = llm.get_logits_from_input_ids(tokens_tensor[0].tolist())
    

# "prompt": "What is the sum of 2 and 3?",
# "fn_name": "fn_add_numbers",
# "args": {"a": 2.0, "b": 3.0}

# "prompt": "Reverse the string 'hello'",
# "fn_name": "fn_reverse_string",
# "args": {"s": "hello"}

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

def constrained_generation(llm, prompt, schema, voca_map):
    tools_dict = build_tool_dict(schema)

    # 构建完整的提示词，告诉模型要生成 JSON 格式的函数调用
    system_prompt = (
        f"User query: {prompt}\n"
        f"Available functions: {list(tools_dict.keys())}\n"
        f"Output a JSON function call in this format: {{\"fn_name\": \"function_name\", \"args\": {{...}}}}\n"
        f"JSON: {{"
    )
    
    current_json = '{"fn_name": "'

    # 收集允许的函数名 token IDs
    allowed_func_ids = []
    for fn_name in tools_dict.keys():
        func_id = get_str_id(voca_map, fn_name)
        if func_id is not None:
            allowed_func_ids.append(func_id)

    # 使用完整的提示词（包括指令）进行编码
    input_ids = llm._encode(system_prompt + '"fn_name": "')[0].tolist()

    func_id = get_next_tokenid(llm, allowed_func_ids, input_ids)
    func_name = llm._decode([func_id])

    print(f"Predicted function name: {func_name} \n")

    # 更新 current_json 和 input_ids
    current_json += func_name
    # 重新构建完整的上下文，包括已生成的函数名
    full_context = system_prompt + current_json + '", "args": {'
    input_ids = llm._encode(full_context)[0].tolist()

    transition_str = '", "args": {'
    current_json += transition_str

    if func_name in tools_dict:
        args_names = tools_dict[func_name]["args_names"]
        args_types = tools_dict[func_name]["args_types"]
        print(f"Function arguments: {args_names} \n")
        print(f"Function argument types: {args_types} \n")

        for i, arg_name in enumerate(args_names):
            current_json += f'"{arg_name}"'
            
            # 更新完整上下文
            full_context = system_prompt + current_json + ': '
            input_ids = llm._encode(full_context)[0].tolist()

            arg_type = args_types.get(arg_name, "str")
            if arg_type in ["float", "int"]:
                allowed_val_ids = get_num_ids(voca_map)
            else:
                allowed_val_ids = range(len(voca_map))  # allow all tokens for string
            
            val_id = get_next_tokenid(llm, allowed_val_ids, input_ids)
            val_str = llm._decode([val_id])
            current_json += f": {val_str}"

            if i < len(args_names) - 1:
                current_json += ", "

        current_json += "}"

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