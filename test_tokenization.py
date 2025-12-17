from llm_sdk import Small_LLM_Model

def main():
    llm = Small_LLM_Model()
    
    # 测试文本
    text = "What is 2 + 2?"
    
    # Encode
    tokens_tensor = llm._encode(text)
    print(f"原始文本: '{text}'")
    print(f"Tokens 张量 (2D): {tokens_tensor}")
    print(f"Tokens 张量形状: {tokens_tensor.shape}")
    print(f"Tokens 列表 (1D): {tokens_tensor[0].tolist()}")
    
    # 解码每个 token 看看它们是什么
    token_ids = tokens_tensor[0].tolist()
    print(f"\n每个 token ID 对应的文本:")
    for i, token_id in enumerate(token_ids):
        decoded = llm._decode([token_id])
        print(f"  Token {i}: ID={token_id:5d} → '{decoded}'")
    
    # 完整解码
    full_text = llm._decode(token_ids)
    print(f"\n完整解码: '{full_text}'")
    
    # 测试不同的文本
    print("\n" + "="*60)
    test_cases = [
        "Hello",
        "Hello world",
        "2+2",
        "2 + 2",
    ]
    
    for test_text in test_cases:
        tokens = llm._encode(test_text)[0].tolist()
        print(f"\n文本: '{test_text}'")
        print(f"Token 数量: {len(tokens)}")
        print(f"每个 token:")
        for i, tid in enumerate(tokens):
            print(f"  {i}: {tid:5d} → '{llm._decode([tid])}'")

if __name__ == "__main__":
    main()
