import sys #standard libraries
import os
import json

from pydantic import BaseModel #3rd party libraries to define data structure
from llm_sdk import Small_LLM_Model
import numpy as np

def main():
    llm = Small_LLM_Model()  # 修复：直接实例化，不要调用 __init__()
    prompt = "What is 2 + 2?"
    tokens_tensor = llm._encode(prompt)

    logits = llm.get_logits_from_input_ids(tokens_tensor[0].tolist())
    print(f"Encoded tokens: {tokens_tensor}")
    print(f"Logits for next token: {logits}")

    highest_logit_index = logits.index(max(logits))
    next_token = llm._decode([highest_logit_index])
    print(f"Predicted next token: {next_token}")

if __name__ == "__main__":
    main()