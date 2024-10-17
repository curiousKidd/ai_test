from openai import OpenAI

import os
import pandas as pd
import tiktoken

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

df = pd.read_csv("sample/fortune_1000_revenue_2022.csv")

# https://platform.openai.com/docs/guides/embeddings/second-generation-models
tokenizer = "cl100k_base"

# https://platform.openai.com/docs/guides/embeddings/how-can-i-tell-how-many-tokens-a-string-has-before-i-embed-it
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # "gpt-4"도
    num_tokens = len(encoding.encode(string))
    return num_tokens

# total token
total_token = 0
for i, info in enumerate(df['info']):
    total_token += num_tokens_from_string(string=df['info'][i], encoding_name=tokenizer)
print(f"total_token: {total_token}")

# total price
total_price_est = total_token / 1000 * 0.0004
print(f"total_price_est: ${total_price_est}")