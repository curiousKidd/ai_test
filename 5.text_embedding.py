from openai import OpenAI

import os
import pandas as pd

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

df = pd.read_csv("sample/fortune_1000_revenue_2022.csv")
print(df.head())

def get_company_info(name: str, revenues: str, market_value: str, employees: str) -> str:
    context = f"{name} has {revenues} revenues, {market_value} market value and {employees} employees"
    return context

# DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)
df["info"] = df.apply(
                func=lambda df: get_company_info(
                            df['name'].strip(),
                            df['revenues'].strip(),
                            df['market_value'].strip(),
                            df['employees'].strip()),
                axis=1) # 0이면 열 기준 1이면 행 기준 데이터 셋
# print(df['info']) 


# https://platform.openai.com/docs/guides/embeddings/use-cases
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# text embedding
vector_result = get_embedding(text=df['info'][0])
# print(f"text_embedding: {vector_result}")

# dimension
print(f"dimension: {len(vector_result)}")

# text embedding for all the dataset
df['ada_embedding'] = df['info'].apply(
                        lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('sample/embedded_fortune_1k_revenue.csv', index=False)
print("embedding_completed...")


