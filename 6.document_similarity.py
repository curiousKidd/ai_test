from openai import OpenAI
import os
import numpy as np
import pandas as pd

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

# 유사도 계산 방식
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#임베딩
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

df = pd.read_csv("sample/embedded_fortune_1k_revenue.csv")
# print(df.head())

query = "what was Amazon revenues"

query_vector = get_embedding(text=query)
# print(query_vector)

print("cosine_similarity : ", cosine_similarity(np.array(np.matrix(df['ada_embedding'][0])).ravel(), query_vector))

df['cosine_similarity'] = df['ada_embedding'].apply(
                            lambda v: cosine_similarity(np.array(np.matrix(v)).ravel(), query_vector))
print(df['cosine_similarity'])

most_similar_index = np.argmax(df['cosine_similarity'])
print(f"Most similar sentence:: {df['info'][most_similar_index]}")



# 임베디드를 통한 값 추출 before & after
print("------------------------------")

print("before...")
embeded_prompt = [
    {"role": "system", "content": "Answer only if you are 100% certain."},
    # {"role": "system", "content": f"Reference: {df['info'][most_similar_index]}"},
    {"role": "user", "content": f"Question: {query}"},
    {"role": "assistant", "content": "Answer: "}
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=embeded_prompt
)
print(f"openai answer: {response.choices[0].message.content}")


print("after...")
embeded_prompt = [
    {"role": "system", "content": "Answer only if you are 100% certain."},
    {"role": "system", "content": f"Reference: {df['info'][most_similar_index]}"},
    {"role": "user", "content": f"Question: {query}"},
    {"role": "assistant", "content": "Answer: "}
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=embeded_prompt
)
print(f"openai answer: {response.choices[0].message.content}")


