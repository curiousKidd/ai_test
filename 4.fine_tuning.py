from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

# before we train
prompt = f"What is Meta Platforms revenue in 2022?"
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": prompt},
        ]
    )
print("before...")
print(response.choices[0].message.content)


# prompt = f"What is Walmart revenue in 2022?"
response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:ai::AIzLxbPw",
        messages = [
            {"role": "user", "content": prompt},
        ]
    )
print("after...")
print(response.choices[0].message.content)

