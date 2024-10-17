import json
from openai import OpenAI
import os
import requests

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))


# pip install requests || poetry add requests

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

STATUS = {"OK": "ok"}

url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
response = requests.get(url)
# print(json.dumps(response.json()) )

_response = response.json()
cnt = 0
if _response['status'] == STATUS["OK"]:
    for article in _response["articles"]:
        _article = f"title: {article['title']}"
        _description = f"description: {article['description']}"

        prompt = f"Translate the following 'title' and 'description' in Korean \n```\n{_article}\n{_description}```"
        r = client.completions.create(model="text-davinci-003",
        prompt=prompt,
        max_tokens=300)
        print(r.choices[0].text)

        if cnt > 3:
            break
        cnt += 1
