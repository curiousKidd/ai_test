import inspect
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))



def factorial(n: int) -> int:
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


# print(inspect.getsource(reverse_string))



response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": f"Provide python docstring for the following function"},
            {"role": "user", "content": f"```{inspect.getsource(factorial)}```"},
        ]
    )

print("답변 : \n" , response.choices[0].message.content)
