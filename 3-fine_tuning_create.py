from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

# 파일 업로드
response = client.files.create(
  file=open("fine_tuning_data_prepared.jsonl", "rb"),
  purpose="fine-tune"
)

file_id = response.id  # 업로드된 파일의 ID를 저장
print(f"Uploaded file ID: {file_id}")

fine_tune = client.fine_tuning.jobs.create(
  training_file=file_id,
  model="gpt-3.5-turbo"
)

print(f"fine_tune: {fine_tune.id}")


# import openai
# import time

# # 파인튜닝 작업 ID
# fine_tune_id = "your-fine-tune-job-id"

# # 주기적으로 상태 확인
# while True:
#     fine_tune_status = client.fine_tuning.retrieve(fine_tune_id)
#     status = fine_tune_status.status
#     print(f"Current status: {status}")
    
#     if status == "succeeded":
#         print("Fine-tuning is complete!")
#         break
#     elif status == "failed":
#         print("Fine-tuning has failed.")
#         break
    
#     # 60초마다 상태를 확인
#     time.sleep(60)

