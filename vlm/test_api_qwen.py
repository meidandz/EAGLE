 
###########################offline#######################################################################
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3", gpu_memory_utilization=0.5)

# Prepare your prompts
prompt = "Your observation about the presence of a nodule with preserved trabecular architecture of hepatocytes and lymphocytes at the edge is accurate. This indeed suggests a chronic liver condition. However, cirrhosis and chronic hepatitis are not the only possibilities. Hint: Consider the significance of the lymphocytes being present at the edge of the nodule. What could this suggest about the ongoing process? Also, think about the conditions that might involve nodules of different sizes in the liver. For example, there are diseases that can progress from cirrhosis to something more severe."
messages = [
    {"role": "system", 
     "content": '''You are an AI assistant specialized in processing pathological diagnosis Q&A pairs. 
                I will provide you with a pathology diagnosis question and its corresponding answer. 
                Please extract keywords from the following pathology description,paying special attention to medical terms and important concepts.
                Output Format: Present the extracted claims as a list in the following format: 
                ["claim1", "claim2", "claim3", ...]'''
                },
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

############################bendi##########################################################################################

# from openai import OpenAI
# # Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_base = "http://172.31.58.9:6006/v1"
# openai_api_key = "EMPTY"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# chat_response = client.chat.completions.create(
#     model="mistralai/Mistral-7B-Instruct-v0.3",
#     messages=[
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": "Tell me something about large language models."},
#     ],
#     temperature=0.7,
#     top_p=0.8,
#     max_tokens=512,
#     extra_body={
#         "repetition_penalty": 1.05,
#     },
# )
# print("Chat response:", chat_response)





# CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server --host 172.31.58.9 --port 6006   --model Qwen/Qwen2-VL-7B-Instruct   --gpu-memory-utilization 0.98  --trust-remote-code
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --host 172.31.58.9 --port 6007   --model alpindale/Llama-3.2-11B-Vision   --gpu-memory-utilization 0.7
# CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server --host 172.31.58.9 --port 6006   --model mistralai/Mistral-7B-Instruct-v0.3   --gpu-memory-utilization 0.98 --load-format mistral --config-format mistral --tokenizer_mode mistral
# CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server --host 172.31.58.9 --port 6006   --model OpenGVLab/InternVL2-8B  --gpu-memory-utilization 0.98 --trust_remote_code 
# OpenGVLab/InternVL2-8B
# mistralai/Pixtral-12B-2409
# mistralai/Mistral-7B-Instruct-v0.3
# alpindale/Llama-3.2-11B-Vision
# meta-llama/Llama-3.2-11B-Vision
# mistralai/Pixtral-12B-2409

# from openai import OpenAI
# import argparse
# import base64
# import requests
# from tqdm import tqdm
# import json
# import os
# import shortuuid



# openai_api_base = "http://172.31.58.9:6006/v1"
# openai_api_key = "EMPTY"
# client = OpenAI(
# api_key=openai_api_key,
# base_url=openai_api_base,
# )

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def request(image_path, prompt):#,system):
#     # Getting the base64 string
#     base64_image = encode_image(image_path)

#     response = client.chat.completions.create(
#     model="Qwen/Qwen2-VL-7B-Instruct",
#     messages=[{
#         "role": "system",
#         "content": "You're an AI assistant specialized in histopathology image interpretation. Provide a paragraph answer as if you're directly observing and analyzing the image."},
#         {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt},
#             {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{base64_image}",
#             },
#             },
#         ],
#         }
#     ],
#     max_tokens=300,
#     )

#     # print(response.choices[0].message.content)

#     return response.choices[0].message.content

# import time
# import http

# def request(image_path, prompt, retries=3, backoff_factor=2):
#     base64_image = encode_image(image_path)
    
#     for attempt in range(retries):
#         try:
#             response = client.chat.completions.create(
#                 model="Qwen/Qwen2-VL-7B-Instruct",
#                 messages=[{
#                     "role": "system",
#                     "content": "You're an AI assistant specialized in histopathology image interpretation. Provide a paragraph answer as if you're directly observing and analyzing the image."
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{base64_image}",
#                             },
#                         },
#                     ],
#                 }],
#                 max_tokens=300,
#             )
#             return response.choices[0].message.content
#         except http.ReadError as e:
#             if attempt < retries - 1:
#                 wait_time = backoff_factor ** attempt
#                 print(f"连接重置。将在 {wait_time} 秒后重试...")
#                 time.sleep(wait_time)
#             else:
#                 raise e  # 如果所有重试失败，则重新抛出异常


# def eval_model_1(args):
#     base_image_path = args.image_folder  # 替换为实际路径

#     with open(args.question_file, 'r') as f:
#         lines = f.readlines()

#     answers_file = os.path.expanduser(args.answers_file)
#     # answers_file = args.answers_file
#     print(answers_file)
#     os.makedirs(os.path.dirname(answers_file), exist_ok=True)
#     ans_file = open(answers_file, "w")

#     for line in tqdm(lines, desc="Processing items"):
#         item = json.loads(line)
#         # system = item['system']
#         # print(line)
#         # idx = item["question_id"]
#         prompt = item['text']
#         image_name = item['image']
#         image_path = f"{base_image_path}{image_name}"
#         answer = request(image_path, prompt)#, system)

#         ans_id = shortuuid.uuid()
#         # ans_file.write(json.dumps({"question_id": idx,
#         #                            "prompt": prompt,
#         #                            "text": answer,
#         #                            "answer_id": ans_id,
#         #                            "metadata": {}}) + "\n")
#         ans_file.write(json.dumps({
#                                    "question": prompt,
#                                    "answer": answer,
#                                    "image_id": image_name.split('.')[0],
#                                    }) + "\n")
#         ans_file.flush()
#     ans_file.close()

# import os
# import json
# import shortuuid
# from tqdm import tqdm

# def eval_model(args):
#     base_image_path = args.image_folder  # 替换为实际路径

#     with open(args.question_file, 'r') as f:
#         lines = f.readlines()

#     answers_file = os.path.expanduser(args.answers_file)
#     failed_file = os.path.expanduser(args.failed_file)  # 新增：保存失败条目的文件
#     print(answers_file)

#     os.makedirs(os.path.dirname(answers_file), exist_ok=True)
#     ans_file = open(answers_file, "w")
#     failed_ans_file = open(failed_file, "w")  # 新增：打开失败条目的文件

#     for line in tqdm(lines, desc="Processing items"):
#         item = json.loads(line)
#         prompt = item['text']
#         image_name = item['image']
#         image_path = f"{base_image_path}{image_name}"

#         try:
#             answer = request(image_path, prompt)  # 调用API或生成答案
#             ans_id = shortuuid.uuid()

#             # 保存成功的答案
#             ans_file.write(json.dumps({
#                 "question": prompt,
#                 "answer": answer,
#                 "image_id": image_name.split('.')[0],
#             }) + "\n")
#             ans_file.flush()
        
#         except Exception as e:
#             # 如果处理失败，记录错误信息
#             error_message = f"Error: {str(e)}"
#             print(error_message)
#             failed_ans_file.write(json.dumps({
#                 "question": prompt,
#                 "image_id": image_name.split('.')[0],
#                 "error": error_message
#             }) + "\n")
#             failed_ans_file.flush()

#     ans_file.close()
#     failed_ans_file.close()

        
#         # result = {"question": prompt,"answer": answer,"image_id": image_name[:-4]}
#         # with open('test.jsonl', 'a') as f:  # 使用'a'模式追加
#         #     json.dump(result, f, indent=4)
#         #     f.write('\n')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
#     parser.add_argument("--model-base", type=str, default=None)
#     parser.add_argument("--image-folder", type=str, default="./playground/dpo_data/quilt_data/quilt_instruct/")
#     parser.add_argument("--question-file", type=str, default="./playground/dpo_data/step1/quiltvqa_image_question_list_all_in.jsonl")
#     parser.add_argument("--answers-file", type=str, default="./playground/dpo_data/step2/Qwen2-VL-7B-Instruct-1-all-in.jsonl")
#     parser.add_argument("--failed-file", type=str, default="./playground/dpo_data/step2/Qwen2-VL-7B-Instruct-failed.jsonl")
#     parser.add_argument("--conv-mode", type=str, default="llava_v1")
#     parser.add_argument("--num-chunks", type=int, default=1)
#     parser.add_argument("--chunk-idx", type=int, default=0)
#     parser.add_argument("--temperature", type=float, default=0.2)
#     parser.add_argument("--top_p", type=float, default=None)
#     parser.add_argument("--num_beams", type=int, default=1)
#     args = parser.parse_args()

#     eval_model(args)


