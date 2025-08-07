# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info

# # default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# # model = Qwen2VLForConditionalGeneration.from_pretrained(
# #     "Qwen/Qwen2-VL-7B-Instruct",
# #     torch_dtype=torch.bfloat16,
# #     attn_implementation="flash_attention_2",
# #     device_map="auto",
# # )

# # default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# # The default range for the number of visual tokens per image in the model is 4-16384.
# # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# # min_pixels = 256*28*28
# # max_pixels = 1280*28*28
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import os
from tqdm import tqdm
import json
import shortuuid

def load_model(model_path="Qwen/Qwen2-VL-7B-Instruct"):
    
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    return processor, model

def inference(processor, model, image_path, prompt):


    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     }
    # ]
    ## Local file path
    messages = [{"role": "system",
                "content": "You're an AI assistant specialized in histopathology image interpretation. Provide a paragraph answer as if you're directly observing and analyzing the image."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text[0])

    return output_text[0]

def eval_model(args, processor, model):
    base_image_path = args.image_folder  # 替换为实际路径

    with open(args.question_file, 'r') as f:
        lines = f.readlines()

    answers_file = os.path.expanduser(args.answers_file)
    failed_file = os.path.expanduser(args.failed_file)  # 新增：保存失败条目的文件
    print(answers_file)
    # print(lines)

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    failed_ans_file = open(failed_file, "w")  # 新增：打开失败条目的文件

    for line in tqdm(lines, desc="Processing items"):
        item = json.loads(line)
        prompt = item['text']
        image_name = item['image']
        image_path = f"{base_image_path}{image_name}"

        try:
            answer = inference(processor, model, image_path, prompt)  # 调用API或生成答案
            ans_id = shortuuid.uuid()

            # 保存成功的答案
            ans_file.write(json.dumps({
                "question": prompt,
                "answer": answer,
                "image_id": image_name.split('.')[0],
            }) + "\n")
            ans_file.flush()
        
        except Exception as e:
            # 如果处理失败，记录错误信息
            error_message = f"Error: {str(e)}"
            print(error_message)
            failed_ans_file.write(json.dumps({
                "question": prompt,
                "image_id": image_name.split('.')[0],
                "error": error_message
            }) + "\n")
            failed_ans_file.flush()

    ans_file.close()
    failed_ans_file.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/dpo_data/quilt_data/quilt_instruct/")
    parser.add_argument("--question-file", type=str, default="./playground/dpo_data/step1/quiltvqa_image_question_list_all_in.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/dpo_data/step2/Qwen2-VL-7B-Instruct-qwen-all-in.jsonl")
    parser.add_argument("--failed-file", type=str, default="./playground/dpo_data/step2/Qwen2-VL-7B-Instruct-qwen-failed.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    # prompt = "Describe this image."
    # image_path = "/data/dingmeidan/LLM/quilt-llava-all/playground/eval_data/quilt_vqa/images/04ktJuzyNfk_roi_410800ab-4178-4949-b57f-e17fe6aa5846.jpg"

    processor, model = load_model("Qwen/Qwen2-VL-7B-Instruct")
    eval_model(args, processor, model)
    # inference(processor, model, image_path, prompt)


