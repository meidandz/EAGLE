"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on vision language models.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""
from transformers import AutoTokenizer, AutoProcessor

from vllm import LLM, SamplingParams
# from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
import os
import json
from tqdm import tqdm
from PIL import Image

# Input image and question
# image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
# question = "What is the content of this image?"


# LLaVA-1.5
def llava_inference(question, tokenizer):

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    
    return prompt


# LLaVA-1.6/LLaVA-NeXT
def llava_next_inference(question, tokenizer):

    prompt = f"[INST] <image>\n{question} [/INST]"

    return prompt


# Fuyu
def fuyu_inference(question, tokenizer):

    prompt = f"{question}\n"
    
    return prompt


# Phi-3-Vision
def phi3v_inference(question, tokenizer):

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501


    return prompt


# PaliGemma
def paligemma_inference(question, tokenizer):

    # PaliGemma has special prompt format for VQA
    prompt = "caption en"

    return prompt


# Chameleon
def chameleon_inference(question, tokenizer):

    prompt = f"{question}<image>"

    return prompt


# MiniCPM-V
def minicpmv_inference(question, tokenizer):

    messages = [{
        'role': 'user',
        'content': f'(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return prompt


# InternVL
def internvl_inference(question, tokenizer):

    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return prompt


# BLIP-2
def blip2_inference(question, tokenizer):

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"

    return prompt


def qwenvl_inference(question, tokenizer):

    prompt = f"{question}Picture 1: <img></img>\n"

    return prompt


# Qwen2-VL
def qwen2vl_inference(question, tokenizer):

    prompt = ("<|im_start|>system\nYou are an AI assistant that specializes in pathological diagnosis questions and answers. If the pathological description of this image contains illusions and errors, please help me rewrite it.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")

    return prompt


# LLama 3.2
def mllama_inference(question, tokenizer):

    prompt = f"<|image|><|begin_of_text|>{question}"

    return prompt


# GLM-4v
def glm4v_inference(question, tokenizer):

    prompt = question

    return prompt


#NVLM-D
def nvlm_d_inference(question,tokenizer):

    messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    return prompt



def run_llava_build():


    llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_llava_next_build():

    
    llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf")
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_fuyu_build():

    llm = LLM(model="adept/fuyu-8b")
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_phi3v_build():

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model="microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True,
        max_num_seqs=5,
    )
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_paligemma_build():

    # PaliGemma has special prompt format for VQA

    llm = LLM(model="google/paligemma-3b-mix-224")
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_chameleon_build():

    llm = LLM(model="facebook/chameleon-7b")
    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_minicpmv_build():

    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    #2.6
    model_name = "openbmb/MiniCPM-V-2_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    # stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6
    stop_tokens = ['<|im_end|>', '<|endoftext|>']
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, tokenizer, stop_token_ids

def run_blip2_build():

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa

    llm = LLM(model="Salesforce/blip2-opt-2.7b")
    tokenizer = None
    stop_token_ids = None
    return llm, prompt, stop_token_ids

def run_qwenvl_build():

    llm = LLM(
        model="Qwen/Qwen-VL",
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=2,
    )

    tokenizer = None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_internvl_build():
    model_name = "OpenGVLab/InternVL2-8B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_num_seqs=5,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return llm, tokenizer, stop_token_ids

def run_qwen2vl_build():
    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=5,
        gpu_memory_utilization=0.9
    )

    tokenizer =None
    stop_token_ids = None

    return llm, tokenizer, stop_token_ids

def run_mllama_build():

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
    )
    tokenizer =None
    stop_token_ids = None
    return llm, tokenizer, stop_token_ids

def run_glm4v_build():

    model_name = "THUDM/glm-4v-9b"

    llm = LLM(model=model_name,
              max_model_len=2048,
              max_num_seqs=2,
              trust_remote_code=True,
              enforce_eager=True)
    tokenizer = None
    stop_token_ids = [151329, 151336, 151338]
    return llm, tokenizer, stop_token_ids

def run_nvlm_d_build():

    model_name = "nvidia/NVLM-D-72B"

    # Adjust this as necessary to fit in GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=4,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    stop_token_ids = None
    return llm, tokenizer, stop_token_ids


model_example_map = {
    "llava-1.5-7b-hf": llava_inference,
    "llava-v1.6-mistral-7b-hf": llava_next_inference,
    "fuyu-8b": fuyu_inference,
    "Phi-3-vision-128k-instruct": phi3v_inference,
    "paligemma-3b-mix-224": paligemma_inference,
    "chameleon-7": chameleon_inference,
    "MiniCPM-V-2_6": minicpmv_inference,
    "blip2-opt-2.7b": blip2_inference,
    "InternVL2-8B": internvl_inference,
    "Qwen-VL": qwenvl_inference,
    "Qwen2-VL-7B-Instruct": qwen2vl_inference,
    "Llama-3.2-11B-Vision-Instruct": mllama_inference,
    "glm-4v-9b": glm4v_inference,
    "NVLM-D-72B":nvlm_d_inference,
}


model_llm_build = {
    "llava-1.5-7b-hf": run_llava_build,
    "llava-v1.6-mistral-7b-hf": run_llava_next_build,
    "fuyu-8b": run_fuyu_build,
    "Phi-3-vision-128k-instruct": run_phi3v_build,
    "paligemma-3b-mix-224": run_paligemma_build,
    "chameleon-7": run_chameleon_build,
    "MiniCPM-V-2_6": run_minicpmv_build,
    "blip2-opt-2.7b": run_blip2_build,
    "InternVL2-8B": run_internvl_build,
    "Qwen-VL": run_qwenvl_build,
    "Qwen2-VL-7B-Instruct": run_qwen2vl_build,
    "Llama-3.2-11B-Vision-Instruct": run_mllama_build,
    "glm-4v-9b": run_glm4v_build,
    "NVLM-D-72B":run_nvlm_d_build,
}


def main(args):

    model = args.model_type   #模型名称
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    
    llm, tokenizer, stop_token_ids = model_llm_build[model]()

    sampling_params = SamplingParams(temperature=args.temperature,
                                     max_tokens=4096,
                                     stop_token_ids=stop_token_ids)
    
    base_image_path = args.image_folder #图片路径

    # ans_name = f"{model}_20k_t{args.temperature}.jsonl"
    ans_name = f"{args.name}_rewrite.jsonl"
    answers_file = os.path.expanduser(os.path.join(args.ans_path, ans_name))
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # failed_name = f"{model}_failed.jsonl"
    # failed_file = os.path.expanduser(os.path.join(args.ans_path, failed_name))
    # os.makedirs(os.path.dirname(failed_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    # failed_ans_file = open(failed_file, "w")
    
    
    with open(args.question_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing items"):
        item = json.loads(line)
        question = item['question']
        answer = item['answer']
        answer = answer.replace('</s>', '')
        image_name = item['image_id'] + '.jpg'


        image_file = f"{base_image_path}{image_name}"

        image = Image.open(image_file).convert('RGB')

        prompt = model_example_map[model](answer, tokenizer)

        assert args.num_prompts ==1 
        if args.num_prompts == 1:
            # Single inference
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            }
        else:
            # Batch inference
            inputs = [{
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            } for _ in range(args.num_prompts)]

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text
            # print(generated_text)

        # 保存成功的答案
        ans_file.write(json.dumps({
            "question": question,
            "answer": generated_text,
            "image_id": image_name.split('.')[0],
        }) + "\n")
        ans_file.flush()
    ans_file.close()
    # failed_ans_file.close()

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="Qwen2-VL-7B-Instruct",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument('--name',
                        type=str,
                        default="quiltllava_answer_file_all_in",
                        help='key_model_name.')
    parser.add_argument('--temperature',
                        type=int,
                        default=0.2,
                        help='temperature.')
    parser.add_argument("--image-folder", 
                        type=str, 
                        default="./playground/dpo_data/quilt_data/quilt_instruct/")
    parser.add_argument("--question-file", 
                        type=str, 
                        default="./playground/dpo_data/step2/quiltllava_answer_file_all_in.jsonl")
    parser.add_argument("--ans-path", 
                        type=str, 
                        default="./playground/example/")

    args = parser.parse_args()
    main(args)

    