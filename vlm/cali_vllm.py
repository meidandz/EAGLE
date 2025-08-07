from vllm import LLM, SamplingParams
import json

easy_prompt="A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the questions.\n\nUSER: For the following paragraph give me a paraphrase of the same using a very small vocabulary and extremely simple sentences that a toddler will understand: "

medium_prompt="A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the questions.\nUSER: For the following paragraph give me a diverse paraphrase of the same in high quality English language as in sentences on Wikipedia:\n"
QA_prompt="A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the questions.\nUSER: Convert the following paragraph into a conversational format with multiple tags of \"Question:\" followed by \"Answer:\":\n"

#三句话全空格
new_medium_prompt="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. USER: For the following paragraph give me a diverse paraphrase of the same in high quality English language as in sentences on Wikipedia:
"""
new_QA_style_prompt="""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. USER: Convert the following paragraph into a conversational format with multiple tags of "Question:" followed by "Answer:":
"""

new_original_text="""The stock rose $2.11, or about 11 percent, to close Friday at $21.51 on the New York Stock Exchange.

Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier."""


system_prompt="A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the questions."

medium_user_prompt="USER: For the following paragraph give me a diverse paraphrase of the same in high quality English language as in sentences on Wikipedia:"

original_text="The stock rose $2.11, or about 11 percent, to close Friday at $21.51 on the New York Stock Exchange.\n\nRevenue in the first quarter of the year dropped 15 percent from the same period a year earlier."

long_original_text="""
First round on stress at work survey. Answering the questionnaire is
voluntary and all answers will be saved anonymously. Please fill in this
questionnaire only if you have some work experience, part-or full time.
Otherwise, you will not be able to answer some of the questions! Here is
a the link to all language version.

Not that there’s a thing wrong with frozen burgers. The key here
is the meat seasonings, which are pretty strong and spicy and just GOOD,
something else I think is really necessary in a turkey burger because
ground turkey otherwise can be kind of flavorless. You’ll need ground
turkey, onion powder, chili powder, salt, pepper, and cayenne pepper for
the burgers. Then the mayo takes garlic and onion. Then we need buns,
clearly, swiss cheese, lettuce, and onion. I LOVE tomatoes but sometimes
find that they get in the way of other flavors, so I left them off of this
burger. Add them if you’d like to your array of toppings! First, we’ll
make the mayo. Grate the garlic directly into the mayo, add a pinch of
salt, and squeeze in the lemon juice. Stir. Done! I love this. Then, we’ll
work on the burgers. Preheat a large skillet to medium-high heat with some
olive oil, preheat the broiler to high, then add all the spices to the
ground turkey.

Whether you like your velvet crushed, vibrant or head-to-toe, there’s
really no denying the sheer luxe and elegance of this timeless textile.
Not only is it super stylish, it can actually be so wearable for day-to-day
wear. Yes, really! This year it’s all about embracing fun gem-toned
velvety pieces. Long gone are the days when velvet was solely associated
with dark moody shades of navy and black. Below we’ve rounded up the most
covetable velvet pieces on the high street right now. We’re already coming
up with outfit ideas! Are you completely obsessed or beyond bored of it?
Save up to $8,086 on one of 1,258 Chrysler 200s near you. Find
your perfect car with Edmunds expert and consumer car reviews, dealer
reviews, car comparisons and pricing tools. We have 4,850,420. Research
2015 Chrysler 200 Sedan 4D 200C I4 prices, used values & 200 Sedan 4D 200C
I4 pricing, specs and more. Many years ago, we wrote about the stalling
problem with the 2011 Chrysler 200, and believe it or not, we still receive
an occasional call regarding the problem.However, a much larger issue has
monopolized the phone lines as of late 2015 Chrysler 200 transmission
problems leaving drivers with check engine lights, harsh shifting, and the
occasional loss of power. The 2015 Chrysler 200 can fetch a premium for
its style and its horsepower--but rear-seat room and handling are better
bargains elsewhere. Find out why the 2015 Chrysler 200 is rated 8.4 by
The. Don’t know where to find the perfect rims for your 2015 Chrysler 200
CARiD.com stores a massive selection of 2015 Chrysler 200 wheels offered
in myriads of design and finish options, including chrome, black, silver,
and so much more.
"""

prompts = [
        f"<s>[INST] {new_medium_prompt}"+new_original_text +" [/INST]",
]
# prompts=[medium_prompt+"\n"+original_text]

#prompts=[f"<s>[INST] {system_prompt} [/INST]"+

#prompts=[code_prompt]

sampling_params = SamplingParams(temperature=0.8,max_tokens=800)

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2",gpu_memory_utilization=0.7)
# llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(generated_text)


    #show c4 dataset
# from datasets import load_dataset

# # 加载数据集
# dataset = load_dataset("stas/c4-en-10k") #This is a small subset representing the first 10K records of the original C4 dataset

# #four types of prompt
# easy_prompt="A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the questions.\nUSER: For the following paragraph give me a paraphrase of the same using a very small vocabulary and extremely simple sentences that a toddler will understand:"

# # hard_prompt="A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the questions.\nUSER:  For the following paragraph give me a paraphrase of the same using very terse and abstruse language that only an erudite scholar will understand. Replace simple words and phrases with rare and complex ones:\n"

# medium_prompt="A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the questions.\nUSER:  For the following paragraph give me a diverse paraphrase of the same in high quality English language as in sentences on Wikipedia:\n"

# # QA_prompt="A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the questions.\nUSER:  Convert the following paragraph into a conversational format with multiple tags of \"Question:\" followed by \"Answer:\":\n"

# current_prompt=medium_prompt

# # Create prompts array
# prompts = []
# for i in range(10):
#     prompts.append(current_prompt + dataset["train"][i]["text"])

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=300)

# llm = LLM(model="/home/yuanzhe/code/LMFlow/output_models/finetuned_phi2_3")#two gpus

# outputs = llm.generate(prompts, sampling_params)

# # 获取当前时间
# current_time = datetime.now()

# # 格式化为所需的字符串格式
# formatted_time = current_time.strftime("%m-%d %H:%M:%S")

# # 打印格式化后的时间
# print(f"INFO {formatted_time}")

# # Create a list to store the results
results = []

# Iterate through the outputs and store in the results list
for output in outputs:
    print(output.outputs[0].text)
    generated_text = output.outputs[0].text
    results.append({"text": generated_text})

# Save the results to a JSON file
output_file_path = "generated_text_results2.json"
with open(output_file_path, "w") as json_file:
    json.dump(results, json_file)

# print(f"Results saved to {output_file_path}")
