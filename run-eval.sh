#########################################################################################################
###################################PATHVQA##################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###############################Llava-lora-Quilt-B-32-vicuna######################################
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 python llava/eval/model_vqa.py \
    --model-path ./checkpoints1/Llava-lora-Quilt-B-32-vicuna \
    --question-file ./playground/eval_data/path_vqa/gt/pvqa_test_wo_ans.jsonl \
    --image-folder ./playground/eval_data/path_vqa/images \
    --answers-file ./playground/eval_data/path_vqa/pred/Llava-lora-Quilt-B-32-vicuna_pvqa.jsonl

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 python llava/eval/quilt_eval.py \
    --gt  ./playground/eval_data/path_vqa/gt/pvqa_test_w_ans.json \
    --pred  ./playground/eval_data/path_vqa/pred/Llava-lora-Quilt-B-32-vicuna_pvqa.jsonl





#########################################################################################################
###################################QUILTVQA##################################################################
##########################################################################################################
###########################################################################################################
###############################Llava-lora-Quilt-B-32-vicuna__20k_gt_quiltllava######################################
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python llava/eval/model_vqa.py \
    --model-path ./checkpoints1/Llava-lora-Quilt-B-32-vicuna \
    --question-file ./playground/eval_data/quilt_vqa/gt/quiltvqa_test_wo_ans.jsonl \
    --image-folder ./playground/eval_data/quilt_vqa/images \
    --answers-file ./playground/eval_data/quilt_vqa/pred/Llava-lora-Quilt-B-32-vicuna_quiltvqa.jsonl

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 python llava/eval/quilt_eval.py \
    --quilt True \
    --gt  ./playground/eval_data/quilt_vqa/gt/quiltvqa_test_w_ans.json \
    --pred  ./playground/eval_data/quilt_vqa/pred/Llava-lora-Quilt-B-32-vicuna_quiltvqa.jsonl