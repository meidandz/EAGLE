import argparse
import os
import json
import pprint
import collections
import random
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate
from quilt_utils import *
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from RaTEScore import RaTEScore
import numpy as np
import evaluate



import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--quilt', type=bool, default=False, help='whether to evaluate on quilt outputs')
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--pred_file_parent_path', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--anchor', type=str, default=None, help='path to anchor prediction file, unused except for eval of lengthy preds', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 

def evaluate(gt, pred, quilt=False, anchor=None):  
    import evaluate
    rouge = evaluate.load('rouge')
    ratescore = RaTEScore()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)  
    closed_scores2 = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    token_edit_distance_scores = collections.defaultdict(list)
    rouge_scores = collections.defaultdict(list)
    rougeL_scores = collections.defaultdict(list)
    meteor_scores = collections.defaultdict(list)
    rate_scores = collections.defaultdict(list)



    for gt_item, pred_item, anchor_item in zip(gt, pred, anchor if anchor else pred):
        gt_value_ori = gt_item['answer'].lower()
        pred_value_ori = pred_item['text'].lower()
        anchor_value_ori = anchor_item['text'].lower()
        
        
        gt_value = normalize_word(gt_value_ori)
        pred_value = normalize_word(pred_value_ori)
        anchor_value = normalize_word(anchor_value_ori)

        pred_value = pred_value[:len(anchor_value)]

        if gt_item['answer_type'] == 'OPEN' or gt_item['answer_type'] == 'other':
            # for open-ended question
            exact_scores['hit'].append(calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(pred_item['question_id'])


            f1_score, precision, recall, acc = calculate_f1score(pred_value, gt_value)
            token_distance = token_edit_distance(pred_value, gt_value)

            
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['acc'].append(acc)
            f1_scores['q_id'].append(pred_item['question_id'])
            token_edit_distance_scores['hit'].append(token_distance)

            rouge_scores['precision'].append((scorer.score(pred_value, gt_value))['rougeL'].precision)
            rouge_scores['recall'].append((scorer.score(pred_value, gt_value))['rougeL'].recall)
            rouge_scores['fmeasure'].append((scorer.score(pred_value, gt_value))['rougeL'].fmeasure)
            
            meteor_scores['hit'].append(round(meteor_score([gt_value.split()], pred_value.split()), 4))
            
            




            pred_value_ori1=[]
            gt_value_ori1=[]
            pred_value_ori1.append(pred_item['text'].lower())
            gt_value_ori1.append(gt_item['answer'].lower())

            rougeL_scores['hit'].append(rouge.compute(predictions=pred_value_ori1, references=gt_value_ori1)['rougeL'])

            # types = [print(gt) for gt in gt_value_ori1]
            # # print(types)
            # print(len(np.array(gt_value_ori1).shape))
            # print(type(gt_value_ori1))
            
            try:
                rate = ratescore.compute_score(pred_value_ori1, gt_value_ori1)[0]
                print(rate)
                # quit()
                rate_scores['hit'].append(rate) 
                # print(rate_scores)
            except:
                continue
            
        



            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))
            
            bleu_scores['q_id'].append(pred_item['question_id'])
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)

        elif gt_item['answer_type'] == 'CLOSED':
            # for close-ended question (Yes/No)
            closed_scores2['q_id'].append(pred_item['question_id'])

            if quilt:
                gt_value = gt_item['yes_no_answer'].lower()

            assert gt_value in ['yes', 'no'], f"assert gt_value in : {pred_item['question_id'], gt_value}"
            answer = gt_value
            # Only keep the first sentence
            #if pred_value.find('.') != -1:
            #    pred_value = pred_value.split('.')[0]

            pred_value = pred_value.replace(',', '')
            # print(pred_value)
            words = pred_value.split(' ')
            # print(words)
            if 'No' in words or 'not' in words or 'no' in words:
                pred_answer = 'no'
            else:
                pred_answer = 'yes'
            
            if pred_answer == answer:
                closed_scores2['hit'].append(1)
            else:
                closed_scores2['hit'].append(0)
                
    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])
    f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
    precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
    recall = sum(f1_scores['recall']) / len(f1_scores['recall'])
    open_acc = sum(f1_scores['acc']) / len(f1_scores['acc'])
    closed_score2 = sum(closed_scores2['hit']) / len(closed_scores2['hit']) if len(closed_scores2['hit']) != 0 else 0.0
    bleu_score = sum(bleu_scores['bleu_score'])/len(bleu_scores['bleu_score'])
    bleu_score_1 = sum(bleu_scores['bleu_score_1'])/len(bleu_scores['bleu_score_1'])
    bleu_score_2 = sum(bleu_scores['bleu_score_2'])/len(bleu_scores['bleu_score_2'])
    bleu_score_3 = sum(bleu_scores['bleu_score_3'])/len(bleu_scores['bleu_score_3'])
    token_edit_distance_score = sum(token_edit_distance_scores['hit'])/len(token_edit_distance_scores['hit'])
    rouge_score_precision = sum(rouge_scores['precision'])/len(rouge_scores['precision'])
    rouge_score_recall = sum(rouge_scores['recall'])/len(rouge_scores['recall'])
    rouge_score_f1 = sum(rouge_scores['fmeasure'])/len(rouge_scores['fmeasure'])
    rougeL_score = sum(rougeL_scores['hit'])/len(rougeL_scores['hit'])
    meteor_scor = sum(meteor_scores['hit'])/len(meteor_scores['hit'])
    rate_score = sum(rate_scores['hit'])/len(rate_scores['hit'])



    return tabulate(
        [
            ['exact match score', exact_score*100], 
            ['f1 score', f1_score*100], 
            ['precision', precision*100], 
            ['recall', recall*100], 
            ['yes/no accuracy', closed_score2*100],
            ['open_acc', open_acc*100],
            ['bleu_score', bleu_score*100],
            ['bleu_score_1', bleu_score_1*100],
            ['bleu_score_2', bleu_score_2*100],
            ['bleu_score_3', bleu_score_3*100],
            ['token_edit_distance', token_edit_distance_score],
            ['rouge_precision', rouge_score_precision*100],
            ['rouge_recall', rouge_score_recall*100],
            ['rouge_f1', rouge_score_f1*100],
            ['rougeL_score', rougeL_score*100],
            ['meteor', meteor_scor*100],
            ['rate_score', rate_score*100],
        ], 
        headers=['Metric', 'Performance']
    )


if __name__ == '__main__':
    args = parse_option()

    gt = json.load(open(args.gt, 'r'))
    pred = load_jsonl(args.pred)
    if args.anchor:
        anchor = load_jsonl(args.anchor)
        anchor_ids = [item['question_id'] for item in anchor]

    gt_ids = [item['id'] for item in gt]
    pred_ids = [item['question_id'] for item in pred]
    
    assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results = evaluate(gt, pred, quilt=args.quilt, anchor=anchor if args.anchor else None)
    pprint.pprint(results)
