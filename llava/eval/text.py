# from rouge_score import rouge_scorer
# from quilt_utils import *

# scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# # scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# candidate_summary = "the cat was found under the bed"
# reference_summary = "the cat was under the bed"

# # candidate_summary = normalize_word(candidate_summary)
# # reference_summary = normalize_word(reference_summary)

# scores = scorer.score(reference_summary, candidate_summary)
# print(scores['rougeL'].fmeasure)
# # for key in scores:
# #     print(f'{key}: {scores[key]}')




from nltk.translate.meteor_score import meteor_score
from quilt_utils import *
 
reference3 = '我 说 这 是 怎 么 回 事，原 来 明 天 要 放 假 了'.split()
reference2 = '我 说 这 是 怎 么 回 事'.split()
hypothesis2 = '我 说 这 是 啥 呢 我 说 这 是 啥 呢'.split()
# reference3：参考译文
# hypothesis2：生成的文本

# reference3 = normalize_word(reference3)
# reference2 = normalize_word(reference2)

# hypothesis2 = normalize_word(hypothesis2)

res = round(meteor_score([reference3], hypothesis2), 4)
print(res)
 
# output:
# 0.4725


from RaTEScore import RaTEScore

pred_report_ori = 'in the image, the positively charged dna strands are holding the negatively charged dna strands together, allowing for the compaction of the dna. this is a result of the electrostatic interaction between the positively charged dna strands and the negatively charged dna strands. the positively charged dna strands repel each other, while the negatively charged dna strands attract each other, creating a stable structure that helps maintain the integrity and stability of the dna.'

gt_report_ori = 'the histone subunits'


# ['in the image, the positively charged dna strands are holding the negatively charged dna strands together, allowing for the compaction of the dna. this is a result of the electrostatic interaction between the positively charged dna strands and the negatively charged dna strands. the positively charged dna strands repel each other, while the negatively charged dna strands attract each other, creating a stable structure that helps maintain the integrity and stability of the dna.']
# ['the histone subunits']

assert len(pred_report_ori) == len(pred_report_ori)

# pred_report = [''.join(pred_report_ori)]
# gt_report = [''.join(gt_report_ori)]

pred_report=[]
gt_report=[]
pred_report.append(pred_report_ori)
gt_report.append(gt_report_ori)
ratescore = RaTEScore()
# Add visualization_path here if you want to save the visualization result
# ratescore = RaTEScore(visualization_path = '')

scores = ratescore.compute_score(pred_report, gt_report)
print(scores)