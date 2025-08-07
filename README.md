# EAGLE: Expert‑Guided Self‑Enhancement for Preference Alignment in Pathology Large Vision‑Language Models
[📄 ACL 2025 Paper](https://aclanthology.org/2025.acl-long.711.pdf)
EAGLE is a three-stage alignment framework designed to enhance large vision-language models (LVLMs) in pathology by leveraging expert-guided self-enhancement preference data.

---

## 🔍 Overview

### 1. Supervised Fine-Tuning (SFT)

Instruction-tune a base LVLM (e.g., Vicuna + QuiltNet) using pathology instruction data.

### 2. Self-Preference Creation

- **Chosen Sampling**: Prompted outputs guided by expert knowledge.
- **Rejected Sampling**: Informed by:
  - Medical NER for entity replacements.
  - Visual masking (e.g., mislocalized regions).

### 3. Preference Following-Tuning (PFT)

Apply DPO to align the model with preferred outputs and penalize rejected ones.
Supports multi-round tuning for performance refinement.

---
