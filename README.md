<div align="center">

# 🦅 EAGLE: Expert‑Guided Self‑Enhancement for Preference Alignment in Pathology


[![Paper](https://img.shields.io/badge/Paper-ACL_2025-blue.svg)](https://aclanthology.org/2025.acl-long.711.pdf)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/Nandzy/EAGLE)


<p align="center">
  <strong>Expert-Guided Self-Enhancement for Preference Alignment in Pathology Large Vision-Language Models</strong>
</p>

![EAGLE Framework](./assets/method-pipeline.jpg)

</div>

---

## 📖 Overview

**EAGLE** is a novel three-stage alignment framework designed to enhance Large Vision-Language Models (LVLMs) in the pathology domain. By leveraging expert-guided self-enhancement and automated preference data generation, EAGLE significantly improves model performance without relying on expensive human annotation.

### The Three-Stage Pipeline

1.  **Supervised Fine-Tuning (SFT)**
    * Fine-tunes a base LVLM (e.g., Vicuna + QuiltNet) using pathology-specific instruction data to establish a strong baseline foundation.

2.  **Self-Preference Creation (Zero Human Annotation)**
    Generates high-quality preference pairs through automated sampling:
    * **✅ Chosen Sampling:** Prompted outputs guided by expert knowledge to ensure clinical relevance.
    * **❌ Rejected Sampling:** Constructed via controlled corruptions to teach the model what *not* to do:
        * *Entity Replacement:* utilizing medical Named Entity Recognition (NER).
        * *Visual Masking:* simulating mislocalization or occlusion of critical image regions.

3.  **Preference-Following Tuning (PFT)**
    * Applies **Direct Preference Optimization (DPO)** to align the model with the generated preference pairs.
    * Supports **multi-round tuning** for iterative refinement of faithfulness and localization.

---

## 💡 Key Contributions

* 🚀 **Scalable Framework:** Introduces an expert-guided self-enhancement pipeline that scales efficiently.
* 📉 **Minimal Cost:** Constructs pathology-specific preference pairs **without human annotation**.
* 🏥 **Clinical Accuracy:** Significantly improves **faithfulness**, **factual accuracy**, and **localization ability** in pathology VQA tasks.

---

## ⚙️ Installation

Follow these steps to set up the environment:

1. **Clone the Repository**
```bash
git clone [https://github.com/meidandz/EAGLE.git](https://github.com/meidandz/EAGLE.git)
cd EAGLE
```

2. **Create and activate Python virtual environment**
    ```bash
    conda create -n eagle python=3.10 -y
    conda activate eagle
    ```

3. **Install package**
    ```bash
    pip install --upgrade pip  # upgrade pip for best compatibility
    pip install -e .
    ```

4. **(Optional) Install training dependencies**
    ```bash
    pip install -e ".[train]"
    pip install flash-attn --no-build-isolation
    ```


## 📎 Citation

```bibtex
@inproceedings{ding2025eagle,
  title={EAGLE: Expert-Guided Self-Enhancement for Preference Alignment in Pathology Large Vision-Language Model},
  author={Ding, Meidan and Zhang, Jipeng and Wang, Wenxuan and Zhong, Haiqin and Wang, Xiaoqin and Lyu, Xinheng and Chen, Wenting and Shen, Linlin},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={14603--14619},
  year={2025}
}
