# 🧾 Invoice Classification Task: Findings & Analysis

## 1. Introduction

This project aimed to classify invoice items into main segments and their respective families. The classification was based on the following segments:

- Healthcare
- Clothing
- Food/Beverage
- Home Appliances
- Electrical Supplies

**Newly Added Segments:**
- Tools/Equipment
- Computing
- Arts/Crafts/Needlework
- Music
- Camping

We evaluated a variety of classification methods, including few-shot learning with **Gemma** and fine-tuning with **Qwen**, to determine optimal performance for both segment and family classification.

---

## 2. Data Collection

Data sources included:
- Noon
- Amazon
- Carrefour
- Lulu
- Tamimi
- Ubuy

**Notes:**
- Scripts were used to automate data collection from Noon and Lulu.
- Manual data was collected for complex or unstructured categories.
- A labeled evaluation set was prepared for consistent benchmarking.

---

## 3. Methodology

### Few-shot Classification
Performed using `gemma2-9b` across several configurations:
- Balanced vs. unbalanced data
- With and without product descriptions
- Hierarchical classification
- English, Arabic, and bilingual input

### Fine-tuning
Performed using multiple versions of `Qwen 2.5`:
- 1.5B, 3B, and 7B model sizes
- Various training steps and language setups

---

## 4. Experiments & Results

### 4.1 Few-shot Classification (Gemma2-9b on CPU)

| Configuration | Inference Time | Segment Accuracy | Segment F1 | Family Accuracy | Family F1 |
|---------------|----------------|------------------|------------|------------------|-----------|
| Balanced + Description (EN) | 14.73s | 0.946 | 0.946 | 0.730 | 0.730 |
| Balanced (EN) | 10.71s | 0.9737 | 0.9737 | 0.6842 | 0.6842 |
| Hierarchical (EN + AR) | ~4.40s | 0.945 | 0.945 | 0.755 | 0.755 |
| All Segments + Families (EN + AR) | ~2.47s | 0.936 | 0.936 | 0.673 | 0.673 |
| Hierarchical (EN only) | 4.33s | 0.948 | 0.948 | 0.776 | 0.776 |
| Hierarchical (AR only) | 4.34s | 0.962 | 0.962 | 0.740 | 0.740 |
| All Segments + Families (EN) | 2.47s | 0.922 | 0.922 | 0.716 | 0.716 |
| All Segments + Families (AR) | 2.46s | 0.952 | 0.952 | 0.663 | 0.663 |
| 10 Segments (EN + AR) | 9.17s | 0.8749 | 0.9029 | 0.632 | 0.632 |

---

### 4.2 Fine-tuning (Qwen 2.5)

| Model | Segment Accuracy | Segment F1 | Family Accuracy | Family F1 |
|-------|------------------|------------|------------------|-----------|
| Qwen 2.5 3B (full data) | 94.46 | 94.61 | 0.60 / 0.57 | 0.59 / 0.56 |
| Qwen 2.5 1.5B (full data) | 72.82 | 71.35 | - | - |
| Qwen 2.5 7B (EN/AR, +400 steps) | - | - | 61.4 / 59.2 | 61.4 / 59.2 |
| Qwen 2.5 3B (AR/EN, train data) | - | - | 0.326 | 0.31 |
| Qwen 2.5 7B (EN/AR, +350 steps) | - | - | 61.2 / 58.7 | 61.2 / 58.7 |

---

## 5. Hierarchical Classification

Best-performing method: **Segment-first classification**, where the model classifies the segment first, then narrows down to families only within that segment.

Other methods tried:
- Full list (segment + family together)
- Predicted segment (segment predicted first, then family – less reliable)

---

## 6. Conclusion & Recommendations

- **Gemma (few-shot)** provided the best results overall, especially with descriptions.
- **Qwen (fine-tuned)** models approached similar performance for segments, with room for improvement in family classification.
- **Hierarchical (segment-first)** methods significantly reduced misclassifications.

**Future Work:**
- Expand training dataset for Qwen
- Experiment with hyperparameters
- Fine-tune models like **Mistral 7B**
- Try **Qwen 32B** and **Mistral Saba** for few-shot setups
- Improve data balance across underrepresented families
- Explore prompt engineering, especially for Arabic inputs


