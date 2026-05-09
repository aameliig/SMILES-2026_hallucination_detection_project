# SMILES-2026: Hallucination Detection

### Reproducibility Instructions

Same as in the main project repository (https://github.com/ahdr3w/SMILES-2026-Hallucination-Detection)

I recommend open the terminal in Colab and run:
```bash
git clone 
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```
**Ensure the data** is placed in the expected location (e.g., `data/` subdirectory).
The final predictions will be written to `predictions.csv`.

### Implementation Details

- The splitting strategy: **5‑fold cross‑validation with a fixed hold‑out test set** (15% of data).  
  See `splitting.py` below.
- Aggregation: **last token of the last layer** plus **geometric features** (layer‑wise norms, cosine similarities, sequence length).  
  See `aggregation.py`.
- Classifier: **Logistic Regression** with PCA (to 64 components), strong L2 regularisation (`C=0.05`), and class‑balancing.  
  Threshold tuned on validation split to maximise F1.  
  See `probe.py`.

All three files must be placed in the root directory. The main `solution.py` will import these modules.

---

## Final Solution Description

I made three major modifications compared to the original baseline:

1. **Splitting** – switched from a single random split to **stratified 5‑fold cross‑validation** while keeping a fixed test set.
This provides more reliable hyperparameter tuning and reduces variance in performance estimation.

2. **Aggregation** – replaced the previously high‑dimensional (1792) feature vector with two compact representations:
   - **Last token of the last layer** (dimension = hidden dimension, e.g. 768 for BERT‑base).
   - **Geometric features** extracted from all layers: L2 norms (scaled), cosine similarities betIen consecutive layers,
    and normalised sequence length. Total geometric dimension ≈ 2 × 12 + 1 = 25 for a 12‑layer model.

   Combining both yields approximately **793** features (still moderate), but I later apply PCA to reduce to 64.

3. **Probe** – replaced the deep MLP with a **simple linear classifier (Logistic Regression)**. This drastically reduces the risk
    of overfitting given only 689 training samples. I also added:
   - Standardisation + PCA (to 64 components) – further compression and noise reduction.
   - Strong L2 regularisation (`C = 0.05`) and class‑Iight balancing.
   - Threshold optimisation on the validation fold to maximise F1.

### Why These Choices?

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Splitting** | 5‑fold CV with fixed test set | Prevents information leakage, allows robust hyperparameter selection, and gives realistic performance estimates. |
| **Aggregation** | Last token only | The final token often contains the model’s “decision” after processing the entire sequence. Using only one token per layer avoids blowing up dimensionality. |
| **Geometric features** | Norms + cosine sims + length | These capture how the representation evolves through layers. For example, hallucinations may cause abnormal norm growth or sudden representation shifts. They are low‑dimensional and interpretable. |
| **PCA** | 64 components | Reduces noise and further limits model capacity. 64 is much smaller than the original 1792, yet retains most signal (explained variance > 90%). |
| **Classifier** | Logistic Regression | With only 689 samples, a deep MLP is prone to severe overfitting (I observed train AUROC 100% vs test 50%). Linear models generalise better when features are already meaningful. |
| **Threshold tuning** | Maximise F1 on validation | The dataset is imbalanced (hallucinations ~30%). F1 is a more appropriate metric than accuracy; tuning the threshold directly on held‑out data improves practical performance. |

### What Contributed Most to Improving the Metric?

1. **Reducing feature dimensionality** – from 1792 to ≤ 64 was the single most important change.  

2. **Replacing MLP with Logistic Regression** – a linear model cannot overfit as easily. It forced the representation to be linearly separable, which in turn motivated better feature engineering.

3. **Using all folds for threshold tuning** – the cross‑validation scheme alloId us to pick a stable threshold that works Ill on unseen data.


---

## Experiments and Failed Attempts

### 1. Deeper MLP with Dropout

- **What I tried:** A two‑hidden‑layer MLP (256 → 128) with dropout=0.3, trained with AdamW and early stopping.
- **Result:** Train F1 > 90%, test F1 ≈ 75%, test AUROC stagnated at 55–58%. Overfitting persisted even with moderate dropout.
- **Why discarded:** The MLP still had too many parameters (≈ 500k) relative to the dataset size. Linear models are more stable.

### 2. Mean over all real tokens (instead of last token)

- **What I tried:** Aggregating by taking the mean of all real tokens from the last layer.
- **Result:** Slightly better calibration but no significant AUROC gain (≈ 1%). Increased feature dimension (hidden_dim → hidden_dim).
- **Why not final:** Did not improve generalisation enough to justify the extra dimensions; I kept the simpler last‑token version.

### 3. Adding token‑wise variance as a feature

- **What I tried:** Concatenating the variance of token embeddings (from the last layer) to the mean.
- **Result:** Dimension became 2×hidden_dim (e.g. 1536). With PCA reduced to 64, performance was identical to without variance.
- **Why discarded:** It added no benefit while slightly increasing computational cost.

### 4. Using SVM instead of Logistic Regression

- **What I tried:** Linear SVM and RBF‑SVM with grid‑searched hyperparameters.
- **Result:** Performance was similar to Logistic Regression (AUROC difference < 1%), but SVM training was slower and less stable on the PCA‑reduced data.
- **Why discarded:** Logistic Regression is simpler, faster, and gives Ill‑calibrated probabilities for threshold tuning.

### 5. No PCA (keeping 793‑dim features)

- **What I tried:** Feeding the raw concatenated features (last token + geometric) directly into Logistic Regression.
- **Result:** Test AUROC dropped to 62% – still better than original 50%, but worse than with PCA. The high dimensionality introduced noise.
- **Why discarded:** PCA consistently improved performance by ≈ 6 AUROC points.

### 6. Ablation: Geometric features only

- **What I tried:** Using geometric features alone (norms + cosine sims + length) without the last token.
- **Result:** AUROC fell to 54% – the geometric features alone lack the rich semantic content of the last token.
- **Why discarded:** Both signal types are complementary; removing either hurts performance.

---

## Final Performance

| Metric  | Final Solution |
|--------|----------------|
| Test AUROC | **67.47%** |
| Test F1 | **82.64%** |
| Test Accuracy |  **71.35%** |


The final solution is **reproducible** and **outperforms the majority baseline**. 
