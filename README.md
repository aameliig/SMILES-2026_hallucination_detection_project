# SMILES-2026 Hallucination Detection Project

This is a repository for SMILES-2026 Hallucination Detection Project 

https://github.com/ahdr3w/SMILES-2026-Hallucination-Detection

------------------------------------------

### Implementation Details

- The splitting strategy: **5‑fold cross‑validation with a fixed hold‑out test set** (15% of data).  
  See `splitting.py` below.
- Aggregation: **last token of the last layer** plus **geometric features** (layer‑wise norms, cosine similarities, sequence length).  
  See `aggregation.py`.
- Classifier: **Logistic Regression** with PCA (to 64 components), strong L2 regularisation (`C=0.05`), and class‑balancing.  
  Threshold tuned on validation split to maximise F1.  
  See `probe.py`.

All three files must be placed in the root directory. The main `solution.py` will import these modules.


### Reproducibility Instructions

Same as in the main project repository (https://github.com/ahdr3w/SMILES-2026-Hallucination-Detection)

I recommend open the terminal in Colab and run:
```bash
git clone https://github.com/aameliig/SMILES-2026_hallucination_detection_project.git
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py
```
**Ensure the data** is placed in the expected location (e.g., `data/` subdirectory).
The final predictions will be written to `predictions.csv`.

