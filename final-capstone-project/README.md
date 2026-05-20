# A Methods Atlas on the Ames Housing Dataset

**MATH 4230 — Applied Statistical Methods for Data Science**  
**Final Capstone Project, Spring 2026**  
**Krrithik Ezhilarasan — California State University, Bakersfield**  
**Instructor: Dr. Anjana Yatawara**

---

## What This Project Is

This capstone applies **thirteen statistical learning methods** from the MATH 4230 syllabus to a single real-world dataset — the [Ames Housing Dataset](https://jse.amstat.org/v19n3/decock.pdf) — and synthesizes the results into a cross-method comparison.

The methods covered:

- Simple and multiple linear regression
- Logistic regression
- Cross-validation and the bootstrap
- Ridge and lasso regularization
- Decision trees (regression and classification)
- Tree ensembles (bagging, random forest, gradient boosting)
- Support vector machines (linear and RBF kernels)
- K-nearest neighbors
- Principal component analysis and principal components regression
- Feedforward neural networks (PyTorch)

The final report is an 83-page document that presents each method in a consistent nine-section format, then synthesizes findings across methods in Chapter 14 and reflects on the project in Chapter 15.

---

## Headline Findings

1. **Eight methods converge on a narrow prediction ceiling.** Test RMSE on log(SalePrice) clusters between **$19,900 and $21,400** across Ridge, Lasso, BIC-selected MLR, full MLR, three neural networks, and gradient boosting — suggesting the ceiling is structural to the data rather than method-specific.

2. **Four independent feature-selection methods agree on the same six predictors.** BIC stepwise selection, Lasso, Random Forest importance, and Gradient Boosting importance all flag the same set as the genuine drivers of price: `Overall Qual`, `Gr Liv Area`, `Year Built`, `Total Bsmt SF`, `Garage Cars`, `Garage Area`.

3. **Ridge regression is the recommended deployment model.** Test RMSE of $19,900, training time under three seconds, fully interpretable coefficients — the best combination of accuracy, interpretability, and operational simplicity in the report.

---

## Repository Contents

```
final-capstone-project/
├── README.md                           This file
├── requirements.txt                    Python package versions
├── data/
│   └── AmesHousing.csv                 Raw Ames dataset (2,930 rows × 82 cols)
├── notebook/
│   └── Final_capstone_proj_notebook.ipynb    Full reproducible notebook
├── report/
│   └── A_Methods_Atlas_on_the_Ames_Housing_Dataset.pdf
│                                       Rendered 83-page PDF report
└── no_header/                          Custom nbconvert LaTeX template
    ├── conf.json                       used to suppress the auto-header on PDF
    └── index.tex.j2                    export
```

---

## How to Reproduce

### Requirements

- Python 3.11 or higher
- A LaTeX installation if rebuilding the PDF (MiKTeX on Windows, TeX Live on macOS/Linux)
- Optionally a GPU for the neural network chapter, though the notebook runs end-to-end on CPU in 5–7 minutes

### Setup

```bash
# Clone the repository
git clone https://github.com/Krrithik/applied-stats-data-science.git
cd applied-stats-data-science/final-capstone-project

# Install dependencies
pip install -r requirements.txt
```

### Run the notebook

Open `notebook/Final_capstone_proj_notebook.ipynb` in Jupyter Lab or Jupyter Notebook. Make sure `AmesHousing.csv` is in the working directory (either copy it from `data/` or update the read path in Chapter 2).

```bash
jupyter notebook notebook/Final_capstone_proj_notebook.ipynb
```

Run all cells top to bottom. Total runtime is approximately 5–7 minutes on a modern laptop without GPU acceleration.

### Rebuild the PDF (optional)

The repository includes the rendered PDF report. To regenerate it from the notebook:

```bash
# From the final-capstone-project/ directory:
jupyter nbconvert --to pdf --no-input \
    --template=no_header \
    --TemplateExporter.extra_template_paths=. \
    notebook/Final_capstone_proj_notebook.ipynb
```

The custom `no_header/` template suppresses the auto-generated header so the project's own title page appears first.

---

## Reproducibility Notes

All results in the report are reproducible from the notebook with `random_state = 42` set globally. Specifically:

- **Train/test split:** 80/20, stratified on `AboveMedian`, producing 2,340 training and 585 test observations.
- **Imputation:** Done on training-set statistics only, after the split, to prevent leakage.
- **Standardization:** Where required (Ridge, Lasso, Logistic, SVM, KNN, PCA, NN), scalers are fit on the training set only.
- **Cross-validation:** 10-fold for most chapters; 5-fold for Chapter 10 (SVM) as a practical concession to training time.

Small numerical differences (R² varying by ±0.01) may appear across runs due to PyTorch's nondeterministic GPU operations or differences in scikit-learn versions. The qualitative findings are stable across these variations.

---

## Dataset

The Ames Housing Dataset was compiled by Dean De Cock as a teaching alternative to the older Boston Housing dataset:

- **Original paper:** De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." *Journal of Statistics Education*, 19(3). [PDF](https://jse.amstat.org/v19n3/decock.pdf)
- **Documentation:** https://jse.amstat.org/v19n3/decock/DataDocumentation.txt
- **Kaggle mirror:** https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset

A copy of the dataset is included in `data/AmesHousing.csv` for full reproducibility.

---

## Acknowledgments

I am grateful to **Dr. Anjana Yatawara** for an excellent semester and for the supplementary deep-learning materials (*Deep Learning: A From-Scratch, Very Gentle Guide* and the accompanying Jupyter lab notebook), which were directly used as the architectural template for the neural network in Chapter 13.

---

## License

This project is submitted as coursework for MATH 4230 at California State University, Bakersfield.

---

## Contact & Further Information

For questions about methodology, results, or reproduction, please refer to the full report:

📄 **Report:** [`report/A_Methods_Atlas_on_the_Ames_Housing_Dataset.pdf`](https://github.com/Krrithik/applied-stats-data-science/blob/main/final-capstone-project/report/A_Methods_Atlas_on_the_Ames_Housing_Dataset.pdf)

📓 **Notebook:** [`notebook/Final_capstone_proj_notebook.ipynb`](https://github.com/Krrithik/applied-stats-data-science/blob/main/final-capstone-project/notebook/Final_capstone_proj_notebook.ipynb)

---

*Last Updated: May 2026*
