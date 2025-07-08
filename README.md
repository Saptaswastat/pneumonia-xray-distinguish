
#  Pneumonia Detection with Transfer Learning

Detect pneumonia vs. normal chest X-ray on PneumoniaMNIST using pretrained Inception-V3 features.

---

##  Dataset
- **PneumoniaMNIST** (part of MedMNIST collection)
- original data pneumoniamnist.npz
- Preprocessed into CSVs:  
  - `pneumoniamnist_train.csv`  
  - `pneumoniamnist_val.csv`  
  - `pneumoniamnist_test.csv`

---

##  Method
- Extracted **2048-dimensional deep features** from pretrained **Inception-V3 (ImageNet)**
- Trained **Random Forest** classifier on these features
- Considered its Precision, accuracy and recall
- Created advanced visualizations: ROC curve, feature importance, UMAP, confusion matrix

---

##  3 Appropriate Evaluation Metrics
- **Accuracy**: overall correctness
- **Recall (Sensitivity)**: detecting pneumonia cases correctly as in medical diagnosis, false negatives(missing pneumonia) more rsikier.
- **ROC-AUC**: has discriminative ability and especially useful when class distributions may slightly change.

---

##  How to reproduce

```r
# Install required R packages
# Check requirements.txt 
install.packages(c("randomForest", "ggplot2", "dplyr", "pROC", "umap", "Rtsne", "reshape2", "gridExtra", "RColorBrewer"))

# Run R code
source("scripts/detect_pneumonia_R.R")
