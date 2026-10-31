# PhishGuard: concept and viva guide

## One-minute explanation
Phishing sites imitate trusted sites. We describe each website with 30 signals coded -1 / 0 / 1, then compare 11 classifiers on three feature strategies to see which detects phishing best, and whether PCA and ensembles help.

## Headline results (this run)
- 11,055 rows -> 5,849 after removing 5,206 duplicates; split 4,679 train / 1,170 test.
- Best by 5-fold CV: **XGBoost on all 30 features, 95.0%**. Held-out test: **95.2% accuracy, 95.7% phishing recall, ROC-AUC 0.993**.
- Ensembles occupy the top ranks; best single model is SVM (RBF) at 93.7%.
- PCA needs 23 of 30 components for 95% variance. It helps Gaussian NB (65% -> 90%), is neutral for SVM / LR / KNN and hurts tree models. Honest finding: PCA did not improve the best model.

## Likely questions
1. **Why remove duplicates?** Copies in train and test leak answers and inflate accuracy.
2. **Why Pipeline?** Scaler/PCA must be fitted on training folds only.
3. **Does PCA always help?** No. It keeps variance, not class information.
4. **Bagging vs boosting vs stacking?** Parallel resampling (less variance) / sequential error fixing (less bias) / a meta-model combining models.
5. **Why recall of the phishing class?** A missed phishing site is costlier than a false alarm. The Test Results page has a threshold slider.
6. **Why did Gaussian NB fail on raw features?** Features are correlated; NB assumes independence. PCA decorrelates them.
7. **Can the model check any URL?** Not directly: most features need page/WHOIS data. The URL Scanner is rule-based and clearly separate.
8. **ROC-AUC?** Probability a random phishing site is ranked above a random legitimate one.
