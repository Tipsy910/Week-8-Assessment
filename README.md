# Week-8-Assessment
                                    Model  Accuracy     AUC  Recall   Prec.  \
lr                    Logistic Regression    0.7689  0.8047  0.5602  0.7208
ridge                    Ridge Classifier    0.7670  0.0000  0.5497  0.7235
lda          Linear Discriminant Analysis    0.7670  0.8055  0.5550  0.7202
rf               Random Forest Classifier    0.7485  0.7911  0.5284  0.6811
nb                            Naive Bayes    0.7427  0.7955  0.5702  0.6543
gbc          Gradient Boosting Classifier    0.7373  0.7914  0.5550  0.6445
ada                  Ada Boost Classifier    0.7372  0.7799  0.5275  0.6585
et                 Extra Trees Classifier    0.7299  0.7788  0.4965  0.6516
qda       Quadratic Discriminant Analysis    0.7282  0.7894  0.5281  0.6558
lightgbm  Light Gradient Boosting Machine    0.7133  0.7645  0.5398  0.6036
knn                K Neighbors Classifier    0.7001  0.7164  0.5020  0.5982
dt               Decision Tree Classifier    0.6928  0.6512  0.5137  0.5636
dummy                    Dummy Classifier    0.6518  0.5000  0.0000  0.0000
svm                   SVM - Linear Kernel    0.5954  0.0000  0.3395  0.4090

              F1   Kappa     MCC  TT (Sec)
lr        0.6279  0.4641  0.4736     0.334
ridge     0.6221  0.4581  0.4690     0.004
lda       0.6243  0.4594  0.4695     0.004
rf        0.5924  0.4150  0.4238     0.024
nb        0.6043  0.4156  0.4215     0.005
gbc       0.5931  0.4013  0.4059     0.018
ada       0.5796  0.3926  0.4017     0.014
et        0.5596  0.3706  0.3802     0.021
qda       0.5736  0.3785  0.3910     0.004
lightgbm  0.5650  0.3534  0.3580     0.053
knn       0.5413  0.3209  0.3271     0.159
dt        0.5328  0.3070  0.3098     0.007
dummy     0.0000  0.0000  0.0000     0.006
svm       0.2671  0.0720  0.0912     0.007

--- ML Pipeline Completed: Metadata Saved to model_metadata.json ---

    🤖 [AI Agent Report]
    จากการประเมินโมเดล LogisticRegression
    พบว่ามีค่า Accuracy อยู่ที่ 76.89% และ F1-Score 62.79%
    สถานะผลลัพธ์: Ready for Deployment
