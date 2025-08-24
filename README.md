#  Stacking Ensemble Model for Mental Illness Prediction

This project implements a stacking ensemble machine learning model to predict mental illness using multiple base learners combined with a meta-learner.  
It also provides a **Streamlit app** for easy interaction and visualization.

---

##  Features
- Data preprocessing and cleaning pipeline  
- Multiple base ML models (Logistic Regression, Random Forest, etc.)  
- Stacking ensemble with meta-learner  
- Model evaluation with performance metrics  
- Interactive **Streamlit web app**  

---

##  Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/arunchinega/Stacking-Ensemble-Model-for-Mental-Illness-Prediction.git
cd Stacking-Ensemble-Model-for-Mental-Illness-Prediction
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
 Stacking-Ensemble-Model-for-Mental-Illness-Prediction
 ┣ 📂 app              # Streamlit application
 ┣ 📂 data             # Dataset files
 ┣ 📂 models           # Saved trained models
 ┣ 📂 src              # Source code for preprocessing, training, evaluation
 ┣ requirements.txt    # Dependencies
 ┗ README.md           # Project documentation

 Paste this **entire block** into your README editor, delete everything else, and then hit **Commit changes**.  

After that, your repo’s homepage will look **clean, professional, and polished** .  

Do you want me to also add a **Results/Example Output** section (like screenshots of the Streamlit app or accuracy metrics) so people instantly see the model’s performance?
##  Results

The stacking ensemble model significantly outperformed simpler models by improving recall for the minority classes while maintaining strong overall accuracy.

### Comparative Performance of Models
| Model                 | Overall Accuracy | Macro F1 Score | NO Recall | UNKNOWN Recall |
|------------------------|------------------|----------------|-----------|----------------|
| Random Forest          | >95%             | <0.50          | ≈ 0.00    | ≈ 0.00         |
| KNN                    | >95%             | <0.50          | ≈ 0.00    | ≈ 0.00         |
| PCA / Factor Analysis  | >95%             | <0.50          | ≈ 0.00    | ≈ 0.00         |
| **XGBoost**            | **96.79%**       | **0.677**      | **63%**   | **45%**        |
| **Soft Voting (Female)** | **97%**        | **0.677**      | **70%**   | **81%**        |
| **Stacking Ensemble (Male)** | **96.5%**  | **0.61**       | **26%**   | **73%**        |
| Logistic Regression (Meta) | **97.3%**    | **0.744**      | **33%**   | **78%**        |
| **Ridge Classifier (Meta)** | **97.4%**   | **0.756**      | **28%**   | **81%**        |
| Lasso Classifier (Meta) | **96.1%**       | **0.327**      | **0%**    | **0%**         |

### Final Unified Stacking Ensemble
- **Overall Accuracy:** 97%  
- **Macro F1 Score:** 0.71  
- **Recall (NO class):** 25%  
- **Recall (UNKNOWN class):** 77%  

The final stacking ensemble demonstrated **balanced prediction across all classes**, outperforming simpler models that only appeared strong due to class imbalance.
