# Project Title
# Virtual Care Assistant for Home Hospitalization

## Project Overview
This project presents an AI-powered assistant designed to support remote clinical decision-making in home hospitalization settings. The system processes patient-reported symptom descriptions in natural language, along with physiological vital signs, to predict the patient’s current status:
- No Change  
- Improvement  
- Deterioration

The assistant is intended to improve early detection of clinical deterioration in situations where no medical staff is physically present.
![PIC](assets/PHOTO-2025-06-03-01-38-37.jpg)

## Objectives
- Extract clinically relevant information from unstructured text.
- Integrate physiological data to enhance prediction accuracy.
- Build a real-time decision-support model combining NLP and tabular data.

## Dataset (Synthetic)
A synthetic dataset was generated to simulate real-world home hospitalization cases:
- Text: Free-text symptom descriptions written in natural, informal language.
- Vitals: Changes in heart rate, respiratory rate, temperature, and blood pressure (Day 2 – Day 1).
- Labels: Manually assigned clinical outcomes (0 = No Change, 1 = Improvement, 2 = Deterioration).

## Tasks Performed

### Exploratory Data Analysis (EDA)
- Label distribution
- Text length distribution
- Distribution and correlation of vital signs
- Boxplots and violin plots grouped by label
- Word clouds for each class

### Preprocessing
- Text cleaning (punctuation, stopwords, lowercasing)
- Delta feature creation for vitals
- Missing value imputation
- Feature scaling

### Modeling

We experimented with the following model families:

- **Text-only models**
  - TF-IDF + Logistic Regression
  - BERT-based classification

- **Vitals-only models**
  - XGBoost
  - LightGBM

- **Fusion models (Text + Vitals)**
  - Classical ML: XGBoost, LightGBM, Random Forest
  - Neural Networks:
    - Simple NN with BERT embeddings and vitals
    - Improved NN with deeper layers, Batch Normalization, Dropout, and learning rate tuning

## Evaluation Metrics
- Accuracy
- F1 Macro
- Cross-validation for generalization
- Test set evaluation

### Best Performing Model:
- Model: `LGBM_Combined_Test`
- Accuracy: 0.971
- F1 Macro: 0.972

![PIC1](assets/PHOTO-2025-06-03-10-56-49.jpg)

## Visualizations
- Comparison plots of model performance
- Confusion matrices
- Feature distributions and correlations
- Text-based analyses (lengths, word clouds)

## Key Insights
- Fusion of structured (vitals) and unstructured (text) data led to superior performance.
- BERT improved understanding of nuanced symptom descriptions.
- Model enhancements like Batch Normalization and Dropout increased model robustness.

## Future Work
- Use real clinical datasets (e.g., MIMIC-IV, sensor logs).
- Implement Named Entity Recognition (NER) and triage reasoning.
- Integrate the system into a conversational chatbot interface.

## Technologies Used
- Python, Pandas, Scikit-learn
- TensorFlow, Keras
- HuggingFace Transformers (BERT)
- XGBoost, LightGBM
- Seaborn, Matplotlib

## Authors
Gabrielle Maor  
Shay Sason  
HIT – Digital Health Technologies  
NLP & LLM Final Project (2025)
