# Virtual Care Assistant for Home Hospitalization
## Project Title

Virtual Care Assistant for Home Hospitalization

## Project Description

This project focuses on developing an NLP-based system capable of understanding patient-reported symptoms in natural, non-medical language. The goal is to extract medically relevant information, assign triage levels, and provide real-time recommendations in the context of home hospitalization, where no clinician is physically present.

The system addresses challenges such as:
- Identifying implicit health cues in everyday language  
- Interpreting informal symptom descriptions  
- Integrating textual data with sensor-based health monitoring  

Key NLP tasks involved:
- Named Entity Recognition (NER) for symptom extraction  
- Symptom normalization and semantic parsing  
- Text classification (e.g., risk levels or disease labels)  
- Natural Language Generation (NLG) for recommendations  

## Datasets Used

1. **Symptom-Based Disease Labeling Dataset** (Kaggle)  
   - Free-text symptom reports paired with disease labels  
   - Used for training NLP models to extract and classify symptoms  

2. **Health Monitoring System Dataset** (Kaggle)  
   - Simulated wearable sensor data (e.g., heart rate, temperature)  
   - Used to supplement text inputs with physiological indicators  

## Model and Prompting

We used a combination of traditional NLP techniques and LLM-based prompting.  
Examples include:
- Extracting symptoms in structured JSON format  
- Generating personalized patient recommendations based on symptoms and vitals  
- Producing clinical summaries with triage assessment  

## Evaluation

Closed-World Evaluation:
- Metrics: Precision, Recall, F1 for symptom extraction and classification  
- Baseline: Rule-based symptom detection

Open-World Evaluation:
- Metrics: BLEU, ROUGE, or LLM-based scoring for text generation  
- Baseline: Template-based recommendations  
