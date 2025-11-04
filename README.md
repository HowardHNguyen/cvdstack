🫀 CVDStack: AI-Driven Cardiovascular Disease Prediction with Stacking Generative AI
Live App: cvdstack.streamlit.app
Repository: GitHub – HowardHNguyen/cvdstack
Creator: Dr. Howard Nguyen, PhD — Data Science & AI, Healthcare Analytics
💡 Overview
CVDStack is a full-stack Generative AI + Machine Learning platform that predicts cardiovascular disease risk with medical-grade accuracy.
Built as part of my doctoral dissertation — “Advancing Heart Failure Prediction: A Comparative Study of Traditional Machine Learning, Neural Networks, and Stacking Generative AI Models” — this project demonstrates how stacked ensemble AI can outperform both classical ML and single deep-learning models in real-world healthcare data.
The model integrates Generative Adversarial Networks (GANs), Random Forest (RF), Gradient Boosting (GBM), and Convolutional Neural Networks (CNNs) into a unified architecture, improving minority-class representation and predictive reliability.
🧠 Motivation
Heart disease remains the leading global cause of death, often driven by silent risk factors hidden in routine clinical data.
Traditional models capture linear relationships but miss subtle, nonlinear interactions.
CVDStack bridges this gap by generating balanced synthetic data (via GANs) and stacking multiple AI learners to achieve deeper, more interpretable insights.
⚙️ Features
•	🧬 Stacking Generative AI Model: Combines GAN + RF + GBM + CNN to enhance learning from imbalanced datasets.
•	📊 Predictive Dashboard: Real-time ROC AUC, accuracy, and precision/recall summaries.
•	🩺 Explainable AI: SHAP-based feature importance to identify clinical drivers (e.g., BMI, sysBP, totChol).
•	🧩 Data Balancing: SMOTE and CTGAN integration for minority-class synthesis.
•	💾 Scalable Pipeline: Handles datasets from 303 to 400 K records seamlessly.
•	🌐 Deployment: Live Streamlit app connected to GitHub root-level Python files (no sub-dirs).
🧩 Methodology
1.	Data Engineering
o	Pre-processed demographic & clinical variables (sex, age, BP, cholesterol, glucose, BMI, smoking).
o	Handled class imbalance using SMOTE and GAN-based data augmentation.
2.	Model Training
o	Trained Random Forest, GBM, XGBoost, CNN, and Generative AI models individually.
o	Integrated them via a Stacking Classifier meta-learner (Logistic Regression).
o	Calibrated predictions using Isotonic Regression to ensure probability reliability.
3.	Evaluation & Validation
o	Cross-validation AUC ≈ 0.99, accuracy ≈ 0.97 on 400 K records.
o	Comparative experiments vs. Logistic Regression, RF, GBM, and CNN confirmed superiority.
o	Avoided overfitting via permutation testing and hold-out validation.
4.	Deployment & Interpretation
o	Streamlit UI built with Plotly and matplotlib for visual diagnostics.
o	Displays model predictions, feature importance, and interactive threshold adjustment.
📈 Results Snapshot
Model	Accuracy	ROC AUC	Key Finding
Logistic Regression	0.59 – 0.68	0.64 – 0.73	Limited by non-linear interactions
Random Forest	0.70 – 0.81	0.63 – 0.90	Strong but sensitive to imbalance
GBM / XGBoost	0.72 – 0.86	0.62 – 0.97	High variance without balancing
CNN / RNN	0.80 – 0.95	0.80 – 0.99	Excellent pattern learning
Stacking Gen AI (ours)	0.97 +	0.99 +	Best overall generalization

🧰 Tech Stack
Layer	Tools / Libraries
Language	Python 3.12 + Google Colab (T4 GPU)
ML / AI	scikit-learn · XGBoost · LightGBM · TensorFlow/Keras · PyTorch · CTGAN · SMOTE
Visualization	Plotly · matplotlib · Streamlit UI
Deployment	Streamlit Cloud + GitHub (main branch root files)
Environment Control	scikit-learn==1.6.1 · joblib==1.4.2 · lightgbm==4.5.0 · xgboost==2.1.1

🚀 How to Run Locally
# Clone repository
git clone https://github.com/HowardHNguyen/cvdstack.git
cd cvdstack

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

📊 Live Dashboard Highlights
•	Model Summary: Displays AUC, Accuracy, Precision, Recall per model.
•	Feature Importance: Highlights clinical drivers (BMI, sysBP, glucose, totChol).
•	Threshold Tuning: Adjust decision threshold to balance sensitivity vs specificity.
•	Data Upload: Supports custom CSV files for external validation.
•	Interpretability Layer: Explains prediction scores for clinician transparency.
🧬 Research Significance
This project proves that Stacking Generative AI can achieve medical-grade predictive accuracy and enhanced fairness for underrepresented patient groups.
It contributes to next-generation AI Health Diagnostics and the vision of AICardioHealth Inc., a startup focused on AI-driven cardiovascular prevention.
🏁 Summary
CVDStack = Generative AI + Stacked Learning + Explainable Healthcare.
It moves beyond prediction to personalized intervention — turning clinical data into life-saving insights.
Predict early. Explain clearly. Act precisely.
🧾 License
© 2025 Howard Nguyen (MaxAIS · AICardioHealth)
💬 Contact
📧 info@howardnguyen.com
🌐 www.maxais.com
🔗 LinkedIn @HowardHNguyen

<img width="468" height="624" alt="image" src="https://github.com/user-attachments/assets/4301eb2e-f0ce-4a19-aa83-70c216fd89cc" />
