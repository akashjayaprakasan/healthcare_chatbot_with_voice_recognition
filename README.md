# AI Healthcare Chatbot with Voice Recognition

An AI-powered healthcare chatbot that predicts possible diseases from user symptoms. It supports both text and voice input and provides health advice with voice output.

> This project is for educational purposes only. It is not a replacement for professional medical advice.

## Features

- Text-based symptom input
- Voice symptom input using Speech Recognition
- Disease prediction using Machine Learning
- Random Forest classifier
- Confidence score for prediction
- Health advice and precautions
- Medicine and food recommendations
- Emergency warning messages
- Text-to-Speech response
- Responsive web interface

## 🛠️ Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Scikit-learn
- Pandas
- Joblib
- Speech Recognition
- Text-to-Speech
- Random Forest Algorithm

## Project Structure

```text
AI-Healthcare-Chatbot/
│
├── app.py
├── train.py
├── chatbot.py
├── requirements.txt
│
├── data/
│   └── symptoms_dataset.csv
│
├── model/
│   └── disease_classifier.pkl
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
