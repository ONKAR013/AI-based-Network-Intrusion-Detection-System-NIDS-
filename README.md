---
title: AI NIDS Student Project
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# 🔐 AI-Powered Network Intrusion Detection System (NIDS)

## 📌 Project Overview
This project implements an **AI-based Network Intrusion Detection System (NIDS)** using Machine Learning.  
It analyzes network traffic data and classifies it as **Benign** or **Malicious** to help detect cyberattacks in real time.

The system uses a **Random Forest Classifier** and provides an interactive **Streamlit dashboard** for training, evaluation, and live traffic simulation.

---

## 🎯 Problem Statement
Traditional network security systems are ineffective against evolving cyber threats.  
An AI-based solution is required to detect malicious network traffic accurately and in real time.

---

## 👥 End Users
- Network Administrators  
- Cybersecurity Analysts  
- IT Security Teams  
- Organizations handling sensitive data  
- Educational institutions and research labs  

---

## 🛠️ Technology Used
- **Programming Language:** Python  
- **Machine Learning Algorithm:** Random Forest Classifier  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - Matplotlib  
  - Seaborn  
- **Frontend / Dashboard:** Streamlit  
- **Dataset Type:** CIC-IDS2017-like Network Traffic Data (Simulated)

---

## 🧠 System Architecture / Workflow
1. Network traffic data is collected or simulated  
2. Data preprocessing and feature selection  
3. Dataset is split into training and testing sets  
4. Model training using Random Forest algorithm  
5. Performance evaluation using Accuracy and Confusion Matrix  
6. Live traffic simulation and prediction through Streamlit dashboard  

---

## 📊 Results
- Successfully classifies network traffic as **Benign** or **Malicious**  
- Achieves **high accuracy** using Random Forest  
- Confusion Matrix shows effective attack detection  
- Supports **real-time packet analysis** using live user input  

---

## 🖥️ Features
- Interactive Streamlit dashboard  
- Adjustable training parameters  
- Visual performance metrics  
- Live network traffic simulator

---

## 📌 Conclusion

The AI-Powered Network Intrusion Detection System successfully demonstrates how Machine Learning can be applied to enhance network security.
By automating intrusion detection, the system reduces dependency on manual monitoring and improves response to cyber threats.
This project highlights the practical use of AI in cybersecurity and provides a scalable foundation for future enhancements.
