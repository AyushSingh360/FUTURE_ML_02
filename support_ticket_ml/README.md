# Customer Support Ticket Classification and Priority Prediction System

## 📊 Executive Summary
In fast-growing SaaS companies and high-volume customer service centers, manually reading, categorizing, and routing support tickets is a major bottleneck. It delays First Response Time (FRT) and allows critical issues to slip through the cracks.

This project delivers an automated, end-to-end Machine Learning solution that instantly analyzes incoming support tickets. It automatically:
1. **Categorizes the ticket intent** (e.g., Billing, Technical Issue, Refund)
2. **Predicts the severity/priority** (Low, Medium, High, Critical)

By implementing this system, organizations can optimize support operations, route tickets to the correct specialized agents instantly, and ensure urgent issues (like server outages or payment failures) are flagged for immediate resolution.

---

## 🎯 How Tickets are Categorized
When a customer submits a ticket, the unstructured text goes through our **Natural Language Processing (NLP) pipeline**. 
- The text is cleaned (removing punctuation, numbers, and stop words like "the" or "is").
- Words are reduced to their root forms (Lemmatization).
- The text is converted into a numerical matrix using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which highlights the most significant keywords in the ticket (e.g., "refund", "crash", "invoice").

The categorized intent is then predicted using a trained **Multinomial Naive Bayes / Random Forest** classifier. The ticket is grouped into actionable categories such as:
- **Billing inquiry**
- **Technical issue**
- **Refund request**
- **Product inquiry**
- **Cancellation request**

*Business Value:* Automatically categorizing tickets reduces manual triage time to zero and prevents "bounce" (tickets being reassigned multiple times between departments).

---

## 🚨 How Priority is Decided
Priority prediction goes beyond simple keyword matching. The system is trained on historical support data to understand the urgency and frustration levels implicit in the text.

Using a **Random Forest Classifier**, the model looks for specific linguistic patterns and feature combinations that correlate with varying levels of business risk. Priorities include:
- **Low / Medium:** General questions, feature requests, or non-blocking bugs.
- **High:** Account lockouts, billing errors causing distress.
- **Critical:** Platform-wide outages, significant data loss, or high-tier enterprise customer escalation factors.

*Business Value:* Urgent problems bypass the standard queue and alert the necessary response team immediately, protecting revenue and reducing churn risk.

---

## 📈 Evaluation Results & Insights
We trained and evaluated the models using a real-world Kaggle dataset (`suraj520/customer-support-ticket-dataset`) consisting of over 8,400 tickets.

- **Category Prediction Model:** Retained the best performing classifier (Multinomial NB / Random Forest) by comparing precision, recall, and F1 scores. 
- **Priority Prediction Model:** Achieved the highest baseline accuracy using a Random Forest algorithm, which effectively captured the non-linear feature relationships of urgency.

**Key Insights:**
- **Keyword Overlap:** We observed that highly emotional or urgent keywords heavily sway the priority model, while domain-specific nouns (e.g., "credit card", "login") distinctly separate the categories.
- **Confusion Matrix:** The visualization of the models demonstrates robust performance, with expected overlap between closely related intents (such as 'Cancellation' vs 'Refund'). 
*(Detailed confusion matrices are automatically saved to the `/visualizations/` folder during evaluation).*

---

## ⚙️ Getting Started (Technical Setup)
This repository is clean, modular, and designed to easily plug into continuous integration pipelines or be containerized via Docker.

### Prerequisites
Ensure you have Python 3.8+ installed. 
```bash
pip install -r requirements.txt
pip install kagglehub
```

### 1. Download the Dataset
The data uses the real-world Kaggle support ticket dataset. Download and format it by running:
```bash
python src/download_kaggle_data.py
```

### 2. Run the Full ML Pipeline 
Train the NLP vectorizers and classifiers from scratch. This outputs the compiled `.pkl` model weights to `/models/`.
```bash
python src/train_models.py
```

### 3. Evaluate the Models
Generate classification reports and Confusion Matrix `.png` files.
```bash
python src/evaluate_models.py
```

### 4. Interactive Live Prediction
Test the system manually by running the inference script:
```bash
python src/predict_ticket.py
```
*Example usage:* Enter `"My payment failed but money was deducted"` -> It will automatically output the predicted `Category` and `Priority`.

### 5. Jupyter Notebook Interactive Analysis
For data science teams, an EDA (Exploratory Data Analysis) notebook is provided to visualize text distribution and word frequencies.
```bash
jupyter notebook notebooks/ticket_classification_analysis.ipynb
```

---
*Developed for optimal support ticket routing and automated technical triage.*
