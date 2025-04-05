# 🛍️ Online Shopper Purchase Intention (OSPI) App

A machine learning-powered Streamlit app to predict whether an online shopper is likely to make a purchase, using session behavior data and multiple trained models.

---

## 🔗 Repository

**GitHub Repo**: [`git@github.com:pranav9989/OSPI.git`](git@github.com:pranav9989/OSPI.git)

---

## 🛠️ Setup Instructions

Follow these steps to set up and run the app on your local machine:

---

### ✅ Step 1: Clone the Repository

```bash
git clone git@github.com:pranav9989/OSPI.git
cd OSPI

### ✅ Step 2: Create a local environment
# For Windows
python -m venv shopper-env
shopper-env\Scripts\activate

# For macOS/Linux
python3 -m venv shopper-env
source shopper-env/bin/activate

### ✅ Step 3: Install Required Packages

pip install -r requirements.txt


### ✅ Step 4: Run Streamlit app
cd deployment
streamlit run app.py
