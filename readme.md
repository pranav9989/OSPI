# 🛍️ Online Shopper Purchase Intention (OSPI) App



A machine learning-powered Streamlit web application designed to predict whether an online shopper is likely to make a purchase based on their session behavior data. This tool utilizes pre-trained machine learning models to provide real-time predictions through an intuitive user interface.


## 🛠️ Technology Stack

*   **Language:** Python 3.x
*   **Web Framework:** Streamlit
*   **Machine Learning:** Scikit-learn 
*   **Data Handling:** Pandas, NumPy, Seaborn, Matplotlib.

---

## ⚙️ Setup and Installation

Follow these steps to set up and run the app on your local machine:

### Prerequisites

*   [Git](https://git-scm.com/)
*   [Python](https://www.python.org/downloads/) (Version 3.8+ recommended)
*   `pip` and `venv` (usually included with Python)

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/pranav9989/OSPI.git
    # Or using SSH:
    # git clone git@github.com:pranav9989/OSPI.git
    cd OSPI
    ```

2.  **Navigate to the Deployment Directory:**
    ```bash
    cd deployment
    ```

3.  **Create and Activate a Virtual Environment:**

    *   **Windows (Git Bash or Command Prompt):**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *(You should see `(venv)` prepended to your terminal prompt)*

4.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

This will automatically open the OSPI app in your default web browser.

---

## ▶️ How to Use

1.  Ensure the app is running (using `streamlit run app.py` from the `deployment` directory within the activated virtual environment).
2.  Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
3.  Use the sidebar or main interface elements to input the required online session features (e.g., Administrative duration, Informational pages visited, ProductRelated duration, BounceRates, ExitRates, PageValues, SpecialDay proximity, VisitorType, etc. - *adjust based on your actual input fields*).
4.  Click the "Predict" or "Submit" button.
5.  The app will display the prediction indicating whether the shopper is likely to make a purchase (`Revenue` = True/False or Yes/No) based on the input data.
