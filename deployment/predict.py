# predict.py

import numpy as np
import pandas as pd
import joblib

# Load once globally (safe for Streamlit)
model = joblib.load('model/Gradient_Boosting.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')

# Required model features
final_features = [
    'Administrative', 'Administrative_Duration', 'ProductRelated',
    'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
    'TrafficType', 'Weekend', 'VisitorType_Other', 'VisitorType_Returning_Visitor',
    'Month_sin', 'Month_cos', 'SpecialDay_1', 'OperatingSystems_2', 'OperatingSystems_3',
    'OperatingSystems_other', 'Browser_2', 'Browser_other', 'Region_2', 'Region_3',
    'Region_4', 'Region_5', 'Region_6', 'Region_7', 'Region_8', 'Region_9'
]

def preprocess_input(user_input_df: pd.DataFrame):
    """Preprocess user input into model-ready format."""
    df = user_input_df.copy()

    # 👉 Apply log1p transformation for skewed numerical columns
    log_cols = [
        'Administrative', 'Administrative_Duration',
        'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues'
    ]
    for col in log_cols:
        df[col] = np.log1p(df[col])

    # 👉 Visitor Type one-hot
    visitor = df['VisitorType'].iloc[0]
    df['VisitorType_Returning_Visitor'] = int(visitor == 'Returning_Visitor')
    df['VisitorType_Other'] = int(visitor == 'Other')

    # 👉 Cyclical encoding for Month
    month = int(df['Month'].iloc[0])
    df['Month_sin'] = np.sin(2 * np.pi * month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * month / 12)

    # 👉 SpecialDay binary
    df['SpecialDay_1'] = int(df['SpecialDay'].iloc[0] == 1)

    # 👉 Weekend to binary
    df['Weekend'] = int(df['Weekend'])

    # 👉 OS encoding
    os_val = df['OperatingSystems'].iloc[0]
    df['OperatingSystems_2'] = int(os_val == 2)
    df['OperatingSystems_3'] = int(os_val == 3)
    df['OperatingSystems_other'] = int(os_val not in [1, 2, 3])

    # 👉 Browser encoding
    browser_val = df['Browser'].iloc[0]
    df['Browser_2'] = int(browser_val == 2)
    df['Browser_other'] = int(browser_val not in [1, 2])

    # 👉 Region one-hot for 2–9
    for i in range(2, 10):
        df[f'Region_{i}'] = int(df['Region'].iloc[0] == i)

    # 👉 Collect all required features (fill 0 if any missing)
    result = pd.DataFrame(index=df.index)
    for col in final_features:
        result[col] = df.get(col, 0)

    # 👉 Scale
    scaled = preprocessor.transform(result)
    return pd.DataFrame(scaled, columns=final_features)

def predict_purchase(user_input_df: pd.DataFrame):
    processed = preprocess_input(user_input_df)
    return model.predict(processed)[0]
