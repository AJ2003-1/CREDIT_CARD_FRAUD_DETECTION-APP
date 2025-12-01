import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import pickle as pkl
from collections import Counter

def losd_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train_sm, y_train_sm)
    return rf, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, conf_matrix, class_report, roc_auc

if __name__ == "__main__":
    df = losd_data(r"C:\Users\aaron\OneDrive\Desktop\DATASETS\card_transdata.csv")
    features = [
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
        "repeat_retailer",
        "used_chip",
        "used_pin_number",
        "online_order"
    ]
    target = "fraud"
    X_scaled, y, scaler = preprocess_data(df, features, target)
    model, X_test, y_test = train_model(X_scaled, y)
    
    print(df[df['fraud']==1])
    
    with open("AJ_data.pkl", "wb") as f:
        pkl.dump(model, f)


    
  