import pandas as pd
import numpy as np
import math
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# -------------------------------------------------------------
#  BREAST SIZE  ➜  TUMOR AREA (FICTIONAL APPROX)
# -------------------------------------------------------------

def compute_radius_from_band(band):
    """
    Converts bra band size to approximate breast radius (cm).
    Fictional but proportional.
    """
    band_to_radius = {
        30: 7.5, 32: 8.0, 34: 8.5,
        36: 9.0, 38: 9.5, 40: 10.0
    }
    return band_to_radius.get(band, 8.5)


def compute_cup_multiplier(cup):
    """
    Cup size multiplier affecting tumor space.
    """
    cup = cup.upper()
    mapping = {
        "A": 0.8, "B": 1.0, "C": 1.3,
        "D": 1.6, "DD": 1.9, "E": 2.2
    }
    return mapping.get(cup, 1.0)


def compute_tumor_area_from_breast_size(size):
    """
    Converts breast size (like 34C, 36DD) into fictional tumor area.
    Formula: π * r^2 * multiplier
    """
    size = size.upper().strip()

    band = int("".join([c for c in size if c.isdigit()]))
    cup = "".join([c for c in size if c.isalpha()])

    radius = compute_radius_from_band(band)
    multiplier = compute_cup_multiplier(cup)

    tumor_area = math.pi * (radius ** 2) * multiplier
    return round(tumor_area, 2)


# -------------------------------------------------------------
#  TUMOR AREA → STAGE (FICTIONAL LOGIC)
# -------------------------------------------------------------
def determine_stage(tumor_area):
    if tumor_area < 250:
        return "Stage 0"
    elif tumor_area < 400:
        return "Stage I"
    elif tumor_area < 650:
        return "Stage II"
    elif tumor_area < 900:
        return "Stage III"
    else:
        return "Stage IV"


# -------------------------------------------------------------
#  PRECAUTIONS BY STAGE
# -------------------------------------------------------------
stage_precautions = {
    "Stage 0": [
        "Maintain monthly self-examination.",
        "Follow a balanced antioxidant-rich diet.",
        "Avoid excessive caffeine and smoking."
    ],
    "Stage I": [
        "Schedule regular clinical screening every 6 months.",
        "Maintain fitness (30 mins daily walk).",
        "Reduce alcohol intake."
    ],
    "Stage II": [
        "Consult a specialist for early evaluation.",
        "Avoid stress and maintain proper sleep.",
        "Shift to low-sodium, low-fat diet."
    ],
    "Stage III": [
        "Immediate medical attention recommended.",
        "Avoid any heavy physical strain.",
        "Follow a structured diet plan under supervision."
    ],
    "Stage IV": [
        "Urgent oncology consultation required.",
        "Do NOT self-medicate at any cost.",
        "Stay hydrated and maintain immune-supportive nutrition."
    ]
}


# -------------------------------------------------------------
#  ML MODEL + BREAST SIZE COMBINED PREDICTION
# -------------------------------------------------------------
def manual_predict(raw_input_features, model, scaler, feature_names, target_names, breast_size):

    df = pd.DataFrame([raw_input_features], columns=feature_names)
    scaled = scaler.transform(df)

    pred_idx = int(model.predict(scaled)[0])
    pred_label = target_names[pred_idx]

    try:
        decision_value = float(model.decision_function(scaled)[0])
    except:
        decision_value = model.predict_proba(scaled)[0].tolist()

    # BREAST SIZE → TUMOR AREA → STAGE
    tumor_area = compute_tumor_area_from_breast_size(breast_size)
    stage = determine_stage(tumor_area)
    precaution_list = stage_precautions[stage]

    return {
        "prediction": pred_label,
        "decision_value": decision_value,
        "breast_size": breast_size,
        "tumor_area": tumor_area,
        "stage": stage,
        "precautions": precaution_list
    }


# -------------------------------------------------------------
#  MAIN PROGRAM
# -------------------------------------------------------------
def main():

    # Load dataset
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train SVM model
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    print("\n================ MODEL PERFORMANCE ================")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Manual prediction example
    print("\n================ MANUAL PREDICTION ================")

    breast_size = input("Enter breast size (e.g., 32B, 34C, 36D, 38DD): ")

    sample_input = cancer.data[0].tolist()

    result = manual_predict(
        raw_input_features=sample_input,
        model=model,
        scaler=scaler,
        feature_names=list(cancer.feature_names),
        target_names=list(cancer.target_names),
        breast_size=breast_size
    )

    print("\n================ FINAL OUTPUT ================")
    print("Breast Size:", result["breast_size"])
    print("Tumor Area (fictional):", result["tumor_area"], "cm²")
    print("Stage:", result["stage"])
    print("Model Prediction (Benign/Malignant):", result["prediction"])
    print("Decision Score:", result["decision_value"])

    print("\nPrecautions for", result["stage"], ":")
    for p in result["precautions"]:
        print(" -", p)


if __name__ == "__main__":
    main()