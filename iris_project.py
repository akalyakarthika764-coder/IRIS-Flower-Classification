import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def run_iris_project():
    print("==================================================")
    print("   IRIS FLOWER CLASSIFICATION PROJECT STARTING   ")
    print("==================================================\n")

    iris = load_iris()
    
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("--- Sample Data (First 5 Rows) ---")
    print(df.head(), "\n")

    X = iris.data  
    y = iris.target  
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}\n")

    print("Training model...")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!\n")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("================ MODEL REPORT ================")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("==============================================\n")

    print("--- Prediction on New Data ---")
    new_flower = [5.1, 3.5, 1.4, 0.2]
    input_data = np.array(new_flower).reshape(1, -1)
    
    predicted_id = model.predict(input_data)[0]
    predicted_name = iris.target_names[predicted_id]
    
    print(f"Input features: {new_flower}")
    print(f"Predicted species: {predicted_name.upper()}")
    print("================================================")

if __name__ == "__main__":
    run_iris_project()