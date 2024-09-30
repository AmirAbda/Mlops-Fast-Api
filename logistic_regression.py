import os
import pandas as pd
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    """Load and prepare the Iris dataset."""
    df = sns.load_dataset("iris")
    X = df.drop("species", axis=1)
    y = df["species"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_pipeline():
    """Create a pipeline with StandardScaler and LogisticRegression."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
    ])

def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """Train the model and print evaluation metrics."""
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    y_pred = pipeline.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def save_model(pipeline, filename='logistic_regression_model.joblib'):
    """Save the trained model to a file."""
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', filename)
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully in {model_path}!")

def main():
    # Load and split the data
    X_train, X_test, y_train, y_test = load_data()
    
    # Create and train the model
    pipeline = create_pipeline()
    train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    
    # Save the model
    save_model(pipeline)

if __name__ == "__main__":
    main()