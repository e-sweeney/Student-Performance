import pandas as pd
import numpy as np
import mlflow
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error




mlflow.set_tracking_uri("http://127.0.0.1:5555")

# Load dataset (Example: Insurance Charges Dataset. Remove first column index)
df = pd.read_csv("Student_Performance.csv")
df["extra_Curr"] = df["extra_Curr"].map({"No": 0, "Yes": 1})





# Features (X) and Target (y)
X = df[["hrs_Studied", "prev_score",  "sleep", "sample_practiced", "extra_Curr"]]
y = df["Result"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Start MLflow Run
with mlflow.start_run():
    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Log Metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # Log Model & Register
    result = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    
    mlflow.register_model(
        model_uri=result.model_uri,
        name="my-linear-regmodel"
    )

    print(f"Model logged with R2: {r2}")

# Save model to a .pkl file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)



print("Model trained and saved as model.pkl!")
