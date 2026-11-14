
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load cleaned dataset
base_path = Path(__file__).parent
df = pd.read_csv(base_path / "data" / "cleaned_house_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 2: Separate inputs and output
X = df[["location", "total_sqft", "bath", "bhk"]]
y = df["price"]

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Step 4: Encode location
preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"), ["location"])
    ],
    remainder="passthrough"
)

# Step 5: Create both models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

# Step 6: Train and evaluate both models
model_folder = base_path / "models"
model_folder.mkdir(exist_ok=True)

for name, regressor in models.items():

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n", name)
    print("-------------------------")
    print("MAE:", round(mae, 2), "lakhs")
    print("RMSE:", round(rmse, 2), "lakhs")
    print("R2 Score:", round(r2, 4))

    # Save each model separately
    if name == "Linear Regression":
        filename = "linear_regression_model.pkl"
    else:
        filename = "random_forest_model.pkl"

    joblib.dump(model, model_folder / filename)

    print("Saved:", filename)

print("\nBOTH MODELS TRAINED AND SAVED SUCCESSFULLY!")
