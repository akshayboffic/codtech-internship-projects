import numpy as np 
import pandas as pd 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

# Configuring Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="etl_pipeline.log",
    filemode='w'
)

# Defining numerical and categorical features
numeric_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']
categorical_features =[]

# Extract Data
def extract_data(filepath):
    logging.info("Starting Data Extraction from %s",filepath)
    try: 
        data =  pd.read_csv(filepath)
        logging.info("Data Extraction Successful. Shape: %s",data.shape)
        return data
    except Exception as e:
        logging.error("Error during Data Extraction: %s", str(e))
        raise

# Transforming Data
def transform_data(df):

    logging.info("Starting data transformation.")
    try:
        numeric_processor = Pipeline(steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
                                        ("Scaler",StandardScaler())])

        categorical_processor = Pipeline(steps=[("imputation_constant",SimpleImputer(fill_value="missing",strategy="constant")),
                                        ("OneHotEncoding",OneHotEncoder())])
        
        
        preprocessor = ColumnTransformer(
            [("categorical",categorical_processor,categorical_features),
            ("numerical",numeric_processor,numeric_features )]
        )

        X = df.drop('target',axis=1)
        y = df['target']

        X = preprocessor.fit_transform(X)

        return X,y  
    
    except Exception as e:
        logging.error("Error during data transformation: %s", str(e))
        raise

# Training ML Model
def train_model(X, y):
    logging.info("Starting model training.")
    try:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and train the XGBoost classifier
        model = XGBClassifier(random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logging.info("Model training complete. Accuracy: %.2f", accuracy)
        logging.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
        return model
    except Exception as e:
        logging.error("Error during model training: %s", str(e))
        raise

# Pipeline Execution
if __name__ == "__main__":

    logging.info("ETL Pipeline Execution Started.")
    
    try:
        # Filepath to the dataset
        input_file = "heart.csv"  # Replace with your actual file path

        # Execute ETL steps
        raw_data = extract_data(input_file)
        X, y = transform_data(raw_data)

        # Train the machine learning model
        trained_model = train_model(X, y)
        logging.info("ETL Pipeline Execution Completed Successfully.")
    except Exception as e:
        logging.error("Pipeline execution failed: %s", str(e))
