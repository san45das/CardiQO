from modules.data_loader import DataLoader
from modules.data_preprocessor import DataPreprocessor
from modules.model_builder import ModelBuilder
from modules.model_evaluator import ModelEvaluator
from modules.predictor import HeartDiseasePredictor

from sklearn.model_selection import cross_val_score
import joblib


# File path 
file_path = "data/heart_disease.csv"  

# Column Definitions
numerical_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
categorical_features = ['sex', 'chest pain type', 'fasting blood sugar',
                        'resting ecg', 'exercise angina', 'ST slope']
target_column = 'target'

# Data Loading
loader = DataLoader(file_path)
df = loader.load_data()

# Preprocessing Setup 
preprocessor = DataPreprocessor(numerical_features, categorical_features)
X_train, X_test, y_train, y_test = preprocessor.split_data(df, target=target_column)
pipeline = preprocessor.build_pipeline()

# Model Building and Training
builder = ModelBuilder(pipeline)
model_pipeline = builder.build_pipeline()
trained_model = builder.train(model_pipeline, X_train, y_train)

# Trained model saving
joblib.dump(trained_model, 'heart_disease_model.pkl')
print("💾 Trained model saved to 'heart_disease_model.pkl'")

# Cross-Validation 
scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
print(f"\n🔁 Cross-validation accuracy: {scores.mean():.2f} (std: {scores.std():.2f})")

# Test Set Evaluation
y_pred = trained_model.predict(X_test)
y_proba = trained_model.predict_proba(X_test)[:, 1]
ModelEvaluator.evaluate(y_test, y_pred, y_proba)

# Single Sample Prediction
predictor = HeartDiseasePredictor(trained_model)
sample = df.drop(columns=['target']).iloc[[0]]
prediction = predictor.predict(sample)[0]
print(f"\n🩺 Predicted Class for Sample 0: {prediction}")


