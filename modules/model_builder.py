from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class ModelBuilder:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def build_pipeline(self):
        return Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

    def train(self, pipeline, X_train, y_train):
        pipeline.fit(X_train, y_train)
        print("✅ Model training complete.")
        return pipeline
