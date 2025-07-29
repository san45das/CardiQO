class HeartDiseasePredictor:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, sample_df):
        return self.pipeline.predict(sample_df)
