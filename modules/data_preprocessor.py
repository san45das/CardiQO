from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features

    def build_pipeline(self):
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)
        ])
        return preprocessor

    def split_data(self, df, target='target'):
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

