from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def get_estimator():

    categorical_cols = ['Sex', 'Pclass', 'Embarked']
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore'),
    )
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    numerical_pipeline = make_pipeline(
        StandardScaler(), SimpleImputer(strategy='constant', fill_value=-1)
    )

    preprocessor = make_column_transformer(
        (categorical_pipeline, categorical_cols),
        (numerical_pipeline, numerical_cols),
    )

    pipeline = Pipeline([
        ('transformer', preprocessor),
        ('classifier', LogisticRegression()),
    ])

    return pipeline
