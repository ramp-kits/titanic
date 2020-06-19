from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier


def get_estimator():

    categorical_cols = ['Sex', 'Pclass', 'Embarked']
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore'),
    )

    preprocessor = make_column_transformer(
        (categorical_pipeline, categorical_cols),
        (SimpleImputer(strategy='constant', fill_value=-1), numerical_cols),
    )

    pipeline = Pipeline([
        ('transformer', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=20, max_depth=5)),
    ])

    return pipeline
