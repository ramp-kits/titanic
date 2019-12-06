import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
from rampwf.utils.importing import import_file

class Classifier(object):
    def __init__(self, workflow_element_names=['classifier']):
        self.element_names = workflow_element_names
        # self.name = 'classifier_workflow'  # temporary

    def train_submission(self, module_path, X_array, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        classifier = import_file(module_path, self.element_names[0])
        clf = classifier.clf
        clf.fit(X_array[train_is], y_array[train_is])
        return clf

    def test_submission(self, trained_model, X_array):
        clf = trained_model
        y_proba = clf.predict_proba(X_array)
        return y_proba

problem_title = 'Titanic survival classification'
_target_column_name = 'Survived'
_ignore_column_names = ['PassengerId']
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = Classifier()

score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    X_df_new = pd.concat(
        [X_df.get(['Fare', 'Age', 'SibSp', 'Parch']),
         pd.get_dummies(X_df.Sex, prefix='Sex', drop_first=True),
         pd.get_dummies(X_df.Pclass, prefix='Pclass', drop_first=True),
         pd.get_dummies(
             X_df.Embarked, prefix='Embarked', drop_first=True)],
        axis=1)
    X_df_new = X_df_new.fillna(-1)
    XX = X_df_new.values
    return XX, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


