# import dependencies
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# import zenml
from zenml.steps import BaseStepConfig, Output, step
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# create zenml steps
@step
def load_data() -> Output(
    df=pd.DataFrame
):
    # load dataset
    df = pd.read_csv('../Dataset/Datasets.csv')

    return df


# data preparation
@step
def data_preparation(df: pd.DataFrame) -> Output(
    data=pd.DataFrame
):
    data = df.copy()
    # Drop specified labels from rows or columns.
    data.drop(['Loan_ID'], axis=1, inplace=True)

    return data


@step
def split_data(df: pd.DataFrame) -> Output(
    x_train=np.ndarray, x_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    # assign independent variable
    x = df.drop(['Loan_Status'], axis=1).values
    # assign dependent variable
    y = df['Loan_Status'].values

    # Applies transformers to columns of an array or pandas DataFrame.
    x = ColumnTransformer([
        # Construct a Pipeline from the given estimators.
        ('Cat_tran', make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder()),
         make_column_selector(dtype_include='object')),
        ('Num_tran', make_pipeline(KNNImputer()), make_column_selector(dtype_include=['int64', 'float64']))
    ]).fit_transform(x)

    #
    y = LabelEncoder().fit_transform(y)

    # Split arrays or matrices into random train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test


@step
def scale_data(x_train: np.ndarray, x_test: np.ndarray) -> Output(
    x_train_scale=np.ndarray, x_test_scale=np.ndarray
):
    # Transform features by scaling each feature to a given range.
    scaler = MinMaxScaler()

    # Fit to data, then transform of training data
    x_train_scale = scaler.fit_transform(x_train)
    # Scale features of testing data
    x_test_scale = scaler.transform(x_test)

    return x_train_scale, x_test_scale


class ModelConfig(BaseStepConfig):
    model_name: str = 'model'

    model_params = {
        'n_estimators': 500,
        'max_depth': 10
    }


@enable_mlflow
@step(enable_cache=False)
def train_model(x_train: np.ndarray, y_train: np.ndarray, config: ModelConfig) -> Output(
    model=ClassifierMixin
):
    params = config.model_params
    # A random forest classifier.
    model = RandomForestClassifier(**config.model_params)
    # Build a forest of trees from the training set
    model.fit(x_train, y_train)

    # mlflow logging
    mlflow.sklearn.log_model(model, config.model_name)

    for param in params.keys():
        mlflow.log_param(f'{param}', params[param])

    return model


@enable_mlflow
@step(enable_cache=False)
def evaluate_model(model: ClassifierMixin, x_test: np.ndarray, y_test: np.ndarray) -> Output(
    recall=float  # accuracy = float, f1 = float, recall = float, precision = float
):
    # Predict class for X.
    y_pred = model.predict(x_test)

    # model metrics
    # Predict class for X.
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # logging metrics score to mlflow
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1', f1)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('precision', precision)

    return recall


@pipeline(enable_cache=False)
def create_pipeline(
        get_data,
        feature_engineering,
        get_train_data,
        preprocess_data,
        rf_model,
        evaluate_rf
):
    df = get_data()
    df = feature_engineering(df=df)
    x_train, x_test, y_train, y_test = get_train_data(df=df)
    x_train, x_test = preprocess_data(x_train=x_train, x_test=x_test)
    model = rf_model(x_train=x_train, y_train=y_train)
    evaluate_rf = evaluate_rf(model=model, x_test=x_test, y_test=y_test)


def main():
    training = create_pipeline(
        get_data=load_data(),
        feature_engineering=data_preparation(),
        get_train_data=split_data(),
        preprocess_data=scale_data(),
        rf_model=train_model(),
        evaluate_rf=evaluate_model()
    )

    training.run()


if __name__ == '__main__':
    main()
