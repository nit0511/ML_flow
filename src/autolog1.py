import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import dagshub

dagshub.init(repo_owner='nit0511', repo_name='ML_flow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/nit0511/ML_flow.mlflow")

# Enable autologging
mlflow.autolog()

# Load a sample dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train a model (MLflow will automatically log parameters and metrics)
with mlflow.start_run() as run:
    model = LogisticRegression(C=1.0)
    model.fit(X_train, y_train)

# Model is automatically logged when the run ends [1]