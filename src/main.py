import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    dataset = pd.read_csv('assets/Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    score = classifier.score(X_test, y_test)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)

    print("Confusion Matrix:")
    print(cm)
    mlflow.log_metric("true_negatives", cm[0][0])
    mlflow.log_metric("false_positives", cm[0][1])
    mlflow.log_metric("false_negatives", cm[1][0])
    mlflow.log_metric("true_positives", cm[1][1])

    mlflow.log_param("classifier", "LogisticRegression")
    mlflow.log_param("random_state", 0)

    mlflow.sklearn.log_model(classifier, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
