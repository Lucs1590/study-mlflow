import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    mlflow.set_experiment("Previsao_Anuncios_Rede_Social")

    mlflow.sklearn.autolog()

    dataset = pd.read_csv('assets/Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7
    )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    with mlflow.start_run(run_name="Regressao_Logistica_Base") as run:

        mlflow.set_tag("ambiente", "desenvolvimento")
        mlflow.set_tag("modelo", "RegressaoLogistica")

        mlflow.log_param("dataset_path", 'assets/Social_Network_Ads.csv')

        classifier = LogisticRegression(random_state=7)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        score = classifier.score(X_test, y_test)
        print("Score: %s" % score)

        print("Confusion Matrix:")
        print(cm)

        mlflow.log_metrics({
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1]
        })

        print(f"Model saved in run {run.info.run_id}")
