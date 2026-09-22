import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.client import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "Ads_Classifier"

if __name__ == "__main__":
    mlflow.set_experiment("MLOPS_Study")
    client = MlflowClient()

    dataset = pd.read_csv('assets/Social_Network_Ads.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.iloc[:, [2, 3]].values,
        dataset.iloc[:, 4].values,
        test_size=0.25,
        random_state=7
    )

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    print("\n--- 1. EXPERIMENT TRACKING AND CREATING CHAMPION (V1) ---")
    with mlflow.start_run(run_name="Model_Base_V1") as run_v1:
        clf_v1 = LogisticRegression(random_state=7).fit(X_train, y_train)
        acc_v1 = clf_v1.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc_v1)

        # The tracking save all (log) and the registry save only the important stuff (model artifacts)
        mlflow.sklearn.log_model(clf_v1, name="model")

        mv1 = mlflow.register_model(
            f"runs:/{run_v1.info.run_id}/model",
            MODEL_NAME
        )

        # ALIASES: Set the version 1 as a production model ("Champion")
        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            mv1.version
        )
        print(
            f"V1 (Logistic Regression) set and tagged as @champion. Acc: {acc_v1:.2f}"
        )

    print("\n--- 2. NEW EXPERIMENT TRACKING AND CREATING CHALLENGER (V2) ---")
    with mlflow.start_run(run_name="New_Model_V2") as run_v2:
        clf_v2 = RandomForestClassifier(random_state=7).fit(X_train, y_train)
        acc_v2 = clf_v2.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc_v2)
        mlflow.sklearn.log_model(
            clf_v2,
            "model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        mv2 = mlflow.register_model(
            f"runs:/{run_v2.info.run_id}/model",
            MODEL_NAME
        )

        # ALIASES: Set the version 2 as a challenger model ("Challenger")
        client.set_registered_model_alias(
            MODEL_NAME,
            "challenger",
            mv2.version
        )
        print(
            f"V2 (Random Forest) set and tagged as @challenger. Acc: {acc_v2:.2f}"
        )

    print("\n--- 3. PROMOTION WORKFLOW ---")
    if acc_v2 > acc_v1:
        print("Decision making: The challenger (V2) is better than the champion (V1). Promoting V2 to Champion...")
        client.set_registered_model_alias(MODEL_NAME, "champion", mv2.version)
        print("V2 now promoted to Champion! The production system will automatically use the new version.")

    print("\n--- 4. ROLLBACK STRATEGY ---")
    print("Ops! The new champion (V2) is not performing well in production. We need to rollback to the previous champion (V1).")
    client.set_registered_model_alias(MODEL_NAME, "champion", mv1.version)
    print("Rollback done! The production system will automatically use the previous champion (V1).")

    modelo_producao = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )
    print(
        "\nModel loaded from production (champion) alias:",
        type(modelo_producao).__name__
    )
