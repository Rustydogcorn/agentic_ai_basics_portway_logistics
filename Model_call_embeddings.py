from mlflow import deployments as dp

def embedding_call(user_input):
    chat = dp.get_deploy_client("databricks")

    response = chat.predict(
        endpoint = "gemini-embeddings",
        inputs = {
            "input":
            user_input}
    )

    return response["data"][0]["embedding"]