from mlflow import deployments as dp

def agent_call(system_content, user_content):
    chat = dp.get_deploy_client("databricks")

    response = chat.predict(
        endpoint = "gemini-free-tier",
        inputs = {
            "messages":
            [{"role":"system", "content": system_content}, 
            {"role":"user", "content": user_content}]
        }
    )

    return response["choices"][0]["message"]["content"]