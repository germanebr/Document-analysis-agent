import os
import requests
from flask import session

def secret_manager_key_retrieve(app_name:str,
                                secret_id:str,
                                correlation_id:str):
    """Connects to the GCP ACL Secret Manager Service to retrieve the selected secret key
    Please modify to the corresponding code according to documentation or the team's needs.
    Inputs:
        - app_name: Name of the application that the key will be used for
        - secret_id: The name of the secret to retrieve from the Secret Manager
        - correlation_id: Alphanumeric reference for keeping track on the retrieval logs"""
    try:
        user_data = session.get('user')
        if not user_data:
            user_data = {"user_email": "system", "user_name": "System Fetch for Cron"}
    except:
        user_data = {"user_email": "system", "user_name": "System Fetch for Cron"}

    querystring = {"app_name": app_name,
                   "secret_id": secret_id,
                   "correlation_id": correlation_id,
                   "user_email": user_data["user_email"],
                   "user_name": user_data["user_name"]}

    # Set verify = False for running on local. Remove or set to True when pushing code
    try:
        try:
            url = "microservice url for connecting to GCP Secret Manager when deployed"
            response = requests.post(url, params = querystring, verify = False)
        except Exception as ex:
            # url = os.getenv("secret_manager_url_local")
            url = "microservice url for connecting to GCP Secret Manager in local"

            response = requests.post(url, params=querystring, verify=False)
        return response.json()['res']
    except Exception as e:
        print(f"Error in Secret Manager APi _________{e}")
        raise e