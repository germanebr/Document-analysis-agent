import requests
import json
from typing import List

def run_BQ_query(query:str) -> List:
    """Runs the given SQL query through the BigQuery microservice.
    Inputs:
        - query: the BigQuery query to execute
    Returns:
        - res: the response obtained"""

    # Currently using a microservice for connecting to BigQuery
    # Please modify this function as necessary accordingly to GCP documentation.
    url = "{bigquery_microservice_endpoint}"
    response = requests.post(url,
                             json = {"query": query,
                                     "records_per_page": 1000})
    # print(response.json())
    return json.loads(response.json()["res"])["data"]