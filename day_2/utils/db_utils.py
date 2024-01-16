from google.cloud import storage
from sqlalchemy import create_engine, text
import pandas as pd
import json
import boto3

PROJECT_ID = 'tanpopo-ml'
BUCKET_NAME = 'ai-education-2023'


class DbUtils():
    def __init__(self, cloud_kind: str = 'gcp'):
        if cloud_kind == 'gcp':
            client = storage.Client(PROJECT_ID)
            bucket = client.get_bucket(BUCKET_NAME)

            blob = bucket.blob('db_settings/db_settings.json')
            content = blob.download_as_string()

        elif cloud_kind == 'aws':
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket='ai-education-2024', Key='db-settings/db_settings.json')
            content = response['Body'].read()

        db_settings = json.loads(content)

        self.db_engine = create_engine('mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={encoding}'.format(**db_settings))

    def regist_score(self, table_name: str, player_name: str, score: float, obstacles: int) -> None:
        regist_df = pd.DataFrame(
            [
                [player_name, score, obstacles]
            ],
            columns=['player_name', 'score', 'obstacles']
        )

        regist_df.to_sql(
            table_name, con=self.db_engine, if_exists='append', index=False
        )

    def get_top_n_player(self, table_name: str, n: int = 10) -> list:
        query = """
            select * from `{table_name}`
            order by score desc, created_datetime  asc
            limit {n}
        """.format(
            table_name=table_name,
            n=n
        )

        with self.db_engine.connect() as connection:
            res = connection.execute(text(query))
            data = [r for r in res]

        return data
