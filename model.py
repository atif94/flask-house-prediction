import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import boto3
import botocore
from credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

bucket_name = 'your-bucket-name'
object_key = 'path/to/House_Price.json'

try:
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    data = response['Body'].read().decode('utf-8')
    df = pd.read_json(data)
except botocore.exceptions.ClientError as e:
    print(f"Error retrieving data from S3: {e}")
    # Handle the error appropriately

x = df['Area(in sq. ft)'].values.reshape(-1, 1)
y = df['Price(in Rs.)'].values.reshape(-1, 1)
lin = LinearRegression()
lin.fit(x, y)
pickle.dump(lin, open('model.pkl', 'wb'))