import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import boto3
from credentials import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# S3 bucket and object key
bucket_name = 'aws-bbucket-name'
object_key = 'House_Price.json'

try:
    # Fetch data from S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    data = response['Body'].read().decode('utf-8')
    df = pd.read_json(data)
except Exception as e:
    print(f"Error retrieving data from S3: {e}")
    # Handle the error appropriately

# Prepare input and output data
X = df['Area(in sq. ft)'].values.reshape(-1, 1)
y = df['Price(in Rs.)'].values.reshape(-1, 1)

# Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
pickle.dump(lin_reg, open('model.pkl', 'wb'))
print("Model saved to 'model.pkl'")

# Sav