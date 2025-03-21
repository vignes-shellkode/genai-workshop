import boto3
import json
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve credentials from .env
aws_access_key = os.getenv("ACCESS_KEY")
aws_secret_key = os.getenv("SECRET_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

# Initialize Bedrock client with credentials
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Inference configuration
inference_config = {
    "temperature": 0.0,
    "maxTokens": 100
}

system_prompt = """You are an AI assistant, and you need to answer the user's question."""

prompt = """How many states are there in the United States of America?"""

converse_api_params = {
    "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "system": [{"text": system_prompt}],
    "messages": [{"role": "user", "content": [{"text": prompt}]}],
    "inferenceConfig": inference_config
}

try:
    response = bedrock_client.converse(**converse_api_params)
    response_text = response['output']['message']['content'][0]['text']

    print(f"Response: {response_text}")

except (ClientError, json.JSONDecodeError) as err:
    print(f"Error: {str(err)}")
