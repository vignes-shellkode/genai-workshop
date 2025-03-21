import json
import boto3
import os
from dotenv import load_dotenv
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

model_input = {
    "taskType": "TEXT_VIDEO",
    "textToVideoParams": {
        "text": "Long shot of a large tiger walking through the grass.",
    },
    "videoGenerationConfig": {
        "durationSeconds": 6,
        "fps": 24,
        "dimension": "1280x720",
        "seed": 1,  # Change the seed to get a different result
    },
}

try:
    # Start the asynchronous video generation job.
    invocation = bedrock_client.start_async_invoke(
        modelId="amazon.nova-reel-v1:0",
        modelInput=model_input,
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{os.getenv("S3_BUCKET_NAME")}"
            }
        }
    )

    # Print the response JSON.
    print(json.dumps(invocation, indent=2, default=str))
except Exception as e:
    # Implement error handling here.
    message = e.response["Error"]["Message"]
    print(f"Error: {message}")
