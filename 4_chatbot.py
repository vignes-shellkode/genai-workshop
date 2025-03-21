import boto3
import json
import os
from dotenv import load_dotenv
load_dotenv()

class ChatMemory:
    """
    Simple chat memory class to maintain conversation history.
    """
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, content):
        """Add a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content):
        """Add an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_messages(self):
        """Get the full conversation history."""
        return self.messages
    
    def clear(self):
        """Clear the conversation history."""
        self.messages = []

def generate_text_with_memory(chat_memory, user_input, max_tokens=100):

    # Add the new user input to memory
    chat_memory.add_user_message(user_input)

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
    
    # Model ID for Anthropic Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Prepare the request body for Claude with the full conversation history
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": chat_memory.get_messages()
    }
    
    # Invoke the model
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body)
    )
    
    # Parse the response
    response_body = json.loads(response['body'].read().decode('utf-8'))
    
    # Extract the generated text from Claude's response
    generated_text = response_body['content'][0]['text']
    
    # Add the assistant's response to memory
    chat_memory.add_assistant_message(generated_text)
    
    return generated_text

def main():
    # Initialize chat memory
    memory = ChatMemory()
    
    print("Chat with Claude (type 'exit' to quit):")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Ending chat session.")
            break
        
        # Get response from Claude with memory
        response = generate_text_with_memory(memory, user_input)
        
        print(f"\nClaude: {response}")

if __name__ == "__main__":
    main()