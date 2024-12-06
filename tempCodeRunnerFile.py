import os

# SECURITY ISSUE: Never commit API tokens/secrets directly in code
# Instead, load from environment variables or config files
huggingface_hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

if huggingface_hub_token is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN environment variable is not set.")