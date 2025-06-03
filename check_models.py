import requests
import json
from typing import Dict, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

OLLAMA_URL = "http://localhost:11434"

def get_local_models() -> Dict[str, dict]:
    """Get list of models available locally through Ollama."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            return {model['name']: model for model in response.json()['models']}
        else:
            print(f"Error: Received status code {response.status_code}")
            return {}
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Is it running?")
        return {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}

def check_model_availability(models: List[str]) -> Dict[str, bool]:
    """Check if specific models are available and can be loaded."""
    results = {}
    local_models = get_local_models()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry on
    )
    
    # Create a session with retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    for model in models:
        if model in local_models:
            try:
                # Try to load the model with a minimal prompt
                response = session.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": "test",
                        "stream": False
                    },
                    timeout=60  # Increased timeout to 30 seconds
                )
                results[model] = {
                    "available": True,
                    "loaded": response.status_code == 200,
                    "size": local_models[model].get('size', 'Unknown'),
                    "modified": local_models[model].get('modified_at', 'Unknown')
                }
            except requests.exceptions.Timeout:
                results[model] = {
                    "available": True,
                    "loaded": False,
                    "error": "Model load timed out after 30 seconds",
                    "size": local_models[model].get('size', 'Unknown'),
                    "modified": local_models[model].get('modified_at', 'Unknown')
                }
            except Exception as e:
                results[model] = {
                    "available": True,
                    "loaded": False,
                    "error": str(e),
                    "size": local_models[model].get('size', 'Unknown'),
                    "modified": local_models[model].get('modified_at', 'Unknown')
                }
        else:
            results[model] = {
                "available": False,
                "loaded": False,
                "error": "Model not found locally"
            }
    
    return results

def main():
    # List of models to check
    models_to_check = [
        "deepseek-r1:7b",
        "gemma3:4b",
        "llama3.2:3b",
        "mistral:7b"
    ]
    
    print("🔍 Checking Ollama models...")
    print("-" * 50)
    
    results = check_model_availability(models_to_check)
    
    for model, status in results.items():
        print(f"\n📦 Model: {model}")
        print(f"   Available: {'✅' if status['available'] else '❌'}")
        if status['available']:
            print(f"   Loaded: {'✅' if status['loaded'] else '❌'}")
            print(f"   Size: {status['size']}")
            print(f"   Last Modified: {status['modified']}")
            if 'error' in status:
                print(f"   Error: {status['error']}")
        else:
            print(f"   Error: {status['error']}")
    
    print("\n" + "-" * 50)
    print("💡 To install missing models, use: ollama pull <model_name>")

if __name__ == "__main__":
    main() 