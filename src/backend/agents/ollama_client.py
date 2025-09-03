"""Ollama API client for model communication"""

import logging
from typing import List
import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Handles Ollama API communication with model loading"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.current_model = None
    
    async def load_model(self, model_name: str):
        """Load a specific model into memory"""
        if self.current_model == model_name:
            return True
            
        try:
            # Unload current model if any
            if self.current_model:
                logger.info(f"Unloading model: {self.current_model}")
                
            logger.info(f"Loading model: {model_name}")
            # Make a small request to ensure model is loaded
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_model = model_name
                logger.info(f"Successfully loaded model: {model_name}")
                return True
            else:
                logger.error(f"Failed to load model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    async def generate_response(self, model_name: str, prompt: str) -> str:
        """Generate response from specified model"""
        # Ensure model is loaded
        if not await self.load_model(model_name):
            raise Exception(f"Failed to load model: {model_name}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 1024
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available small models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()["models"]
                # Filter for small models only
                small_models = []
                for model in models:
                    name = model["name"]
                    # Include models with 1b, 1.5b, 2b, or 3b in name
                    if any(size in name.lower() for size in ["1b", "1.5b", "2b", "3b"]):
                        # Exclude larger models
                        if not any(large in name.lower() for large in ["7b", "8b", "70b", "13b"]):
                            small_models.append(name)
                return small_models
            return []
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []