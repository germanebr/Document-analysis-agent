import asyncio
import random
import vertexai.preview.generative_models as generative_models

from config import Config
from google.cloud import aiplatform
from google.oauth2 import service_account
from typing import Optional
from vertexai.generative_models import GenerativeModel, GenerationConfig

class Gemini():
    def __init__(self,
                 model_name:str = "gemini-1.5-flash-002",
                 temperature:float = 0.0,
                 max_tkns:int = 8192,
                 top_p:float = 0.95,
                 retries:int = 3,
                 backoff_factor:int = 2):
        
        credentials = service_account.Credentials.from_service_account_file(Config.GCP_CREDENTIALS)
        aiplatform.init(project = Config.GCP_DEV_PROJECT,
                        credentials = credentials)
        
        self.temperature = temperature
        self.max_tkns = max_tkns
        self.top_p = top_p
        
        self.safety_settings = {generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE}
        
        model = GenerativeModel(model_name)
        self.chat = model.start_chat()

        self.chat_history = []

        self.retries = retries
        self.backoff_factor = backoff_factor

    async def run_gemini_async(self,
                               prompt:str,
                               schema:Optional[dict] = None) -> str:
        """Gives the input prompt to the LLM and generates an answer.
        Inputs:
            - prompt: the set of instructions to give to the LLM
            - schema: the response schema that the LLM must follow for the response
        Returns:
            - answer: the generated answer from the LLM"""
        
        try:
            answer = await self.chat.send_message_async(prompt,
                                                        generation_config = GenerationConfig(temperature = self.temperature,
                                                                                             max_output_tokens = self.max_tkns,
                                                                                             top_p = self.top_p,
                                                                                             response_mime_type = "application/json" if schema else None,
                                                                                             response_schema = schema),
                                                        safety_settings = self.safety_settings)
            
            self.chat_history.append({"user": prompt,
                                      "assistant": answer.text})

            return answer.text
        
        except Exception as e:
            raise

    async def retry_llm_call_async(self,
                                   prompt:str,
                                   schema:Optional[dict] = None) -> str:
        """Runs LLM calls with an exponential backoff retry in case the API call is not completed.
        Inputs:
            - prompt: the prompt to execute
            - schema: the response schema that the LLM must follow for the response
        Returns:
            - answer: the generated LLM answer"""
        
        delay = 0
        for attempt in range(self.retries):
            try:
                return await self.run_gemini_async(prompt, schema)
            except Exception as e:
                if attempt == (self.retries - 1):
                    return e
                
                delay = min(delay * self.backoff_factor + random.uniform(0, 1), 60) # Maximum 60 seconds of wait
                await asyncio.sleep(delay)

    async def run_prompts_async(self,
                                prompts:list[str],
                                keys:Optional[list[str]] = None,
                                schemas:Optional[list[dict]] = None) -> dict:
        """Runs the given list of prompts and stores them in their corresponding key value.
        Inputs:
            - prompts: the list of prompts to execute
            - keys: the list of the key value for each prompt that wants to be executed
            - schemas: the list of response schema for each prompt
        Returns:
            - answer: the dictionary with all the LLM responses ordered by their key value pair"""
        
        if keys is None:
            keys = [f"response_{i}" for i in range(len(prompts))]
        elif len(keys) != len(prompts):
            raise ValueError("The number of keys must match the number of prompts.")
        
        if schemas and (len(schemas) != len(prompts)):
            raise ValueError("The number of response schemas must match the number of prompts.")

        responses = dict(zip(keys, [await self.retry_llm_call_async(prompt, schema) for prompt,schema in zip(prompts, schemas)]))

        return {"status": all(isinstance(value, str) for value in responses.values()),
                "content": responses}
    
    def get_chat_history(self) -> list[dict]:
        return self.chat_history