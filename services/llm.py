import os
import traceback
from typing import List, Dict, AsyncGenerator

from openai import AsyncOpenAI

from dotenv import load_dotenv
from openai import OpenAIError, OpenAI

# Load .env file
load_dotenv()

openai_model = os.getenv('OPENAI_API_MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE_URL')

print(
    f"openai_model: {openai_model} openai.api_key: {os.getenv('OPENAI_API_KEY')} openai.api_base: {os.getenv('OPENAI_API_BASE_URL')}")


async def converse(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """
    Given a conversation history, generate an iterative response of strings from the OpenAI API.

    :param messages: a conversation history with the following format:
    `[ { "role": "user", "content": "Hello, how are you?" },
       { "role": "assistant", "content": "I am doing well, how can I help you today?" } ]`

    :return: a generator of delta string responses
    """
    aclient = AsyncOpenAI(api_key=openai_api_key,
                          base_url=base_url)
    try:
        async for chunk in await aclient.chat.completions.create(model=openai_model,
                                                                 messages=messages,
                                                                 max_tokens=1600,
                                                                 stream=True):
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except OpenAIError as e:
        traceback.print_exc()
        yield f"oaiEXCEPTION {str(e)}"
    except Exception as e:
        yield f"EXCEPTION {str(e)}"


def create_conversation_starter(user_prompt: str) -> List[Dict[str, str]]:
    """
    Given a user prompt, create a conversation history with the following format:
    `[ { "role": "user", "content": user_prompt } ]`

    :param user_prompt: a user prompt string
    :return: a conversation history
    """
    return [{"role": "user", "content": user_prompt}]

def converse2(prompt, messages=None, model="gpt-4o-turbo",
              max_tokens=1500,
              temperature=0,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0):
    client = OpenAI(
        base_url=base_url,
        api_key= openai_api_key
    )

    # Add the user's message to the list of messages
    if messages is None:
        messages = []

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    ).choices[0].message.content

    # Add the assistant's message to the list of messages
    messages.append({"role": "assistant", "content": response})

    return response, messages
