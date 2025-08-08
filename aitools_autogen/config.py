import os
import sys
from dotenv import load_dotenv

load_dotenv()
openai_model = os.getenv('OPENAI_API_MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE_URL')

config_list_openai = [
    {
        'base_url': base_url,
        'api_key':openai_api_key,
        'model': openai_model
    }
]

llm_config_openai = {
    "timeout": 300,
    "seed": 42,
    "config_list": config_list_openai,
    "temperature": 0.1,
    "allow_format_str_template": True
}


DEFAULT_MODEL = "gpt-4o-mini"
# FAST_MODEL = "gpt-3.5-turbo"
# Regular expression for finding a code block
CODE_BLOCK_PATTERN = r"(.*?)```(\w*)\n(.*?)\n```"
WORKING_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "coding")
UNKNOWN = "unknown"
TIMEOUT_MSG = "Timeout"
DEFAULT_TIMEOUT = 600
WIN32 = sys.platform == "win32"
PATH_SEPARATOR = WIN32 and "\\" or "/"
