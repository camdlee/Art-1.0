import re
from typing import Optional, List, Dict, Any, Union

import autogen
import requests
from autogen import Agent, ConversableAgent

# This is a general agent with a specific name and a specific reply function.
# This is an example of how to implement a custom agent.
class OpenAPIAgent(autogen.AssistantAgent):
    """
    An Autogen-based assistant agent that extracts OpenAPI specification URLs from incoming messages and retrieves their content.

    This custom agent listens for incoming messages containing URLs, identified OpenAPI specifications links, and fetches their raw specification text.
    """
    def __init__(self):
        """
        Initialize the OpenAPI Agent.

        Sets the agent name to "OpenAPI Agent", disables LLM configuration by default,
        limits the number of consecutive auto replies to 3, and registers a custom
        reply function.
        """
        super().__init__(name="OpenAPI Agent", llm_config=None,
                         max_consecutive_auto_reply=3)
        self.register_reply([Agent,None], self._reply_func)

    def _reply_func(self,
                    recipient: ConversableAgent,
                    messages: Optional[List[Dict]] = None,
                    sender: Optional[Agent] = None,
                    config: Optional[Any] = None,
                    ) -> Any | Union[str, Dict, None]:
        """
        Custom reply function that looks for URLs in the first message,
        and if found, returns the retrieved OpenAPI specification.

        Args:
            recipient (ConversableAgent): The recipient of the reply.
            messages (list[dict], optional): A list of conversation messages.
            sender (Agent, optional): The agent sending the message.
            config (Any, optional): Additional configuration.

        Returns:
            tuple[bool, Optional[str]]:
                - First element is True if a valid OpenAPI spec was retrieved, False otherwise.
                - Second element is the raw specification text or None.
        """
        urls = self._extract_urls(messages[0]["content"])
        if (urls is None) or (len(urls) == 0):
            return False, None
        return True, self._get_openapi_spec(urls[0])

    # Write a function that given a str extracts a list of urls present in the string
    def _extract_urls(self, text):
        """
        Extract a list of URLs from a given string.

        Args:
            text (str): The input text to search for URLs.

        Returns:
            list[str]: A list of extracted URLs.
        """
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_regex, text)
        return urls

    def _get_openapi_spec(self, url) -> Optional[str]:
        """
        Retrieve the raw OpenAPI specification from the given URL.

        Args:
            url (str): The URL to fetch.

        Returns:
            Optional[str]: The raw OpenAPI spec as text if successful, otherwise None.
        """
        response = requests.get(url)
        if response.status_code == 200:
            return str(response.text)
        else:
            return None
