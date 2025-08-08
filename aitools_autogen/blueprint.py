from typing import Optional
from autogen import ConversableAgent


class Blueprint:
    """
    A container class that manages a set of conversational agents and coordinates
    message passing between them.

    The `Blueprint` maintains references to an initiator agent and a recipient agent,
    along with optional configuration settings for the agents and language models.
    It can be used to start an interaction between agents in an automated workflow.
    """
    def __init__(self,
                 agents: Optional[list[ConversableAgent]] = None,
                 config_list: Optional[list[dict]] = None,
                 llm_config: Optional[dict] = None):
        self._agents = agents or None
        self._config_list = config_list or None
        self._llm_config = llm_config or None

        self._initiator = agents[0] if agents else None
        self._recipient = agents[1] if agents and len(agents) > 1 else None

    async def initiate_work(self, message: str):
        if self._initiator and self._recipient:
            self._initiator.initiate_chat(recipient=self._recipient, message=message)
        else:
            raise ValueError("No initiator or recipient agent found.")