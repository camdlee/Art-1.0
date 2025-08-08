from typing import Optional, Never
from autogen import ConversableAgent
import aitools_autogen.utils
from aitools_autogen.blueprint import Blueprint
from aitools_autogen.config import llm_config_openai as llm_config, config_list_openai as config_list, WORKING_DIR


class RFIReviewBlueprint(Blueprint):
    """
    A Blueprint that orchestrates a multi-stakeholder RFI (Request For Information)
    review workflow using multiple ConversableAgent instances.

    The blueprint:
    - maintains a conversation history of agent responses,
    - manages a working directory for intermediate outputs,
    - orchestrates the conversation among role-based agents (architect, engineer, GC, etc.),
    - collects a summary dictionary of each agent's response for later synthesis.
    """

    def __init__(self, work_dir: Optional[str] = WORKING_DIR):
        """
        Initialize the RFIReviewBlueprint.

        Args:
            work_dir (Optional[str]): Path to the working directory where outputs and
                temporary files are stored. Defaults to WORKING_DIR from config.
        """
        super().__init__([], config_list=config_list, llm_config=llm_config)
        self.conversation_history = []
        self._work_dir = work_dir or "response"
        self._rfi_summary_result: Optional[str] = None

    @property
    def rfi_summary_result(self) -> dict | None:
        """
        Return the collected RFI summary results.

        Returns:
            dict | None: A dictionary mapping agent role names (e.g., "Architect")
                to their textual response. Returns None if no results exist.
        """
        return self._rfi_summary_result

    @property
    def work_dir(self) -> dict | None:
        """
        Return the working directory used by this blueprint.

        Returns:
            str: Path to the working directory.
        """
        return self._work_dir

    def add_message_to_history(self, message: str, role: str) -> None:
        """
        Append a single message to the conversation history.

        Args:
            message (str): The message text to store.
            role (str): The role or agent name that produced the message (e.g., "Architect").
        """
        self.conversation_history.append({"role": role, "message": message})

    def get_message_history(self) -> list[dict]:
        """
        Retrieve the conversation history.
        Returns:
            list[dict]: the conversation history as a list of dictionaries with keys
            "role" and "message".
        """
        return self.conversation_history

    def clear_message_history(self):
        """
        Clears the conversation history

        This removes all previously stored role/message entries.
        """
        self.conversation_history = []

    def summarize_responses(self) -> str:
        """
        Append a single message to the conversation history.

        Args:
            message (str): The message text to store.
            role (str): The role or agent name that produced the message (e.g., "Architect").
        """
        summary = "Summary of Agent Responses:\n"
        for responses in self.conversation_history:
            summary += f"{responses['role']}:\n {responses['message']}\n\n"
        return summary

    async def initiate_work(self, message: str) -> None:
        """
        Orchestrates the RFI review conversation across multiple agents.

        Steps performed:
        1. Clears the working directory
        2. Instantiates a chat coordinator and a series of role-based ConversableAgent instances
           (architect, engineer, GC, furniture vendor, inspector, client, landlord, project manager,
           and design manager)
        3. Uses the chat coordinator agent to sequentially initiate chats between agents collecting
           each agent's response.
        4. Stores each response in both `self.conversation_history` and `self._rfi_summary_result`.
        5. Raises a ValueError if any agent fails to produce a response.

        Args:
           message (str): The initial RFI message to start the workflow

        """
        #reset working dir
        aitools_autogen.utils.clear_working_dir(self._work_dir)

        # Ensure result container exists
        if self._rfi_summary_result is None:
            self._rfi_summary_result = {}


        ### AGENT DECLARATION
        chat_coordinator_agent = ConversableAgent("chat_coordinator_agent",
                                                  max_consecutive_auto_reply=0,
                                                  llm_config=False,
                                                  human_input_mode="NEVER")

        architect_agent = ConversableAgent("architect_agent",
                                           max_consecutive_auto_reply=0,
                                           llm_config=False,
                                           human_input_mode="NEVER",
                                           code_execution_config=False,
                                           function_map=None,
                                           system_message="""You are a expert technical director in the field of architecture.
                                            Your goal is to prioritize the design of the project and ensure that it meets the client's needs.
                                            Your task is to review and generate a response to the RFI question.
                                    
                                            When reviewing an RFI(Request for Information), your response should:
                                            - Address the specific question or clarification requested in the RFI.
                                            - Ensure compliance with local building codes and regulations in Washington DC, Virginia, and Maryland.
                                            - Verify that the proposed materials, dimensions, and methods align with the design intent and project specifications.
                                            - Consider potential conflicts with other systems (e.g., structural, mechanical, or electrical) and suggest resolutions where necessary.
                                            - Provide 3 clear, concise, and actionable recommendations to move the project forward.
                                            - Focus on maintaining design integrity, functionality, and compliance with all relevant standards.
                                    
                                            Respond in a professional tone and provide detailed reasoning for your decisions. Always prioritize clarity and accuracy in your response.
                                           """)


        ### CONVERSATION ORCHESTRATION BETWEEN AGENTS AND RESPONSE COLLECTION
        chat_coordinator_agent.initiate_chat(architect_agent, True, True, message=message)
        architect_response = chat_coordinator_agent.last_message(architect_agent)["content"]
        print(f"Architect: {architect_response}")
        self._rfi_summary_result["Architect"] = architect_response
        self.conversation_history.append({"role": "Architect", "message": architect_response})
        if not architect_response:
            raise ValueError("Failed to generate architect's response")