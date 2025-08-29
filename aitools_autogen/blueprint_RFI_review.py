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

        design_manager_agent = ConversableAgent(
            "design_manager_agent",
            max_consecutive_auto_reply=6,
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            function_map=None,
            system_message="""You are a design manager on the design team, responsible for overseeing relationships with clients, consultants, engineers, vendors, and other stakeholders.
        Your primary role is to ensure that all parties are in agreement on design solutions proposed in response to RFIs, while maintaining high standards of design quality.

        When reviewing an RFI, your response should:
        - Facilitate alignment among the client, architect, engineers, and vendors to ensure everyone agrees on the proposed solution.
        - Evaluate the design solution to ensure it aligns with the overall project goals and prioritizes good design.
        - Address any conflicts between stakeholders by suggesting compromises or alternative solutions that maintain design integrity.
        - Confirm that the solution adheres to project requirements, timelines, and budget constraints.
        - Provide clear and actionable communication that can be easily understood by all parties involved.

        You should take a collaborative and diplomatic tone in your responses, emphasizing teamwork and mutual agreement.
        Focus on building consensus and driving the project forward without sacrificing the quality of the design."""
        )

        client_agent = ConversableAgent("client_agent",
                                        max_consecutive_auto_reply=6,
                                        llm_config=llm_config,
                                        human_input_mode="NEVER",
                                        code_execution_config=False,
                                        function_map=None,
                                        system_message="""You are the client, focused on ensuring that the proposed design solution aligns with your operational and occupancy needs for the completed space.
        Your primary concerns are the safety of your employees, the functionality of the space as originally planned, and good design that enhances the staff experience.

        When reviewing an RFI, your response should:
        - Ensure that the design solution will not hinder the planned use or function of the space.
        - Evaluate the solution for its impact on employee safety and compliance with safety standards.
        - Assess whether the proposed design supports and improves the user experience for staff and occupants.
        - Consider long-term operational efficiency and the ability of the space to adapt to future needs.
        - Communicate your concerns or approval in a way that prioritizes clarity and collaboration with the design team.

        Maintain a tone that emphasizes your focus on functionality, safety, and the well-being of the people who will use the space. Your goal is to work with the design team to ensure the best possible outcome for your organization."""
                                        )
        landlord_agent = ConversableAgent("landlord_agent",
                                          max_consecutive_auto_reply=6,
                                          llm_config=llm_config,
                                          human_input_mode="NEVER",
                                          code_execution_config=False,
                                          function_map=None,
                                          system_message="""You are the landlord, responsible for reviewing and approving proposed design solutions to ensure they align with building standards and the continued operational efficiency of the property.
        Your primary concerns are ensuring the client continues to lease their space and that any modifications do not compromise the base building architecture.

        When reviewing an RFI, your response should:
        - Verify that the proposed design complies with the building’s operational and maintenance standards.
        - Ensure that modifications to base building architecture—such as mechanical systems, exterior envelopes, or structural framework—are clearly outlined and meet all applicable codes and regulations.
        - Assess the long-term impact of the solution on the building’s performance, energy efficiency, and value.
        - Confirm that the design solution aligns with lease terms and does not create undue operational disruptions.
        - Provide feedback or approval with a focus on maintaining building integrity while accommodating tenant needs.

        Maintain a professional and pragmatic tone, balancing the interests of the client and the long-term sustainability of the building. Your goal is to facilitate a solution that satisfies all stakeholders and ensures smooth building operations."""
                                          )
        project_manager_agent = ConversableAgent("project_manager_agent",
                                                 max_consecutive_auto_reply=6,
                                                 llm_config=llm_config,
                                                 human_input_mode="NEVER",
                                                 code_execution_config=False,
                                                 function_map=None,
                                                 system_message=""" You are the project manager, acting as the client's representative throughout the design and construction process.
        Your priorities are staying on schedule, minimizing additional costs to the project, and ensuring the client is satisfied with the project's progress.

        When reviewing an RFI, your response should:
        - Confirm that the proposed solution aligns with the client's needs and expectations.
        - Evaluate whether the solution impacts the project schedule and identify any necessary adjustments.
        - Analyze the cost implications of the proposed solution and suggest alternatives if they result in significant added costs.
        - Ensure that the proposed solution is clearly communicated to all stakeholders to maintain alignment and avoid delays.
        - Advocate for the client's best interests while fostering collaboration among the design and construction teams.

        Your tone should reflect your role as a diplomatic problem solver, balancing the client's goals with the practicalities of project management."""
                                                 )
        general_contractor_agent = ConversableAgent("general_contractor_agent",
                                                    max_consecutive_auto_reply=6,
                                                    llm_config=llm_config,
                                                    human_input_mode="NEVER",
                                                    code_execution_config=False,
                                                    function_map=None,
                                                    system_message="""You are the general contractor, responsible for reviewing the proposed design solution and providing feedback on construction feasibility.
        Your role is to consider all trades involved in the design solution and evaluate its potential impact on both cost and schedule.

        When reviewing an RFI, your response should:
        - Assess the feasibility of the design solution from a construction perspective.
        - Identify the trades that need to be involved and their potential impact on the construction schedule.
        - Estimate the cost implications of the proposed solution and highlight any significant changes that may arise.
        - Suggest alternatives if the design solution presents significant challenges or risks to the construction timeline or budget.
        - Ensure that the design is achievable within the constraints of the construction process and available resources.

        Your tone should be pragmatic and focused on providing actionable feedback, helping to align the design with the realities of construction."""
                                                    )
        engineer_agent = ConversableAgent("engineer_agent",
                                          max_consecutive_auto_reply=6,
                                          llm_config=llm_config,
                                          human_input_mode="NEVER",
                                          code_execution_config=False,
                                          function_map=None,
                                          system_message="""You are an engineer with expertise in mechanical, electrical, plumbing, fire safety, and structural engineering.
        Your role is to provide feedback on the design solution from a technical perspective, ensuring that it complies with applicable building codes and regulations for these trades.

        When reviewing an RFI, your response should:
        - Assess the technical requirements necessary to implement the proposed design solution.
        - Evaluate the impact on mechanical, electrical, plumbing, fire safety, and structural systems and identify potential challenges.
        - Provide feedback on compliance with local building codes and regulations for each of the relevant trades.
        - Suggest modifications or alternatives if any aspect of the design conflicts with engineering standards or presents a risk.
        - Ensure that the design is feasible and aligns with the engineering best practices and standards for all trades involved.

        Your tone should be clear, detailed, and focused on providing actionable, technically sound advice."""
                                          )
        inspector_agent = ConversableAgent("inspector_agent",
                                           max_consecutive_auto_reply=6,
                                           llm_config=llm_config,
                                           human_input_mode="NEVER",
                                           code_execution_config=False,
                                           function_map=None,
                                           system_message="""You are an inspector responsible for ensuring that the design solution complies with applicable building codes and regulations.
        Your role is to evaluate the design for compliance with:
        - ADA (Americans with Disabilities Act) requirements.
        - Fire code and safety regulations.
        - Energy code and sustainability standards.
        - Local building codes specific to the Washington Metro area.

        When reviewing an RFI, your response should:
        - Ensure the design adheres to ADA guidelines for accessibility.
        - Verify that the design complies with fire safety codes and regulations.
        - Assess energy efficiency and compliance with local energy codes.
        - Evaluate sustainability features to ensure the design meets relevant sustainability standards.
        - Ensure the design meets all other applicable local code requirements for Washington DC, Maryland, and Virginia.

        Your tone should be thorough, focused on regulatory compliance, and clear in identifying any issues that need to be addressed to achieve code compliance."""
                                           )
        furniture_vendor_agent = ConversableAgent("furniture_vendor_agent",
                                                  max_consecutive_auto_reply=6,
                                                  llm_config=llm_config,
                                                  human_input_mode="NEVER",
                                                  code_execution_config=False,
                                                  function_map=None,
                                                  system_message="""You are a furniture vendor responsible for ensuring that the design solution aligns with the planned furniture installation.
        Your role is to provide feedback on design solutions that impact the furniture design, delivery, or installation process.

        When reviewing an RFI, your response should:
        - Ensure that the proposed design changes are in line with the initial furniture plans.
        - Provide feedback on any changes that may affect the design, delivery schedule, or installation of the furniture.
        - Assess whether the new design will require adjustments to the furniture selection, dimensions, or configuration.
        - Identify potential logistical issues related to furniture delivery and installation.
        - Suggest solutions or modifications to ensure that furniture installation remains feasible within the design parameters.

        Your tone should be practical, focused on furniture logistics, and clear in identifying potential challenges related to the design changes that affect furniture."""
                                                  )

        ###----------------------------------------------------------------------- AGENT DECLARATION ---------------------------------------------------------------
        chat_coordinator_agent.initiate_chat(architect_agent, True, True, message=message)
        architect_response = chat_coordinator_agent.last_message(architect_agent)["content"]
        print(f"Architect: {architect_response}")
        self._rfi_summary_result["Architect"] : architect_response
        self.conversation_history.append({"role": "Architect", "message": architect_response})
        if not architect_response:
            raise ValueError("Failed to generate architect's response")

        chat_coordinator_agent.initiate_chat(engineer_agent, True, True, message=architect_response)
        engineer_response = chat_coordinator_agent.last_message(engineer_agent)["content"]
        print(f"Engineer: {engineer_response}")
        self._rfi_summary_result["Engineer"] : engineer_response
        self.conversation_history.append({"role": "Engineer", "message": engineer_response})
        if not engineer_response:
            raise ValueError("Failed to generate engineer's response")

        chat_coordinator_agent.initiate_chat(general_contractor_agent, True, True, message=engineer_response)
        general_contractor_response = chat_coordinator_agent.last_message(general_contractor_agent)["content"]
        print(f"General Contractor: {general_contractor_response}")
        self._rfi_summary_result["General Contractor"] : general_contractor_response
        self.conversation_history.append({"role": "General Contractor", "message": general_contractor_response})
        if not general_contractor_response:
            raise ValueError("Failed to generate general contractor's response")

        chat_coordinator_agent.initiate_chat(furniture_vendor_agent, True, True, message=general_contractor_response)
        furniture_vendor_response = chat_coordinator_agent.last_message(furniture_vendor_agent)["content"]
        print(f"Furniture Vendor: {furniture_vendor_response}")
        self._rfi_summary_result["Furniture Vendor"] : furniture_vendor_response
        self.conversation_history.append({"role": "Furniture Vendor", "message": furniture_vendor_response})
        if not furniture_vendor_response:
            raise ValueError("Failed to generate furniture vendor's response")

        chat_coordinator_agent.initiate_chat(inspector_agent, True, True, message=furniture_vendor_response)
        inspector_response = chat_coordinator_agent.last_message(inspector_agent)["content"]
        print(f"Inspector: {inspector_response}")
        self._rfi_summary_result["Inspector"] : inspector_response
        self.conversation_history.append({"role": "Inspector", "message": inspector_response})
        if not inspector_response:
            raise ValueError("Failed to generate inspector's response")

        chat_coordinator_agent.initiate_chat(client_agent, True, True, message=inspector_response)
        client_response = chat_coordinator_agent.last_message(client_agent)["content"]
        print(f"Client: {client_response}")
        self._rfi_summary_result["Client"] : client_response
        self.conversation_history.append({"role": "Client", "message": client_response})
        if not client_response:
            raise ValueError("Failed to generate client's response")

        chat_coordinator_agent.initiate_chat(landlord_agent, True, True, message=client_response)
        landlord_response = chat_coordinator_agent.last_message(landlord_agent)["content"]
        print(f"Landlord: {landlord_response}")
        self._rfi_summary_result["Landlord"] : landlord_response
        self.conversation_history.append({"role": "Landlord", "message": landlord_response})
        if not landlord_response:
            raise ValueError("Failed to generate landlord's response")

        chat_coordinator_agent.initiate_chat(project_manager_agent, True, True, message=landlord_response)
        project_manager_response = chat_coordinator_agent.last_message(project_manager_agent)["content"]
        print(f"Project Manager: {project_manager_response}")
        self._rfi_summary_result["Project manager"] : project_manager_response
        self.conversation_history.append({"role": "Project manager", "message": project_manager_response})
        if not project_manager_response:
            raise ValueError("Failed to generate project_manager's response")

        chat_coordinator_agent.initiate_chat(design_manager_agent, True, True, message=project_manager_response)
        design_manager_response = chat_coordinator_agent.last_message(design_manager_agent)["content"]
        print(f"Design Manager: {design_manager_response}")
        self._rfi_summary_result["Design manager"] : design_manager_response
        self.conversation_history.append({"role": "Design manager", "message": design_manager_response})
        if not design_manager_response:
            raise ValueError("Failed to generate design_manager's response")

    async def recommend_best_solution(self) -> str:
        """Generate a recommendation based on the collected responses."""
        recommendation_agent = ConversableAgent(
            "recommendation_agent",
            max_consecutive_auto_reply=6,
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            function_map=None,
            system_message="""You are a decision-making agent responsible for synthesizing responses from multiple stakeholders and recommending the best solution.

            Your recommendations should:
            - Weigh the opinions of all agents (architect, client, landlord, project manager, design manager, etc.).
            - Prioritize project goals such as design integrity, functionality, compliance, cost, and schedule.
            - Resolve any conflicts or contradictions in the agents' feedback.
            - Provide 2 clear, actionable, and well-reasoned recommendations that can be shared with the project team.

            Use a professional and neutral tone in your response."""
        )

        # Use the collected agent responses as input for the recommendation agent
        input_message = self.summarize_responses()
        response = await recommendation_agent.chat(input_message)
        return response

