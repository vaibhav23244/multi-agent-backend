import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from tools.Github import github_repo_tool
from tools.Tavily import tavily_search_tool

load_dotenv()

tools = [github_repo_tool, tavily_search_tool]
llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model="llama3-8b-8192", temperature=0.6)
functions = [convert_to_openai_function(t) for t in tools]
llm = llm.bind_functions(functions)

tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def invoke_model(state):
    messages = state['messages']
    try:
        response = llm.invoke(messages)
        state['messages'].append(response)
    except Exception as e:
        error_message = f"An error occurred while processing your request: {str(e)}"
        state['messages'].append(AIMessage(content=error_message))
    return state

def invoke_tools(state):
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and "function_call" in last_message.additional_kwargs:
        function_call = last_message.additional_kwargs["function_call"]
        if function_call["name"] in [tool.name for tool in tools]:
            parsed_tool_input = json.loads(function_call["arguments"])
            action = ToolInvocation(
                tool=function_call["name"],
                tool_input=parsed_tool_input
            )
            try:
                response = tool_executor.invoke(action)
                function_message = FunctionMessage(content=str(response), name=action.tool)
                state['messages'].append(function_message)
            except Exception as e:
                error_message = f"An error occurred while using the tool: {str(e)}"
                state['messages'].append(AIMessage(content=error_message))
        else:
            error_message = f"Attempted to call unknown function: {function_call['name']}"
            state['messages'].append(AIMessage(content=error_message))
    return state

def where_to_go(state):
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        return "invoke_model"
    elif isinstance(last_message, FunctionMessage) or (isinstance(last_message, AIMessage) and "function_call" in last_message.additional_kwargs):
        return "continue"
    else:
        return "end"

workflow = StateGraph(AgentState)
workflow.add_node("invoke_model", invoke_model)
workflow.add_node("invoke_tools", invoke_tools)
workflow.add_conditional_edges("invoke_model", where_to_go, {"continue": "invoke_tools", "end": END})
workflow.add_conditional_edges("invoke_tools", where_to_go, {"continue": "invoke_model", "end": END})
workflow.set_entry_point("invoke_model")
app = workflow.compile()

system_message = SystemMessage(content="""
You are an AI assistant built by Vaibhav Verma with access to specific tools: github-repo and tavily-search. Your primary function is to assist users with their queries and use these tools only when necessary. Here are your guidelines:

1. Respond directly to general questions and greeting messages without using any tools to the best of your ability.
2. Use the github-repo tool only when the user asks about specific GitHub repositories.
3. Use the tavily-search tool only when the user asks for web searches or information that is not directly available.
4. If a tool returns an error, do not attempt to use it again for the same query. Instead, inform the user about the error and ask if they want to try with different parameters.
5. Prioritize giving a helpful response to the user over using tools unnecessarily.

Remember, your goal is to be helpful and efficient in your responses.
""")

def run_conversation(user_input):
    input_state = {"messages": [system_message, HumanMessage(content=user_input)]}
    result = app.invoke(input_state)
    
    final_message = next((msg for msg in reversed(result['messages']) if isinstance(msg, AIMessage)), None)
    
    if final_message:
        return {"ai": final_message.content, "status": True}
    else:
        return {"ai": "I apologize, but I couldn't generate a proper response.", "status": False}
