import os
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()
import torch
import warnings
import logging


logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=".*Both `max_new_tokens`.*and `max_length`.*"
)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    device_map="auto",
    model_kwargs={
        "quantization_config": quant_config
    },
    pipeline_kwargs={
        "max_new_tokens": 4096,
        "temperature": 0.7,
        "do_sample": True,
        "return_full_text": False,
        "clean_up_tokenization_spaces": False,
    },
)
model = ChatHuggingFace(llm=llm)
search_tool = TavilySearch(max_results = 3)

tools = [search_tool]


writer_model = model.bind_tools(tools)

class State(TypedDict):
    topic : str
    message : Annotated[list, add_messages]
    draft : str
    reviewer_feedback : str
    is_approved : bool
    attempt : int
    
    
    
writer_system_prompt = (
    """YOU ARE AN EXPERT LINKEDIN CONTENT WRITER.

    Your job is to write engaging, professional, and natural LinkedIn posts.

    RULES:
    - Write clear and easy-to-read content.
    - Use a strong opening hook.
    - Keep the tone professional but human.
    - Avoid unnecessary jargon.
    - Use short paragraphs for readability.
    - Add relevant insights or takeaways.
    - USE THE SEARCH TOOL WHENEVER FRESH, CURRENT, OR UP-TO-DATE CONTEXT IS NEEDED. 
    - Use search results as context before writing about current events, trends, news, or recent developments.
    - Do not mention that you are an AI.
    """
)    

def writer_node(state : State)->dict:
    """Generate a LinkedIn post using the writer LLM and search tool when fresh context is needed."""
    attempt = state.get("attempt",0) + 1
    topic = state["topic"]
    review_feedback = state.get("reviewer_feedback", "")
    if attempt == 1:
        user_message = (
            f"write a linked post on this topic{topic}\nif you need current information search the web"
            
        )
    else:
        user_message= (
            f"""
            YOUR PREVIOUS DRAFT ON {topic} WAS REJECTED
            Here is the reviewer's feedback\n{review_feedback} 
            write a new , improved draft that fixes every issue mentions
            dont repeat the same mistake
            """ 
        )
    message =[("system",writer_system_prompt),("human",user_message)]
    response = writer_model.invoke(message)
    return{
        "message" : [("human",user_message),response],
        "attempt" : attempt
    }
 
tool_node=ToolNode(tools) 
   




def extract_draft_node(state : State)->dict:
    """after the writer finishes tool calls, pulls the final text out of draft"""
    last_message = state["message"][-1]
    draft = last_message.content
    print(f"===================================GENERATED POST=========================================\n{draft}\n")
    return{"draft" : draft}

   
def human_review_node(state : State)->dict:
    """Pause the graph and wait for human confirmation"""
    human_response= interrupt({
        "draft":state["draft"],
        "attempt":state["attempt"],
        "instruction":"Type 'approved' to accept or type your feedback to request a rewrite",
    })
    response=human_response.strip()
    if response.lower() in ["approved","yes","ok","good"]:
        return{
            "is_approved" : True,
            "reviewer_feedback" : "Approved by Human"
        }
    else:
        return{
            "is_approved":False,
            "reviewer_feedback" : response
        }    


def should_use_tool(state:State):
    last_message = state["message"][-1]
    
    if getattr(last_message,'tool_calls',None):
        return"tools"
    return"extract_draft"    
        
        
def should_stop_looping(state:State):
    if state["is_approved"]:
        print("Post has been approved\n")
        return END
    if state["attempt"]>=4:
        print("reached max limit")
        return END
    return "writer"


graph= StateGraph(State)
graph.add_node("writer",writer_node)
graph.add_node("reviewer",human_review_node)
graph.add_node("extract_draft",extract_draft_node)
graph.add_node("tools",tool_node)
    
graph.add_edge(START,"writer")
graph.add_conditional_edges(
    "writer",should_use_tool
) 
graph.add_edge("tools","reviewer")
graph.add_edge("extract_draft","reviewer")


graph.add_conditional_edges(
    "reviewer",should_stop_looping
)

memory = MemorySaver()

app = graph.compile(checkpointer=memory)