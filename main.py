from dotenv import load_dotenv
from typing import TypedDict,Annotated,List

# langgraph
from langgraph.graph import StateGraph,state,END

# langchain
from langchain_groq import ChatGroq

llm_model = ChatGroq(
    model='openai/gpt-oss-20b'
)

print(llm_model.invoke('what is you name').content)