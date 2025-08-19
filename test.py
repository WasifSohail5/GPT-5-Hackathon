from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

chat = ChatOpenAI(
    openai_api_base="https://api.aimlapi.com/v1",
    openai_api_key="361a27a60f6b4138982fd15278917fed",
    model="openai/gpt-5-chat-latest"
)
response = chat.invoke([HumanMessage(content="What is the capital of Pakistan?")])
print(response.content)