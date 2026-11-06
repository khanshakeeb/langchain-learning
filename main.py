from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def main():
    llm = init_chat_model("ollama:llama3.1:8b", temperature=0)
    response = llm.invoke("Say 'Hello from LangChain!' and nothing else.")
    print(response.content)


if __name__ == "__main__":
    main()
