from src.groq_client import get_groq_client
from src.dataset_loader import load_dataset
from src.qa_engine import ask_dataset
import os

def main():
    # Init Groq client
    client = get_groq_client()

    

    # Ask user for dataset file path
    while True:
        file_path = input("Enter the dataset file path (or type 'exit' to quit): ").strip()
        if file_path.lower() == "exit":
            print("👋 Goodbye!")
            return
        if not os.path.exists(file_path):
            print("❌ File not found. Please enter a valid path.")
            continue
        try:
            df = load_dataset(file_path)
            conversation_memory = []
            break
        except Exception as e:
            print(e)
            continue



    # Interactive Q&A loop
    while True:
        question = input("\nAsk about your dataset (or type 'exit'): ").strip()
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break

        try:
            # answer = ask_dataset(question, df, client)
            # print("💡 Answer:", answer)
            # Pass memory to ask_dataset
            answer = ask_dataset(question, df, client, conversation_memory)
            print("💡 Answer:", answer)

            # ✅ Store user question and bot answer in memory
            conversation_memory.append({"role": "user", "content": question})
            conversation_memory.append({"role": "assistant", "content": answer})
        except Exception as e:
            print("❌ Error while processing your question:", e)

if __name__ == "__main__":
    main()