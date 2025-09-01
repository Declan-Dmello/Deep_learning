from rag_pipeline import rag_response

print("Conversational RAG System (type 'exit' to quit)")

while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        break
    answer = rag_response(query)
    print("Bot:", answer)
