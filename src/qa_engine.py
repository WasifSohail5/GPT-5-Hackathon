def ask_dataset(question: str, df, client, memory=[]):  # ✅ added memory parameter
    schema = {
        "columns": df.dtypes.astype(str).to_dict(),
        "rows": df.head(3).to_dict(orient="records")
    }

    # Include conversation memory in the prompt
    memory_text = ""
    for entry in memory:
        memory_text += f"{entry['role'].capitalize()}: {entry['content']}\n"

    prompt = f"""
    You are a data assistant. The user uploaded a dataset.
    Dataset schema: {schema}

    Conversation so far:
    {memory_text}

    User question: {question}

    Based on the dataset schema and the conversation above,
    provide a conversational answer about the dataset.
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
