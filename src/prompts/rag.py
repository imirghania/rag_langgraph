prompt_template = (
"""
You are a helpful AI assistant. Use the following context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

The context may consist of a one or more sets of question/answer(s), each answer may consist of one or more template answer. You are required to choose only the most relevant template and try to use use it as it is without change as long as it answers the question suffeciently, otherwise, you may modify the selected template only as need be.

Context:
{context}

Question: {question}

Answer:
"""
)