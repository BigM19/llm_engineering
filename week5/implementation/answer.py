from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gemini-2.5-flash-lite"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
RETRIEVAL_K = 10

SYSTEM_PROMPT_ANSWER = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

SYSTEM_PROMPT_SEARCH = """
You are a helpful assistant that focuses on formatting user questions for retrieval.
You are given a question and the conversation history.
Your task is to combine the question and the conversation history into a single string
that can be used for retrieval.
The conversation history might not be relevant to the latest question, so only include
the parts of the conversation that are relevant to the question.
If the conversation history is not relevant, just return the question.
Only return the combined string; do not include any explanations or formatting.
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(temperature=0, model=MODEL)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

def create_search_query(question: str, history: list[dict] = []) -> str:
    """
    Create a search query for retrieval based on the question and conversation history.
    """
    prior_user_messages = "\n".join(m["content"] for m in history)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_SEARCH),
        HumanMessage(content=f"Previous questions: {prior_user_messages}"),
        HumanMessage(content=f"Current question: {question}")
    ]
    response = llm.invoke(messages)
    return response.content


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    query = create_search_query(question, history)
    docs = fetch_context(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_ANSWER.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
