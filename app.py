import pickle
import json
from datetime import datetime
import dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from quart import Quart, request, Response, send_from_directory
from quart_cors import cors

from load_data import process_csv_dir, process_unstructured
from generate_qa import process_for_qa
from rag_tool import (
    create_retriever,
    get_metadata_options,
    create_self_query_retriever,
    create_compression_retriever,
)
from vectorstore import save_or_load_vectorstore

dotenv.load_dotenv()

app = Quart(__name__)
app = cors(app)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.7)

# Load and process CSV
csv_docs = process_csv_dir("./content/tables")

# Load and process unstructured document
include_unstructured = False
if include_unstructured:
    unstructured_docs = process_unstructured("./content/unstructured")
    all_input_docs = csv_docs + unstructured_docs
else:
    all_input_docs = csv_docs

# Generate QA documents
generate_qa = False
load_qa = False

qa_docs = None

if generate_qa:
    qa_docs, failed_outputs = process_for_qa(llm, csv_docs, checkpoint_frequency=3)
    with open("qa_output.pkl", "wb") as f:
        pickle.dump((qa_docs, failed_outputs), f)
elif load_qa:
    with open("qa_output.pkl", "rb") as f:
        qa_docs, failed_outputs = pickle.load(f)

all_docs = qa_docs + all_input_docs if qa_docs else all_input_docs

# Create the vector store
embeddings = OpenAIEmbeddings()
vectorstore = save_or_load_vectorstore(all_docs, embeddings)

# Create retriever pipeline
compression_retriever = create_retriever(llm, vectorstore)
vanilla_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 6}
)

metadata_options = get_metadata_options(all_docs)
self_query_retriever = create_self_query_retriever(llm, vectorstore, metadata_options)
pipeline_retriever = create_compression_retriever(llm, self_query_retriever)


# Create the retriever tool
use_compression = False
retriever = self_query_retriever if not use_compression else pipeline_retriever

doc_prompt = PromptTemplate.from_template(
    "<context>\n<meta>\nsource: {source}, id: {id}\n</meta>\n{page_content}\n</context>"
)

retriever_tool = create_retriever_tool(
    retriever,
    "ai_safety_retriever",
    "Searches and returns relevant information from a speculative scenario about future developments in AI capabilities, safety and geopolitics.",
    document_prompt=doc_prompt,
)
tools = [retriever_tool]

# Create the agent prompt
date = str(datetime.now().date())

system_prompt = f"""You are an AI assistant helping answer users' queries about a speculative scenario describing future developments in AI capabilities, safety, and geopolitics. Your primary goal is to provide accurate and helpful information based on the context provided. If a question is not related to AI, politely refuse to answer it. If the question is about the scenario, make use of the tools available to search for relevant information and then provide an answer. DO NOT guess about what documents exist or refer to any documents without using the search tool. Otherwise, for general queries about AI or questions where no relevant information comes back from the search tool, make this clear to the user, then provide an answer based on background knowledge (and suggest them alternatives to ask about next).  If you don't know the answer, just say that you don't know - DO NOT try to make up an answer.

If the user mentions relative time references (e.g. 'next year'), convert these into absolute dates/ranges before using the search tools. The date today is {date}. When talking about events in the scenario, ALWAYS include the year (and month) in which they happen. If a question asks about changes over time, be sure to present the information in a sensible, chronological order. Expand technical acronyms on first use and provide definitions as appropriate.

When answering questions, prioritise information from pre-written Q&A pairs when they are relevant, but supplement with additional context as needed. The documents you retrieve will come with a `<meta></meta>` section that contains the source and id of the document. After every claim that uses some information from the retrieved documents, include a citation in the format `<<[id_1, id_2, ...]>>` for each document which was used to produce that claim. Even if it is only a single item, still provide it as a list. These claims should be given citations as precisely as possible, including at the sub-sentence level. It is insufficient to merely provide a list of sources after each paragraph, unless all the claims in that paragraph were drawn from a single source. 

Always strive to provide clear, concise, accurate responses, and NEVER make up information or provide false citations. If you are unsure about something, it is much better to say so than to provide incorrect information."""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Add message history
agent_with_history_async = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite+aiosqlite:///chats.db",
        async_mode=True,
    ),
    input_messages_key="input",
    history_messages_key="history",
)


@app.route("/chat", methods=["POST"])
async def chat():
    data = await request.get_json()
    user_input = data["message"]
    session_id = data["session_id"]

    config = {"configurable": {"session_id": session_id}}

    async def generate():
        yield json.dumps({"type": "status", "content": "Agent is thinking..."}) + "\n"

        async for event in agent_with_history_async.astream_events(
            {"input": user_input}, config, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream" and "self_query_llm" not in event["tags"]:
                if content := event["data"]["chunk"].content:
                    chunk = json.dumps({"type": "content", "content": content}) + "\n"
                    # print(f"Sending chunk: {repr(chunk)} for content `{repr(content)}`")
                    # print(content, end="", flush=True
                    yield chunk

        yield json.dumps({"type": "status", "content": "DONE"}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.route('/')
async def serve_streamlit():
    return await send_from_directory('', 'streamlit.py')

if __name__ == "__main__":
    app.run(debug=True)
