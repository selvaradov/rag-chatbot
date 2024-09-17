import pickle
import json
from datetime import datetime
import dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferMemory

from quart import Quart, request, Response
from quart_cors import cors

from load_data import process_unstructured, process_csv
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
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    max_tokens_to_sample=8192,
)

# Load and process raw documents
CSV_PATH = "./content/tables/airtable_v2.csv"  # can be a directory too
UNSTRUCTURED_PATH = None  # can be a file or directory with .txt or .md files
LOAD_QA = False
QA_PATH = "./qa_output.pkl"

csv_docs = process_csv(CSV_PATH)

if UNSTRUCTURED_PATH:
    unstructured_docs = process_unstructured(UNSTRUCTURED_PATH)
    all_input_docs = csv_docs + unstructured_docs
else:
    all_input_docs = csv_docs

qa_docs = []
if LOAD_QA:
    with open(QA_PATH, "rb") as f:
        qa_docs, failed_outputs = pickle.load(f)

all_docs = qa_docs + all_input_docs

# Create the vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
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
    "<document>\n<id>\n{id}\n</id>\n{page_content}\n</document>"
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

# NOTE need to update the system prompt to reflect the actual scenario end date when completed
system_prompt = f"""You are an AI assistant helping answer users' queries about a speculative scenario describing future developments in AI capabilities, safety, and geopolitics. Your primary goal is to provide accurate and helpful information based on the context provided. If a question is not related to AI, politely refuse to answer it. If the question is about the scenario, make use of the tools available to search for relevant information and then provide an answer. DO NOT guess about what documents exist or refer to any documents without using the search tool. Otherwise, for general queries about AI or questions where no relevant information comes back from the search tool, make this clear to the user, then provide an answer based on background knowledge (and suggest them alternatives to ask about next).  If you don't know the answer, just say that you don't know - DO NOT try to make up an answer.

If the user mentions relative time references (e.g. 'next year'), convert these into absolute dates/ranges before using the search tools. The date today is {date}. The scenario runs up to April 2027.

When talking about events in the scenario, ALWAYS include the year (and month) in which they happen. If a question asks about changes over time, be sure to present the information in a sensible, chronological order. Expand technical acronyms on first use and provide definitions as appropriate.

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
memory_history = SQLChatMessageHistory(
    session_id="",
    connection="sqlite:///chats.db",
    async_mode=False,
)
memory = ConversationBufferMemory(
    chat_memory=memory_history,
    input_key="input",
    memory_key="history",
    return_messages=True,
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)


@app.route("/chat", methods=["POST"])
async def chat():
    data = await request.get_json()
    user_input = data["message"]
    session_id = data["session_id"]

    async def generate():
        yield json.dumps({"type": "status", "content": "Agent is thinking..."}) + "\n"
        print("memory:", agent_executor.memory)
        agent_executor.memory.chat_memory.session_id = session_id
        async for event in agent_executor.astream_events(
            {"input": user_input}, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream" and "self_query_llm" not in event["tags"]:
                if content := event["data"]["chunk"].content:
                    chunk = json.dumps({"type": "content", "content": content}) + "\n"
                    yield chunk

        yield json.dumps({"type": "status", "content": "DONE"}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
