from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from pydantic import ValidationError
import json
import os
from langchain.schema import Document


class QAPair(BaseModel):
    question: str = Field(description="The main question")
    answer: str = Field(description="The answer to the question")
    variations: List[str] = Field(description="Variations of the main question")


class QAOutput(BaseModel):
    qa_pairs: List[QAPair] = Field(description="List of QA pairs")

class RowQAPair(BaseModel):
    question: str = Field(description="The main question about a specific timeframe")
    answer: str = Field(description="The answer to the question")
    variations: List[str] = Field(description="Variations of the main question")

class ColumnQAPair(BaseModel):
    question: str = Field(description="The main question about changes over time")
    answer: str = Field(description="The answer to the question")
    variations: List[str] = Field(description="Variations of the main question")

class RowQAOutput(BaseModel):
    qa_pairs: List[RowQAPair] = Field(description="List of row-wise QA pairs")

class ColumnQAOutput(BaseModel):
    qa_pairs: List[ColumnQAPair] = Field(description="List of column-wise QA pairs")

chunk_output_parser = PydanticOutputParser(pydantic_object=QAOutput)
row_output_parser = PydanticOutputParser(pydantic_object=RowQAOutput)
column_output_parser = PydanticOutputParser(pydantic_object=ColumnQAOutput)

chunk_template = """
Based exclusively on the following information, generate ALL reasonable questions that a user might ask related to it, along with an answer that should be around 2-3 sentences long. Bear in mind that the user may not have read the information directly, so your task is to preempt their questions which can be answered by the information available. Because of this, you MUST refer to years in absolute terms - if the information you have been given references a specific time period, ALWAYS state in both the question and the answer the year/s under consideration. The same is true about topics - DO NOT create questions like "What is the subject matter of the information given?", because the user would not ask that without having the information in front of them.

In addition, you should provide 3-4 variations of each main question.

Information: {info}

{format_instructions}

Remember to only use the information provided when choosing the questions and answers. Every question should be answerable from the information provided, and be clearly grounded at a specific time.

IMPORTANT: In your response, include ONLY the JSON data that matches the required format. Do NOT include the schema definition or any other explanatory text.
"""
row_template = """
Based on the following information about a specific time frame, generate ALL reasonable questions (roughly 20) that a user might ask, along with clear, accurate answers based solely on the information provided. Focus on events and details specific to the given time frame, including how events in different categories interact with each other. Because the user may not have read the information themself, you must mention the time frame in both the question and the answer. DO NOT create questions like "What is the subject matter of the information given?", because the user would not ask that without having the information in front of them.

In addition, you should provide 3-4 variations of each main question.

<information>
{info}
</information>

Time frame: {timeframe}

{format_instructions}

Remember to only use the information provided when choosing the questions and answers. Every question should be answerable from the information provided and be clearly grounded at a specific time.

IMPORTANT: In your response, include ONLY the JSON data that matches the required format. Do NOT include the schema definition or any other explanatory text.
"""

column_template = """
Based on the following information about a specific topic over time, generate ALL reasonable questions (roughly 20) a user might ask, along with clear, accurate answers based solely on the information provided. Ensure you highlight the progression or changes over time, with the details in your answer given chronologically and explicitly mentioning the time frame of each event. Because the user may not have read the information themselves, you must mention the topic in both the question and the answer. DO NOT create questions like "What is the subject matter of the information given?", because the user would not ask that without having the information in front of them.

In addition, you should provide 3-4 variations of each main question.

Information: {info}

Topic: {topic}

{format_instructions}

Remember to only use the information provided and focus on the changes and progression over time.

IMPORTANT: In your response, include ONLY the JSON data that matches the required format. Do NOT include the schema definition or any other explanatory text.
"""

chunk_prompt = ChatPromptTemplate.from_template(chunk_template)
row_prompt = ChatPromptTemplate.from_template(row_template)
column_prompt = ChatPromptTemplate.from_template(column_template)


def generate_chunk_qa(llm, info):
    messages = chunk_prompt.format_messages(
        info=info, format_instructions=chunk_output_parser.get_format_instructions()
    )
    output = llm.invoke(messages)
    try:
        return chunk_output_parser.parse(output.content), None
    except ValidationError as e:
        return None, (str(e), output.content)
    
def generate_row_qa(llm, info, timeframe):
    messages = row_prompt.format_messages(
        info=info, timeframe=timeframe, format_instructions=row_output_parser.get_format_instructions()
    )
    output = llm.invoke(messages)
    try:
        return row_output_parser.parse(output.content), None
    except ValidationError as e:
        return None, (str(e), output.content)

def generate_column_qa(llm, info, topic):
    messages = column_prompt.format_messages(
        info=info, topic=topic, format_instructions=column_output_parser.get_format_instructions()
    )
    output = llm.invoke(messages)
    try:
        return column_output_parser.parse(output.content), None
    except ValidationError as e:
        return None, (str(e), output.content)


def save_checkpoint(qa_docs, failed_outputs, last_processed_index, checkpoint_file):
    serializable_qa_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in qa_docs
    ]

    checkpoint = {
        "qa_docs": serializable_qa_docs,
        "failed_outputs": failed_outputs,
        "last_processed_index": last_processed_index,
    }

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)

    print(f"Checkpoint saved. Last processed index: {last_processed_index}")


def process_for_qa(
    llm, documents, checkpoint_file="qa_checkpoint.json", checkpoint_frequency=25
):
    qa_docs = []
    failed_outputs = []

    # Load from checkpoint if it exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        qa_docs = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in checkpoint["qa_docs"]
        ]
        failed_outputs = checkpoint["failed_outputs"]
        start_index = checkpoint["last_processed_index"] + 1
        print("Starting from", start_index)
        print("QA docs:", qa_docs)
    else:
        start_index = 0

    # Group documents by year and topic
    timeframe_docs = {}
    topic_docs = {}

    for doc in documents:
        timeframe = doc.metadata["date"]
        topic = doc.metadata["topic"]
        
        if timeframe not in timeframe_docs:
            timeframe_docs[timeframe] = []
        timeframe_docs[timeframe].append(doc)
        
        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append(doc)

    
    # Process row-wise (by year)
    for year, docs in timeframe_docs.items():
        combined_info = "\n\n-----\n\n".join([doc.page_content for doc in docs])
        qa_output, error = generate_row_qa(llm, combined_info, year)
        ## debug line
        print(f"Processed year: {year}, QA Output: {qa_output}, Error: {error}")
        
        if qa_output is None:
            assert error is not None
            failed_outputs.append({"error": error[0], "content": error[1], "metadata": {"year": year}})
        else:
            for j, qa_pair in enumerate(qa_output.qa_pairs):
                id = f"{year}_{j}_qa"
                questions = [qa_pair.question] + qa_pair.variations
                metadata = {
                    "answer": qa_pair.answer,
                    "questions": questions,
                    "year": year,
                    "type": "row_qa",
                    "id": id,
                }
                content = "Q: " + " Q: ".join(questions) + f"\nA: {qa_pair.answer}"
                qa_doc = Document(page_content=content, metadata=metadata, id=id)
                qa_docs.append(qa_doc)

        if len(qa_docs) % checkpoint_frequency == 0:
            save_checkpoint(qa_docs, failed_outputs, len(qa_docs) - 1, checkpoint_file)

    # Process column-wise (by topic)
    for topic, docs in topic_docs.items():
        combined_info = "\n".join([f"{doc.metadata['date']}: {doc.page_content}" for doc in docs])
        qa_output, error = generate_column_qa(llm, combined_info, topic)
        print(f"Processed topic: {topic}, QA Output: {qa_output}, Error: {error}")
        
        if qa_output is None:
            assert error is not None
            failed_outputs.append({"error": error[0], "content": error[1], "metadata": {"topic": topic}})
        else:
            for j, qa_pair in enumerate(qa_output.qa_pairs):
                id = f"{topic}_{j}_qa"
                questions = [qa_pair.question] + qa_pair.variations
                metadata = {
                    "answer": qa_pair.answer,
                    "questions": questions,
                    "topic": topic,
                    "type": "column_qa",
                    "id": id,
                }
                content = "Q: " + " Q: ".join(questions) + f"\nA: {qa_pair.answer}"
                qa_doc = Document(page_content=content, metadata=metadata, id=id)
                qa_docs.append(qa_doc)

        if len(qa_docs) % checkpoint_frequency == 0:
            save_checkpoint(qa_docs, failed_outputs, len(qa_docs) - 1, checkpoint_file)

    # Process each cell chunk individually
    for i, doc in enumerate(documents[start_index:], start=start_index):
        qa_output, error = generate_chunk_qa(llm, doc.page_content)
        print(f"Processed chunk: {doc.id}, QA Output: {qa_output}, Error: {error}")

        if qa_output is None:
            assert error is not None
            failed_outputs.append(
                {"error": error[0], "content": error[1], "metadata": doc.metadata}
            )
        else:
            for j, qa_pair in enumerate(qa_output.qa_pairs):
                id = f"{doc.id}_{j}_qa"
                questions = [qa_pair.question] + qa_pair.variations

                metadata = {
                    "answer": qa_pair.answer,  # In case we want it separately later,
                    "questions": questions,  # ditto
                    "chunk": doc.page_content,  # This is useful because although the doc metadata will contain the `source` and also row, topic, etc, we may have chunked up the cell into several portions.
                    "type": "chunk_qa",
                    "id": id,
                    **doc.metadata,
                }

                content = "Q: " + " Q: ".join(questions) + f"\nA: {qa_pair.answer}"

                # Create a document for this QA
                qa_doc = Document(
                    page_content=content,
                    metadata=metadata,
                    id=id,  # not sure what the best approach is here for IDs of the QAs, maybe just do a uuid? This is easier for debugging though
                )
                qa_docs.append(qa_doc)

        # Save checkpoint every 'checkpoint_frequency' documents
        if (i + 1) % checkpoint_frequency == 0:
            save_checkpoint(qa_docs, failed_outputs, i, checkpoint_file)

    # Save final checkpoint
    save_checkpoint(qa_docs, failed_outputs, len(documents) - 1, checkpoint_file)

    return qa_docs, failed_outputs

if __name__ == "__main__":
    import pickle
    import json
    import dotenv

    from langchain_openai import ChatOpenAI

    from load_data import process_csv_dir
    from generate_qa import process_for_qa

    dotenv.load_dotenv()

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.7)

    # Load and process CSV
    csv_docs = process_csv_dir("./content/tables")

    qa_docs, failed_outputs = process_for_qa(llm, csv_docs, checkpoint_frequency=3)
    with open("qa_output.pkl", "wb") as f:
        pickle.dump((qa_docs, failed_outputs), f)