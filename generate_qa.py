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


output_parser = PydanticOutputParser(pydantic_object=QAOutput)

template = """
Based exclusively on the following information, generate ALL reasonable questions that a user might ask related to it, along with an answer that should be around 2-3 sentences long. Bear in mind that the user may not have read the information directly, so your task is to preempt their questions which can be answered by the information available. Because of this, you MUST refer to years in absolute terms - if the information you have been given references a specific time period, ALWAYS state in both the question and the answer the year/s under consideration. The same is true about topics - DO NOT create questions like "What is the subject matter of the information given?", because the user would not ask that without having the information in front of them.

In addition, you should provide 3-4 variations of each main question.

Information: {info}

{format_instructions}

Remember to only use the information provided when choosing the questions and answers. Every question should be answerable from the information provided, and be clearly grounded in a specific year/s.

IMPORTANT: In your response, include ONLY the JSON data that matches the required format. Do NOT include the schema definition or any other explanatory text.
"""

prompt = ChatPromptTemplate.from_template(template)


def generate_qa(llm, info):
    messages = prompt.format_messages(
        info=info, format_instructions=output_parser.get_format_instructions()
    )
    print("messages:", messages)
    output = llm(messages)
    print("output:", output)
    try:
        return output_parser.parse(output.content), None
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

    for i, doc in enumerate(documents[start_index:], start=start_index):
        qa_output, error = generate_qa(llm, doc.page_content)

        if error:
            failed_outputs.append(
                {"error": error[0], "content": error[1], "metadata": doc.metadata}
            )
        else:
            for qa_pair in qa_output.qa_pairs:
                questions = [qa_pair.question] + qa_pair.variations

                metadata = {
                    "answer": qa_pair.answer,  # In case we want it separately later,
                    "questions": questions,  # ditto
                    "chunk": doc.page_content,  # This is useful because although the doc metadata will contain the `source` and also row, topic, etc, we may have chunked up the cell into several portions.
                    **doc.metadata,
                }

                content = "Q: " + " Q: ".join(questions) + f"\nA: {qa_pair.answer}"

                # Create a document for this QA
                qa_doc = Document(
                    page_content=content,
                    metadata=metadata,
                    id=f"{doc.id}_{i}",  # not sure what the best approach is here for IDs of the QAs, maybe just do a uuid?
                )
                qa_docs.append(qa_doc)

        # Save checkpoint every 'checkpoint_frequency' documents
        if (i + 1) % checkpoint_frequency == 0:
            save_checkpoint(qa_docs, failed_outputs, i, checkpoint_file)

    # Save final checkpoint
    save_checkpoint(qa_docs, failed_outputs, len(documents) - 1, checkpoint_file)

    return qa_docs, failed_outputs
