import asyncio
import time
import os
import pickle
from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
import aiofiles
from tqdm import tqdm


class QAPair(BaseModel):
    question: str = Field(description="The main question")
    answer: str = Field(description="The answer to the question")
    variations: List[str] = Field(description="Variations of the main question")


class QAOutput(BaseModel):
    qa_pairs: List[QAPair] = Field(description="List of QA pairs")


chunk_output_parser = PydanticOutputParser(pydantic_object=QAOutput)
row_output_parser = PydanticOutputParser(pydantic_object=QAOutput)
column_output_parser = PydanticOutputParser(pydantic_object=QAOutput)

prompt_components = {
    "base": {
        "intro": "Based exclusively on the following information, generate ALL reasonable questions that a user might ask related to it, along with an answer that should be around 2-5 sentences long. A response that is only one sentence long is WHOLLLY INSUFFICIENT.",
        "context": "Bear in mind that the user may not have read the information directly, so your task is to preempt their questions which can be answered by the information available.",
        "time_reference": "ALWAYS state in both the question and the answer the year/s under consideration.",
        "topic_reference": 'DO NOT create questions like "What is the subject matter of the information given?" or "In the given year...", because the user WOULD NOT ask that without having the information in front of them.',
        "variations": "In addition, you should provide 3-4 variations of each main question.",
        "info_placeholder": "Information: {info}",
        "format_instructions": "{format_instructions}",
        "reminder": "Remember to only use the information provided when choosing the questions and answers. Every question ABSOLUTELY MUST be answerable from the information provided, and be clearly grounded at a specific time.",
        "output_instruction": "IMPORTANT: In your response, include ONLY the JSON data that matches the required format. Do NOT include the schema definition or any other explanatory text.",
    },
    "row": {
        "focus": "Focus on events and details specific to the given time frame, including how events in different categories interact with each other.",
        "time_frame": "Time frame: {timeframe}",
    },
    "column": {
        "focus": "Ensure you highlight the progression or changes over time, with the details in your answer given chronologically and explicitly mentioning the time frame of each event.",
        "topic": "Topic: {topic}",
    },
}


def generate_prompt_template(prompt_type: str) -> str:
    base = prompt_components["base"]
    specific = prompt_components.get(prompt_type, {})

    template = f"{base['intro']}\n\n{base['context']}\n{base['time_reference']}\n{base['topic_reference']}\n\n{base['variations']}\n\n"

    if specific:
        template += f"{specific.get('focus', '')}\n\n"

    template += f"{base['info_placeholder']}\n\n"

    if specific:
        template += f"{specific.get('time_frame', '')}\n{specific.get('topic', '')}\n\n"

    template += f"{base['format_instructions']}\n\n{base['reminder']}\n\n{base['output_instruction']}"

    return template


# Generate prompt templates
chunk_template = generate_prompt_template("base")
row_template = generate_prompt_template("row")
column_template = generate_prompt_template("column")

# Create prompts
chunk_prompt = ChatPromptTemplate.from_template(chunk_template)
row_prompt = ChatPromptTemplate.from_template(row_template)
column_prompt = ChatPromptTemplate.from_template(column_template)


class Checkpoint(BaseModel):
    qa_docs: List[dict]
    failed_outputs: List[dict]
    processed_ids: List[str]


async def generate_qa(llm, info, prompt, parser, metadata):
    messages = prompt.format_messages(
        info=info, **metadata, format_instructions=parser.get_format_instructions()
    )
    output = await llm.ainvoke(messages)
    try:
        return parser.parse(output.content), None
    except OutputParserException as e:
        print("Error parsing output:", e)
        return None, (str(e), output.content)


async def save_checkpoint(checkpoint: Checkpoint, checkpoint_file: str):
    async with aiofiles.open(checkpoint_file, "w") as f:
        await f.write(checkpoint.model_dump_json())
    print(f"Checkpoint saved. Processed {len(checkpoint.processed_ids)} items.")


async def load_checkpoint(checkpoint_file: str) -> Checkpoint:
    if os.path.exists(checkpoint_file):
        async with aiofiles.open(checkpoint_file, "r") as f:
            content = await f.read()
        return Checkpoint.model_validate_json(content)
    return Checkpoint(qa_docs=[], failed_outputs=[], processed_ids=[])


async def process_qa_task(llm, task, checkpoint: Checkpoint, semaphore):
    async with semaphore:
        if task["id"] in checkpoint.processed_ids:
            return

        if task["type"] == "row":
            qa_output, error = await generate_qa(
                llm,
                task["info"],
                row_prompt,
                row_output_parser,
                {"timeframe": task["timeframe"]},
            )
        elif task["type"] == "column":
            qa_output, error = await generate_qa(
                llm,
                task["info"],
                column_prompt,
                column_output_parser,
                {"topic": task["topic"]},
            )
        else:  # chunk
            qa_output, error = await generate_qa(
                llm, task["info"], chunk_prompt, chunk_output_parser, {}
            )

        if qa_output is None:
            assert error is not None
            checkpoint.failed_outputs.append(
                {"error": error[0], "content": error[1], "metadata": task["metadata"]}
            )
        else:
            for j, qa_pair in enumerate(qa_output.qa_pairs):
                id = f"{task['id']}_{j}_qa"
                questions = [qa_pair.question] + qa_pair.variations
                metadata = {
                    "answer": qa_pair.answer,
                    "questions": questions,
                    "type": f"{task['type']}_qa",
                    "id": id,
                    **task["metadata"],
                }
                content = "Q: " + " Q: ".join(questions) + f"\nA: {qa_pair.answer}"
                qa_doc = {"page_content": content, "metadata": metadata, "id": id}
                checkpoint.qa_docs.append(qa_doc)

        checkpoint.processed_ids.append(task["id"])


async def process_for_qa(
    llm, documents, checkpoint_file="qa_checkpoint.json", max_concurrent=5
):
    checkpoint = await load_checkpoint(checkpoint_file)

    # Prepare tasks
    tasks = []
    timeframe_docs = {}
    topic_docs = {}

    for doc in documents:
        timeframe = doc.metadata["timeframe"]
        topic = doc.metadata["topic"]

        if timeframe not in timeframe_docs:
            timeframe_docs[timeframe] = []
        timeframe_docs[timeframe].append(doc)

        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append(doc)

        tasks.append(
            {
                "type": "chunk",
                "id": doc.id,
                "info": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    for timeframe, docs in timeframe_docs.items():
        combined_info = (
            "<document>\n"
            + "\n</document>\n\n<document>\n".join([doc.page_content for doc in docs])
            + "\n</document>"
        )
        tasks.append(
            {
                "type": "row",
                "id": f"row_{timeframe}",
                "info": combined_info,
                "timeframe": timeframe,
                "metadata": {"timeframe": timeframe},
            }
        )

    for topic, docs in topic_docs.items():
        combined_info = (
            "<document>\n"
            + "\n</document>\n\n<document>\n".join([doc.page_content for doc in docs])
            + "\n</document>"
        )
        tasks.append(
            {
                "type": "column",
                "id": f"column_{topic}",
                "info": combined_info,
                "topic": topic,
                "metadata": {"topic": topic},
            }
        )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_task(task, pbar):
        if task["id"] in checkpoint.processed_ids:
            print("Skipping already processed task:", task["id"])
            pbar.update(1)
            return

        start_time = time.time()
        print(f"Starting task: {task['id']}")
        await process_qa_task(llm, task, checkpoint, semaphore)
        end_time = time.time()
        print(f"Finished task: {task['id']} in {end_time - start_time:.2f} seconds")
        pbar.update(1)

        if len(checkpoint.processed_ids) % 10 == 0:
            await save_checkpoint(checkpoint, checkpoint_file)

    async def process_tasks():
        running_tasks = set()
        with tqdm(total=len(tasks), desc="Processing QA tasks") as pbar:
            for task in tasks:
                if len(running_tasks) >= max_concurrent:
                    # Wait for at least one task to complete
                    done, running_tasks = await asyncio.wait(
                        running_tasks, return_when=asyncio.FIRST_COMPLETED
                    )

                task_coroutine = process_task(task, pbar)
                running_tasks.add(asyncio.create_task(task_coroutine))

            # Wait for remaining tasks to complete
            if running_tasks:
                await asyncio.wait(running_tasks)

    await process_tasks()
    await save_checkpoint(checkpoint, checkpoint_file)

    qa_docs = [Document(**doc) for doc in checkpoint.qa_docs]
    return qa_docs, checkpoint.failed_outputs


if __name__ == "__main__":
    import dotenv
    from load_data import process_csv

    dotenv.load_dotenv()

    CSV_PATH = "./content/tables/airtable_v2.csv"  # can be a directory too
    QA_PICKLE = "qa_output.pkl"
    FAILURES_PICKLE = "failed_outputs.pkl"
    CHECKPOINT_FILE = "qa_checkpoint.json"

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        max_tokens_to_sample=8192,
    )

    csv_docs = process_csv(CSV_PATH)

    # NOTE if there are 429 rate limit errors, you can adjust the max_concurrent parameter
    # to a lower value. On the Tier 2 plan you can make 1,000 requests per minute, but Tier 1
    # is only 50, so it's worth just paying the $40 if you haven't already.
    # https://docs.anthropic.com/en/api/rate-limits
    qa_docs, failed_outputs = asyncio.run(
        process_for_qa(
            llm, csv_docs, checkpoint_file=CHECKPOINT_FILE, max_concurrent=50
        )
    )

    with open(QA_PICKLE, "wb") as f:
        pickle.dump(qa_docs, f)
    with open(FAILURES_PICKLE, "wb") as f:
        pickle.dump(failed_outputs, f)
