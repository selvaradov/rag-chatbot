import pickle
import csv

with open("qa_output.pkl", "rb") as f:
    qa_docs, failed_outputs = pickle.load(f)

trimmed_docs = []
for qa_doc in qa_docs:
    trimmed_docs.append(qa_doc.page_content)

with open("qa_output_readable.csv", "w", newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Question", "Variations", "Answer"])

    for doc in qa_docs:
        entry = doc.page_content
        # Split the string into questions and answer
        qas = entry.split("\nA: ")
        questions = qas[0].split(" Q: ")[1:]  # Skip the first empty split result
        answer = qas[1]

        # Primary question is the first question
        question = questions[0]

        # Combine the rest as variations
        variations = ", ".join(questions[1:])

        # Write row to CSV
        csv_writer.writerow([question, variations, answer])
