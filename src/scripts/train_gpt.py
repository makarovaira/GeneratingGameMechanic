import datetime
import json
import os
import signal
import time

from openai import OpenAI

from src.scripts.dataset import generate_dataset, prepare_data


def main():
    training_data = generate_dataset("/Users/pancakeswya/sunrise-infinite/test.json")
    client = OpenAI(
        base_url = "https://api.x.ai/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    training_file_name = "training_data.jsonl"

    prepare_data(training_data, training_file_name)

    training_file_id = client.files.create(
        file=open(training_file_name, "rb"),
        purpose="fine-tune"
    )
    print(f"Training File ID: {training_file_id}")

    response = client.fine_tuning.jobs.create(
        training_file=training_file_id.id,
        model="grok-2-latest",
        hyperparameters = {
            "n_epochs": 40,
        }
    )
    job_id = response.id
    status = response.status

    print(f'Fine-tunning model with jobID: {job_id}.')
    print(f"Training Response: {response}")
    print(f"Training Status: {status}")

    def signal_handler(sig, frame):
        status = client.fine_tuning.jobs.retrieve(job_id).status
        print(f"Stream interrupted. Job is still {status}.")
        return

    print(f"Streaming events for the fine-tuning job: {job_id}")

    signal.signal(signal.SIGINT, signal_handler)

    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
    try:
        for event in events:
            print(
                f'{datetime.datetime.fromtimestamp(event.created_at)} {event.message}'
            )
    except Exception:
        print("Stream interrupted (client disconnected).")

    status = client.fine_tuning.jobs.retrieve(job_id).status
    if status not in ["succeeded", "failed"]:
        print(f"Job not in terminal status: {status}. Waiting.")
        while status not in ["succeeded", "failed"]:
            time.sleep(2)
            jobs = client.fine_tuning.jobs.retrieve(job_id)
            status = client.fine_tuning.jobs.retrieve(job_id).status
            if (jobs.error is not None) and (jobs.error.code is not None):
                print(f"Metadata: {jobs.metadata}, Error: {jobs.error}")
            print(f"Status: {status}")
    else:
        print(f"Finetune job {job_id} finished with status: {status}")
    print("Checking other finetune jobs in the subscription.")
    result = client.fine_tuning.jobs.list()
    print(f"Found {len(result.data)} finetune jobs.")

    fine_tuned_model = result.data[0].fine_tuned_model
    print(fine_tuned_model)
if __name__ == "__main__":
    main()