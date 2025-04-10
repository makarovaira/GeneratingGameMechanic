import json

from src.scripts.message import form_question, form_messages

def generate_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    training_data = []
    for name, description in data['mechanics'].items():
        assets, examples = description['assets'], description['examples']
        question = form_question(name, assets)
        for example_name, example_data in examples.items():
            training_data.append({"messages": form_messages(question, example_name, example_data)})
    return training_data

def prepare_data(dictionary_data, final_file_name):
    with open(final_file_name, 'w') as outfile:
        for entry in dictionary_data:
            json.dump(entry, outfile)
            outfile.write('\n')