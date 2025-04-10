import json

prompt = """You are a professional game developer assistant, you should generate ideas of unique basic game mechanics, for 2d games, give answer using only JSON schema: "name" - type string, name of generated mechanic; "notes" - type string, description of generated mechanic; "source" - type string, full source code written in javascript using Phaser framework for 2d graphics, make sure that json is valid and javascript code will run in browser, you should prefer write more simple code and check that assets used in code are matching user specified paths or you will get punished by losing your job or salary"""

def form_question(name, assets):
    return f"""Generate mechanic for theme "{name}", for javascript code use only assets with these relative path {assets}, make sure that JSON answer is valid and java script code will not fail running in browser"""

def form_messages(question, example_name, example_data):
    answer = {
        'name': example_name,
        'notes': example_data['notes'],
        'source': example_data['source']
    }
    return [
        {'role': 'system', "content": prompt},
        {'role': 'user', "content": question},
        {'role': 'assistant', "content": json.dumps(answer)},
    ]