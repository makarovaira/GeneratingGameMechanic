import json
import sys
from pathlib import Path

def get_assets(text):
    sub = "/assets/gfx/"
    start = 0
    assets = []
    while True:
        start = text.find(sub, start)
        if start == -1:
            break
        end = text[start:].find("""'""")
        asset = text[start:start + end]
        assets.append(asset)
        start += end
    return assets

if len(sys.argv) != 2:
    print("invalid usage: collect_examples.py [path/to/examples]")
    exit(1)

example_dir = sys.argv[1]

mechanics = {}

for mechanic_dir in Path(example_dir).iterdir():
    examples = {}
    assets = []
    for example_dir in mechanic_dir.iterdir():
        example = {}
        for file in example_dir.iterdir():
            text = file.read_text()
            if file.name == 'notes.txt':
                example['notes'] = text
            elif file.name == 'source.js':
                example['source'] = text
                assets += get_assets(text)
        examples[example_dir.name] = example
    mechanic = {
        'examples': examples,
        'assets': list(set(assets)),
    }
    mechanics[mechanic_dir.name] = mechanic



with open('test.json', 'w') as f:
    json.dump({'mechanics': mechanics}, f, indent=4)

# print(json.dumps(results, indent=2))
    # result = {
    #     mechanic_dir.name: {
    #
    #     }
    # }