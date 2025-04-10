import json
import uuid
from base64 import b64decode

import os
import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import OpenAI

from PIL import Image

client = OpenAI(
        base_url = "https://api.x.ai/v1",
        api_key=os.environ["GROK_API_KEY"],
)

template = """
var GameState = function(game) {
};

// Load images and sounds
GameState.prototype.preload = function() {
    // function implementation
    // load assets as in example this example `this.game.load.image('bullet', '/assets/gfx/bullet.png');`
};

// Setup the example
GameState.prototype.create = function() {
    // function implementation
};

GameState.prototype.shootBullet = function() {
    // function implementation
};

// The update() method is called every frame
GameState.prototype.update = function() {
    // function implementation
};

var game = new Phaser.Game(848, 450, Phaser.AUTO, 'game');
game.state.add('game', GameState, true);
"""

def make_prompt(assets):
# For javascript code you can use assets which described by json array in which we have json objects with keys "path" - is path to use in Phaser loading assets code(be sure you use exactly this path), "description" - is description of image"". This is assets which were described in previous sentence `{assets}`, use them in game code only if there matching game mechanics meaning.
    return f"""
You are a professional game developer assistant, you should generate ideas of unique basic game mechanics, 
for 2d games, give answer using only JSON schema: 
"name" - type string, name of generated mechanic; 
"notes" - type string, description of generated mechanic; 
"source" - type string, full source code written in javascript using Phaser framework for 2d graphics, make sure that json is valid and javascript code will run in browser, you should prefer write more simple code and check that assets used in code are matching user specified paths or you will get punished by losing your job or salary
"controls" - type string, description for how to control the game(keys, mouse and etc)
Important: Use only Phaser 2.0.7(link to release page at git https://github.com/phaserjs/phaser/tree/a6bc859246ca6cab7cd9fd06a8055a99fb6ab8e0) framework as in example, make sure that it will run in chromium based browsers and all the keys are actually evaluate actions, also check that every property that your using is defined, be completely sure that it would not break or you will be punished.
For games assets you can use images which have these paths: {assets}. Use them in game if there are matching game theme and use exact same paths as in the array.
Make sure that JSON answer is valid and java script code will not fail running in browser.
Here is the template "{template}", to write the code to, add code only to functions block which has `// function implementation` row.
Do not let the user go out of bounds of the drawing screen.
For jumping movement prefer "arrow up" key, but be sure jumping will work on my Phaser version 2.0.7
"""

def resize(path: str):
    im = Image.open(path)
    im.thumbnail((256, 256), Image.Resampling.LANCZOS)
    im.save(path, "JPEG")

def save_image(image: str):
    filename = str(uuid.uuid4()) + ".jpg"
    path = "/Users/pancakeswya/sunrise-infinite/ui/assets/" + filename

    with open(path, mode="wb") as f:
        f.write(b64decode(image))
    try:
        resize(path)
    except IOError:
        print(f"{path} failed to resize")
    return "/assets/" + filename

def query_assets(description: str):
    answer = client.images.generate(
        model="grok-2-image",
        prompt=f"""
            Generate 2d sprite for a game using this description:
            {description}
            """,
        n=4,
        response_format="b64_json"
    )
    assets = []
    for data in answer.data:
        path = save_image(data.b64_json)
        asset = {
            'path': path,
            'description': data.revised_prompt,
        }
        assets.append(asset)
    return assets


def query_generation(description: str):
    # answer = client.chat.completions.create(
    #     model="grok-2-latest",
    #     messages=[
    #         {"role": "system", "content":
    #             """
    #             You're analyst for a big game developer company,
    #             you need to generate unique game mechanic based description,
    #             generate simple description in one short sentence"""},
    #         {"role": "user", "content": f"Game Mechanic description: {description}"}
    #     ]
    # )
    # description = answer.choices[0].message.content
    print(description)
    question = f"""
        Generate simple demo using this description of mechanic: {description}.
    """
    assets = [
        "assets/gfx/bullet.png",
        "assets/gfx/flag.png",
        "assets/gfx/ball.png",
        "assets/gfx/ship.png",
        "assets/gfx/ball.svg",
        "assets/gfx/light.png",
        "assets/gfx/block.png",
        "assets/gfx/rocket.png",
        "assets/gfx/ground.png",
        "assets/gfx/smoke.png",
        "assets/gfx/blurred - circle.png",
        "assets/gfx/explosion.png",
        "assets/gfx/monster.png",
        "assets/gfx/player.png",
    ]
    answer = client.chat.completions.create(
        model="grok-2-latest",
        messages=[
            {"role": "system", "content": make_prompt(assets)},
            {"role": "user", "content": question}
        ],
        response_format={"type": "json_object"}
    )
    return answer.choices[0].message.content

app = FastAPI()

class Item(BaseModel):
    name: str

@app.post("/api/game")
async def get_game(item: Item):
    res = query_generation(item.name)
    return res

app.mount("/", StaticFiles(directory="/Users/pancakeswya/sunrise-infinite/ui", html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


