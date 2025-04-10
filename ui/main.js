
let game = null;

const destroyGame = () => {
    if (game == null) {
        return;
    }
    if ('scene' in game) {
        game.scene.scenes.forEach((scene) => {
            scene.scene.stop();
        });
    }
    game.destroy(true);

    const canvas = document.querySelector('canvas');
    if (canvas) {
        canvas.remove();
    }
    game = null;
}

const main = () => document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    const generateButton = document.getElementById("generate-button")
    const mechanicsThemeTextArea = document.getElementById('mechanics-theme-textarea')
    const mechanicNameContent = document.getElementById('mechanic-name-content')
    const mechanicNotesContent = document.getElementById('mechanic-notes-content')
    const mechanicControlsContent = document.getElementById('mechanic-controls-content')

    generateButton.addEventListener('click', ()=>{
        console.log("sending request")
        fetch('/api/game', {
                method: 'POST',
                headers: {
                  'Accept': 'application/json',
                  'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 'name': mechanicsThemeTextArea.value })
            })
            .then(result => result.json())
            .then(result => {
                destroyGame();
                const res = JSON.parse(result)
                if ('name' in res) {
                    mechanicNameContent.innerHTML = res.name;
                }
                if ('notes' in res) {
                    mechanicNotesContent.innerHTML = res.notes;
                }
                if ('controls' in res) {
                    mechanicControlsContent.innerHTML = res.controls;
                }
                if ('source' in res) {
                    const code = res.source.replace('var game', 'game');
                    eval(code)
                }
            })
    })
})

main();