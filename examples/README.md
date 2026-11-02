# Examples

Start with an image edit. The CLI runs the prompt upsampler with your instruction and reference image before
Klein generates the result. Keep prompt upsampling enabled for editing, generation and layouts.

```bash
python scripts/cli.py "put a red bandana around the dog's neck" --model distilled --image examples/inputs/3.png --out bandana.png --seed 22
```

Fixed seeds, so each command reproduces the file it writes (run from the repo root).
The reproduction commands below create `inputs/2.png` and `inputs/3.png` before using them as edit references.
Each image's `.json` holds the prompt the model saw.

```bash
python scripts/cli.py "a white ceramic mug on a wooden table by a window, morning light" --out examples/inputs/2.png --seed 11
python scripts/cli.py "a golden retriever sitting on a sandy beach, overcast sky" --out examples/inputs/3.png --seed 12
python scripts/cli.py "A man walking with his dog, afternoon sun" --model base --out examples/base/man_dog.png --seed 7
python scripts/cli.py "turn this into a figurine on a desk" --model base --image examples/inputs/1.png --out examples/base/1_figurine.png --seed 33
python scripts/cli.py "fill the mug with coffee and put a croissant next to it" --model base --image examples/inputs/2.png --out examples/base/2_coffee.png --seed 21
python scripts/cli.py "put a red bandana around the dog's neck" --model base --image examples/inputs/3.png --out examples/base/3_bandana.png --seed 22
python scripts/cli.py "print the dog from the second image onto the mug" --model base --image examples/inputs/2.png --image examples/inputs/3.png --out examples/base/2_3_mugprint.png --seed 23
python scripts/cli.py "hot-air balloons over a green valley at sunrise" --model base --boxes examples/inputs/balloons_boxes.json --out examples/base/balloons.png --seed 1
python scripts/cli.py "add a small potted cactus in the marked spot" --model base --image examples/inputs/2.png --boxes examples/inputs/cactus_boxes.json --out examples/base/cactus.png --seed 1
python scripts/cli.py "A man walking with his dog, afternoon sun" --model distilled --out examples/distilled/man_dog.png --seed 7
python scripts/cli.py "turn this into a figurine on a desk" --model distilled --image examples/inputs/1.png --out examples/distilled/1_figurine.png --seed 33
python scripts/cli.py "fill the mug with coffee and put a croissant next to it" --model distilled --image examples/inputs/2.png --out examples/distilled/2_coffee.png --seed 21
python scripts/cli.py "put a red bandana around the dog's neck" --model distilled --image examples/inputs/3.png --out examples/distilled/3_bandana.png --seed 22
python scripts/cli.py "print the dog from the second image onto the mug" --model distilled --image examples/inputs/2.png --image examples/inputs/3.png --out examples/distilled/2_3_mugprint.png --seed 23
python scripts/cli.py "hot-air balloons over a green valley at sunrise" --model distilled --boxes examples/inputs/balloons_boxes.json --out examples/distilled/balloons.png --seed 1
python scripts/cli.py "add a small potted cactus in the marked spot" --model distilled --image examples/inputs/2.png --boxes examples/inputs/cactus_boxes.json --out examples/distilled/cactus.png --seed 1
```

| | input | base | distilled |
|---|---|---|---|
| figurine | <img src="inputs/1.png" width="220"> | <img src="base/1_figurine.png" width="220"> | <img src="distilled/1_figurine.png" width="220"> |
| coffee + croissant | <img src="inputs/2.png" width="220"> | <img src="base/2_coffee.png" width="220"> | <img src="distilled/2_coffee.png" width="220"> |
| red bandana | <img src="inputs/3.png" width="220"> | <img src="base/3_bandana.png" width="220"> | <img src="distilled/3_bandana.png" width="220"> |
| two references: dog printed on the mug | <img src="inputs/2.png" width="220"> <img src="inputs/3.png" width="220"> | <img src="base/2_3_mugprint.png" width="220"> | <img src="distilled/2_3_mugprint.png" width="220"> |
| layout: two balloons placed by [boxes](inputs/balloons_boxes.json) | — | <img src="base/balloons.png" width="220"> | <img src="distilled/balloons.png" width="220"> |
| layout edit: cactus added in a [box](inputs/cactus_boxes.json) | <img src="inputs/2.png" width="220"> | <img src="base/cactus.png" width="220"> | <img src="distilled/cactus.png" width="220"> |
| text-to-image | "A man walking with his dog, afternoon sun" | <img src="base/man_dog.png" width="220"> | <img src="distilled/man_dog.png" width="220"> |

## Typography

Distilled, with the upsampler; each `.json` holds the input prompt and the prompt the model saw.

```bash
python scripts/cli.py "a letterpress gig poster on cream stock, wood type: 'THE LOW TIDES' very large at the top, below it 'with special guests' small, then 'MOTH LIGHT' and 'HARBOUR SOUND' medium, at the bottom 'Saturday 21 March, Doors 8pm, The Old Chapel, Margate' and 'Tickets 12 pounds' in small caps, two inks, black and brick red, nothing else written" --model distilled --width 832 --height 1248 --seed 1 --out examples/typography/gig_poster.png
python scripts/cli.py "a 1960s cereal box, 'MORNING OATS' in bold rounded letters, 'with real honey' in a banner, a cartoon bear holding a spoon, 'NET WT 16 OZ' at the bottom corner and 'A good start to the day' in script, flat colours, photographed straight on" --model distilled --width 832 --height 1248 --seed 1 --out examples/typography/cereal_box.png
python scripts/cli.py "an open field notebook on a wooden table, handwritten in pencil: the heading 'Lake Vyrnwy, 14 October', then the lines 'wind NW, light', 'two herons on the far shore', 'water 9 degrees', 'the fog lifted at ten', a small pencil sketch of a heron, a pressed leaf, nothing else written" --model distilled --width 1248 --height 832 --seed 2 --out examples/typography/field_notes.png
python scripts/cli.py "a tea tin, dark green enamel: 'HIGHLAND BREAKFAST' in gold caps on the front, 'Assam and Ceylon leaf' beneath, 'est. 1884' in a roundel, and on the side in small white type 'Brew for four minutes. One spoon per cup and one for the pot.', hard side light, product photograph" --model distilled --width 1024 --height 1024 --seed 1 --out examples/typography/tea_tin.png
```
