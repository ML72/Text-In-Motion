# Text in Motion: Visualizing Prose as Stylistic Dance Sequences

This codebase provides a framework for generating new dance routines using motion matching on the AIST++ dataset.

## Setup

Follow these steps to set up the environment and prepare the data:

1. **Set up the Python Environment:**
   Set up your environment with Python 3.12. If you are using Conda, it's recommended to create a new environment:
   ```bash
   conda create -n dance_identity python=3.12
   conda activate dance_identity
   ```

2. **Install Dependencies:**
   Install all required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data and Models:**
   - Ensure your AIST++ dataset is located inside the `data/motions/` folder. This step should already be done, just confirm that `data/motions/` is populated.
   - Download the SMPL models (Male, Female, and Neutral) from the [official SMPL website](https://smpl.is.tue.mpg.de/) and place them inside the `models/smpl/` folder. Ensure the following 6 files are present in the directory so the scripts can properly load all genders and metadata:
     - `SMPL_FEMALE.npz`
     - `SMPL_FEMALE.pkl`
     - `SMPL_MALE.npz`
     - `SMPL_MALE.pkl`
     - `SMPL_NEUTRAL.npz`
     - `SMPL_NEUTRAL.pkl`

4. **Create the Motion Index:**
   Create the index of moves by running:
   ```bash
   python create_index.py
   ```

5. **Create the Motion Codebook:**
   Create the discretized feature codebook by running:
   ```bash
   python create_codebook.py
   ```

6. **Create the Plausibilities Graph:**
   Create the graph of transition plausibilities by running:
   ```bash
   python create_plausibilities.py
   ```

## Generating Dance Routines

Once your setup is complete and the index is created, you can generate new dance samples in two ways:

**Mode A (Autonomous Exploration):**
Generate a random sequence of a given length:
```bash
python generate_sample.py --num_frames 1000
```

**Mode B (Guided Generation):**
Guide generation through specific codebook regions using a DNA sequence (a comma-separated string of region IDs):
```bash
python generate_sample.py --input_dna "114, 12, 125, 140, 57"
```

**Mode C (Text Guided Generation):**
Guide generation using a string of text. The text is trimmed and converted into bytes, which directly map to codebook regions:
```bash
python generate_sample.py --input_text "I love dance"
```

All modes support an optional `--gender` argument (`neutral`, `male`, or `female`) to change the body model used during rendering. For example:
```bash
python generate_sample.py --input_text "I love dance" --gender female
```

The generated samples will be saved in the `results/` directory. Each generated routine produces three files:
- A move sequence file (`.pkl`)
- A video visualization (`.mp4`)
- A run data file (`.json`)

## Webapp

This repository contains code for a website about this project. All website code is in the `webapp` folder. To interact with the website code, switch into the `webapp` directory:

```bash
cd webapp
```

Read `webapp/README.md` for instructions on how to set up and run the website locally.

## How This Works

### Indexing and Codebook Creation

To build the searchable database of motions, the system first creates a unified index mapping all valid frames (`create_index.py`). Then, it processes these frames into abstract stylistic tokens (`create_codebook.py`):
1. **Feature Extraction:** It reconstructs 3D representations using the SMPL body model and extracts local behavior features (joint poses, root velocities, and foot contacts) while discarding world-space position variations to make the data translation-invariant.
2. **Temporal Windowing:** A sliding window captures chunks of movement (default 20 frames) representing the "stylistic future" of each frame.
3. **Dimensionality Reduction and Quantization:** The high-dimensional temporal features are compressed into a 64D space using PCA, and subsequently assigned discrete region values (0 to 255) using K-Means clustering.

**Output:**
- The combined motion index is saved to `data/index/motion_index.npz`, containing the concatenated SMPL `poses` and `trans` for all dataset frames, along with `file_indices` and `frame_indices` to map each frame back to its original source file and local frame number.
- The codebook is saved to `data/index/codebook.npz`. Every valid frame receives an integer token `0-255` reflecting its motion behavior cluster. The final `W-1` (default 19) frames of any isolated clip are marked with a token of `-1`, signifying that they lack sufficient future frames to construct a full behavioral window. The downstream engine ignores these `-1` frames as selectable transitional targets.

### Plausibility Graph Construction

The plausibility graph is generated using `create_plausibilities.py`. This graph encodes the physical plausibility of transitions between codebook regions. Each node represents a codebook region, and edges are weighted by the cost of transitioning between regions based on motion continuity metrics. This graph is critical for ensuring smooth transitions during guided generation (Mode B) and is saved as `data/index/plausibility_graph.pkl`.

### Sample Generation

The sample generation step (`generate_sample.py`) operates in two modes using motion matching and an offline plausibility graph:

* **Mode A (Autonomous Exploration):** The engine starts at a random valid frame and plays the motion. To maintain temporal consistency, it locks playback for a minimum of 30 frames. After this period, it searches the entire database to find the lowest-cost transition to a new sequence, continuously creating a novel, unbounded dance routine.
* **Mode B (Guided Generation):** The engine follows a user-provided "DNA" target sequence, represented as a list of codebook regions. It seamlessly transitions to the requested regions. If a direct transition to the next requested region is physically implausible within the search window, the system performs a safe jump and uses Dijkstra's algorithm on the precomputed plausibility graph (`create_plausibilities.py`) to inject bridge regions, dynamically routing the choreography back to the user's intended DNA sequence.
* **Mode C (Text Guided Generation):** The engine follows a target sequence generated by trimming the provided text and converting it into a string of bytes. Each byte perfectly maps to a 0-255 codebook region, providing a new way to interactively guide choreography through words.

In all modes, the final output includes a 3D rendered video, raw pose data, and a `run_data.json` file logging the exact sequence of regions executed.
