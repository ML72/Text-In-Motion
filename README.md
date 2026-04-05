# Autonomous Dance Identity

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
   - Download the SMPL models (Male, Female, and Neutral) from the [official SMPL website](https://smpl.is.tue.mpg.de/) and place the `.npz` files inside the `models/smpl/` folder.

4. **Create the Motion Index:**
   Before generating samples, create the index of moves by running:
   ```bash
   python create_index.py
   ```

## Generating Dance Routines

Once your setup is complete and the index is created, you can generate new dance samples by running:

```bash
python generate_sample.py
```

The generated samples will be saved in the `results/` directory. Each generated routine produces two files:
- A move sequence file (`.pkl`)
- A video visualization (`.mp4`)

