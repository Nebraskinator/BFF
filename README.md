# Abiogenesis Simulation

This simulation models an abiogenetic process using a grid-based instruction sequence system with simple rule sets for head movement, tape modification, and loop handling. The simulation runs iteratively, with options for saving state and visualizations at specified intervals.

Project inspired by [BFF Family](https://arxiv.org/pdf/2406.19108) simulations

# Example runs:

[Spontaneous Emergence of Replicators](https://www.youtube.com/watch?v=P-fpHKOhPSg)

Settings Used:

```bash
python abiogenesis.py --height 128 --width 256 --depth 64 --num_instructions 64 --num_sims 5000000
```

 


[Extended Run](https://www.youtube.com/watch?v=zefGNLQRyCY)

Isolated Replicators: coming soon

Settings Used:

```bash
python abiogenesis.py --height 256 --width 512 --depth 64 --num_instructions 64 --num_sims 24000000 --stateful_heads True
```


## Requirements

- Python 3.x
- PyTorch
- NumPy
- OpenCV

## Installation

Make sure you have the required packages installed. You can install them using pip:

```bash
pip install torch numpy opencv-python
```

## Usage

Run the simulation from the command line with customizable parameters:

```bash
python abiogenesis.py [OPTIONS]
```

### Command-Line Arguments

- `--height`: Height of the tape (default: 128).
- `--width`: Width of the tape (default: 256).
- `--depth`: Depth of the tape (default: 64).
- `--num_instructions`: Number of unique instructions (default: 64).
- `--device`: Device to run the simulation on (`cpu` or `cuda`, default: `cuda`).
- `--num_sims`: Number of simulation iterations (default: 1,000,000).
- `--mutate_rate`: Mutation rate during simulation (default: 0.0).
- `--results_path`: Path to save results, including images and state files (default: `results/run_0`).
- `--image_save_interval`: Interval (in iterations) to save visualizations as images (default: 500).
- `--state_save_interval`: Interval (in iterations) to save the simulation state (default: 100,000).
- `--stateful_heads`: Only allow heads to reset position when executing the terminal instruction (default: False).
- `--load`: Resume run from checkpoint. Must enter a path to a valid checkpoint (default: '').

### Example Usage

1. **Run the simulation on GPU with default parameters:**

    ```bash
    python abiogenesis.py
    ```

2. **Run the simulation on CPU with a custom mutation rate and save results to a specified path:**

    ```bash
    python abiogenesis.py --device cpu --mutate_rate 0.001 --results_path results/run_1
    ```

3. **Set the height, width, depth of the tape and run 5 million iterations:**

    ```bash
    python abiogenesis.py --height 64 --width 128 --depth 32 --num_sims 5000000
    ```

4. **Save images every 1000 iterations and states every 200,000 iterations:**

    ```bash
    python abiogenesis.py --image_save_interval 1000 --state_save_interval 200000
    ```

### Output Directory Structure

If a results path is specified (e.g., `results/run_0`), the following subdirectories will be created automatically:

- `img/`: Contains the PNG images generated during the simulation, with filenames zero-padded to 9 digits (e.g., `000000001.png`).
- `states/`: Contains serialized state files of the simulation, saved as `.p` files with zero-padded filenames (e.g., `000000001.p`).

### Notes

- The simulation supports CUDA acceleration, making it highly efficient on compatible GPUs.
- Images and states are saved at user-specified intervals, allowing for checkpoints and visual analysis of the simulation's progress.
- The tape system is fully parameterized, allowing for detailed control over simulation behavior and complexity.
