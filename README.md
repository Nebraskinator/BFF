# Abiogenesis Simulation

Abiogenesis refers to the emergence of life from non-living matter. This process has never been directly observed, and the conditions necessary for abiogenesis remain unknown. Earth is the only planet where life has been found, but it's also the only planet we've explored in detail, leaving open the question of whether life is common or rare in the universe.

At the heart of this mystery are two contrasting ideas about how life emerges: (1) given enough complexity and time, life is inevitable, or (2) life arose from an extremely rare chance event. These two perspectives offer very different views on pre-biotic Earth. Life as we know it requires a complex and specific arrangement of chemicals—an arrangement that seems highly unlikely to occur by pure chance. However, given the vast size of the universe, it's possible that stochastic processes resulted in life, or at least a self-replicating precursor, by a fluke. Alternatively, life may have been the result of natural processes that tend to accumulate the necessary components for self-replication, steadily increasing the likelihood of life until it became almost inevitable.

This simulation explores which "rules" and environments result in an attraction toward self-replication. It features a grid of simple instructions that modify themselves and their surroundings. Depending on the initial conditions and constraints, self-replicating instruction sequences can spontaneously emerge and evolve, helping to provide insight into the conditions that might have given rise to life.

This project was inspired by [BFF](https://arxiv.org/pdf/2406.19108) simulations.

# Results:

The simulation begins with a grid of randomized instruction sequences. Each sequence controls two read-write heads that are spatially constrained to their local area. These heads can move and modify sequences based on the instructions they execute.

In this simulation, there are 64 possible instruction values, but only 20 of them correspond to specific actions:

- Moving one of the heads by 1 position in the x, y, or z direction (12 instructions in total)
- Incrementing or decrementing the value at the position of one of the heads (4 instructions)
- Entering or exiting a loop in the instruction sequence, based on the value at the first head (2 instructions)
- Copying a value from one head to the other (2 instructions)

[This video](https://www.youtube.com/watch?v=P-fpHKOhPSg) demonstrates the spontaneous emergence and evolution of self-replicating instruction sequences under these conditions. As the simulation progresses, the sequences display more complex and effective replication mechanisms. [This video](https://www.youtube.com/watch?v=xsQBqFXrb7Q) shows the replication kinetics of sequences isolated from key epochs and allowed to grow in a sandbox environment.

In these visualizations, each instruction sequence is depicted as a small square, with different colors representing various instruction types. [This video](https://www.youtube.com/watch?v=TP4nlFbBFIQ) demonstrates how a self-replicating sequence interacts with its neighboring positions in the simulation. The orange faces highlight the locations of the read-write heads, while the intersecting red lines point to the specific instruction currently being executed. This visualization provides a glimpse into how sequences modify themselves and their environment, revealing the dynamics of replication and interaction within the grid.

Settings Used:
```bash
python bff_2d.py --height 128 --width 256 --depth 64 --num_instructions 64 --num_sims 5000000
```

The dynamics of self-replicating sequence emergence can be influenced by applying different constraints to the environment. In [this particular simulation](https://www.youtube.com/watch?v=zefGNLQRyCY), the read-write heads do not reset to their "home" positions when the instruction sequence encounters a crash. This forces self-replicating sequences to execute without logical errors.

Settings Used:
```bash
python bff_2d.py --height 256 --width 512 --depth 64 --num_instructions 64 --num_sims 24000000 --stateful_heads True
```

In each of these simulations, there appears to be attraction toward replication due to the rapid emergence of replicating sequences. One could argue that the copy instruction acts as a catalyst for replication by allowing sequences to duplicate key instructions and propagate more easily. In simulations without this instruction, self-replicating sequences fail to arise spontaneously, supporting this hypothesis. In [this experiment](https://www.youtube.com/watch?v=C-p72uG-gEk), handed-coded replication sequences seeded into the environment quickly terminate, indicating that while replication is theoretically possible, the absence of the copy instruction prevents the robust, sustained replication necessary for long-term survival and evolution in this environment. 

Unlike this simplified system, the natural world operates without a basic "copy instruction." Replication in biology is an incredibly complex process. More work is necessary to identify the conditions necessary for spontaneous emergence of replication in the absence of a copy instruction.

Settings Used:
```bash
python bff_2d.py --height 256 --width 512 --depth 64 --num_instructions 64 --num_sims 24000000 --loop_condition both --mutate_rate 0.00001 --no_copy --seed 200000
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
python bff_2d.py [OPTIONS]
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
- `--loop_condition`: Condition loop on 0 at head0 'value' or matching the values at both heads 'match' (default: `value`)
- `--loop_option`: Adds an additional set of loop instructions with opposite conditions (default: False)
- `--no_copy`: Removes copy operations from instruction set (default: False)
- `--seed`: Number of hand-coded replicators to seed into the simulation (default: 0)
- `--color_scheme`: `default`, `random`, or `dark` color scheme for visualizations (default: `default`)

### Example Usage

1. **Run the simulation on GPU with default parameters:**

    ```bash
    python bff_2d.py
    ```

2. **Run the simulation on CPU with a custom mutation rate and save results to a specified path:**

    ```bash
    python bff_2d.py --device cpu --mutate_rate 0.001 --results_path results/run_1
    ```

3. **Set the height, width, depth of the tape and run 5 million iterations:**

    ```bash
    python bff_2d.py --height 64 --width 128 --depth 32 --num_sims 5000000
    ```

4. **Save images every 1000 iterations and states every 200,000 iterations:**

    ```bash
    python bff_2d.py --image_save_interval 1000 --state_save_interval 200000
    ```

### Output Directory Structure

If a results path is specified (e.g., `results/run_0`), the following subdirectories will be created automatically:

- `img/`: Contains the PNG images generated during the simulation, with filenames zero-padded to 9 digits (e.g., `000000001.png`).
- `states/`: Contains serialized state files of the simulation, saved as `.p` files with zero-padded filenames (e.g., `000000001.p`).

### Notes

- The simulation supports CUDA acceleration, making it highly efficient on compatible GPUs.
- Images and states are saved at user-specified intervals, allowing for checkpoints and visual analysis of the simulation's progress.
- The tape system is fully parameterized, allowing for detailed control over simulation behavior and complexity.
