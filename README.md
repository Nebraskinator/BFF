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

#### Command Line Usage

You can run the simulation directly from the command line using a configuration file or by specifying parameters via command-line arguments. Here are some basic examples of how to use the command-line interface:

1. **Run a Simulation with Default Parameters:**

```bash
python bff_2d.py
```

2. **Specify Custom Parameters in the Command Line:**

```bash
python bff_2d.py --height 128 --width 256 --sequence_length 64 --num_heads 3 --num_sims 1000000 --mutate_rate 0.00001 --results_path results/custom_run
```

3. **Load a Configuration File:**

```bash
python bff_2d.py --config path/to/config.ini
```

#### Instruction Set

The simulation supports a rich set of customizable instructions that control both the internal behavior of the sequences and their interactions with neighboring sequences:

- **Movement Instructions:**
  - The read/write heads can move along three dimensions within a configurable range. Each head can be moved by a unit step in either direction. These instructions enable heads to navigate their local environment, influencing which instructions or data they modify.
  - Altering the number of heads and their maximum range of travel:
    ```bash
    python abiogenesis.py --num_heads 4 --head_range 2
    ```
  - **Instruction Example:** `head0_dim1_move+1` moves the first head one step in the positive `y` direction.

- **Value Modification:**
  - Heads can directly modify the value at their current position by incrementing or decrementing the instruction stored there. These modifications allow heads to alter their own sequence or nearby sequences, creating potential mutations or changes in behavior.
  - **Instruction Example:** `head1_value-1` decrements the value at the position of the second head by one.

- **Looping Mechanisms:**
  - Sequences can create conditional loops based on the values encountered by the heads. Loops can be conditioned either on the presence of a specific value (e.g., zero) or on matching values between two heads. This flexibility allows for complex behaviors, such as conditional repetition or decision-making based on local data.
  - Example command to alter looping conditions and number of conditions:
    ```bash
    python abiogenesis.py --loop_condition both --loop_options 5
    ```
  - **Instruction Example:** `loop_skip_if_head0_==_0` skips the loop if head 0 encounters a value of zero.

- **Copying Between Heads:**
  - One of the most powerful instructions in the simulation is the **copy** operation, where the value at one head can be copied to the position of another head. This operation allows sequences to duplicate instructions or share information between different parts of the grid, driving the replication process.
  - Example command to run a simulation without copy instructions:
    ```bash
    python abiogenesis.py --no_copy True
    ```
  - **Instruction Example:** `copy_head0_to_head1` copies the value at head 0’s position to head 1’s position.

- **No-Operation (NOP):**
  - In addition to active instructions, the grid can include "no-op" values, which are essentially placeholders that can store data without causing any action to occur. These no-ops can be dynamically changed by heads as needed.
  - The number of no-ops is equal to the num_instructions value minus the number of operative instructions. The number of operative instructions depends on the number of heads and the number of loop options.
    ```bash
    python abiogenesis.py --num_instructions 128
    ```

#### Other Simulation Features

- **Head Reset and Error Handling:**
  - The behavior of the read/write heads can be customized to either reset to a home position after a crash or continue executing, simulating different strategies for error handling in evolving systems. Options include resetting on termination, on encountering invalid instructions, or never resetting.
  - Example command:
    ```bash
    python abiogenesis.py --head_reset never
    ```

- **Mutation Mechanism:**
  - Random mutations can be introduced into the instruction set at a configurable rate, simulating the natural variability seen in biological systems. Mutations introduce changes that may either hinder or enhance a sequence's ability to replicate.
  - Example command to set mutation rates:
    ```bash
    python abiogenesis.py --mutate_rate 0.0001
    ```

- **Seeding Hand-Coded Sequences:**
  - The simulation allows for the seeding of hand-coded replicators, providing the opportunity to test the stability and evolution of known replication patterns within the simulated environment.
  - Example command to seed hand-coded replicators:
    ```bash
    python abiogenesis.py --seed 500
    ```

- **Visual and Video Output:**
  - At specified intervals, the simulation generates visualizations showing the current state of the grid. A video can be created at the end of the simulation to observe how self-replicating sequences evolve over time.
  - Example command to create a video after the simulation:
    ```bash
    python abiogenesis.py --video --video_framerate 30 --video_resize 0.5
    ```

### Command-Line Arguments

- `--config`: Path to configuration file.
- `--height`: Number of instruction sequences in height dimension (default: 128).
- `--width`: Number of instruction sequences in width dimension (default: 256).
- `--sequence_length`: Length of each instruction sequence (default: 64).
- `--num_instructions`: Number of unique instructions (default: 64).
- `--num_heads`: Number of read-write heads for each instruction sequence (default: 2).
- `--head_range`: Maximum travel range in height, width dimensions for each head (default: 1).
- `--device`: Device to run the simulation on (`cpu` or `cuda`, default: `cuda`).
- `--num_sims`: Number of simulation iterations (default: 1,000,000).
- `--mutate_rate`: Mutation rate during simulation (default: 0.0).
- `--results_path`: Path to save results, including images and state files (default: `results/run_0`).
- `--image_save_interval`: Interval (in iterations) to save visualizations as images (default: 500).
- `--state_save_interval`: Interval (in iterations) to save the simulation state (default: 100,000).
- `--head_reset`: Set conditions for resetting the heads to their home positions upon termination and crash (default: 'always).
- `--load`: Resume run from checkpoint. Must enter a path to a valid checkpoint (default: '').
- `--loop_condition`: Condition loop on head reading 0 `value` or matching the read values at two heads `match` (default: `value`)
- `--loop_options`: Sets the number of different loop logic conditions allowed (default: 1)
- `--no_copy`: Removes copy operations from instruction set (default: `False`)
- `--seed`: Number of hand-coded replicators to seed into the simulation (default: 0)
- `--color_scheme`: Color scheme for visualizations: `default` `random` `dark` `pastel` `neon` (default: `default`)
- `--video`: Save a video of the simulation upon completion (default: `False`)
- `--video_framerate`: Sets the framerate for the video (default: 15)
- `--video_resize`: Resizes the video compared to the saved image sizes (default: 1.)




### Notes

- The simulation supports CUDA acceleration.
