# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:57:17 2024
"""

import os
import argparse
import torch
import cv2
import numpy as np

class Abiogenesis(object):
    def __init__(self, height, width, tape_len, num_instructions=64, device='cpu', reset_heads=True):
        assert tape_len**0.5 % 1 == 0
        self.reset_heads = reset_heads
        self.device = device
        self.num_instructions = num_instructions
        self.tape = torch.randint(low=0, high=num_instructions, size=(height, width, tape_len), dtype=torch.int32, device=device)
        self.ip = torch.zeros((height, width), device=device).long()
        self.heads = torch.ones((height, width, 2, 3), dtype=torch.int64, device=device)
        self.heads[:, :, :, -1] = 0
        self.generate_instruction_sets()
        self.iters = 0
            

    def generate_instruction_sets(self):   

        # Set instruction logic kwargs and precompute colors for visualization
          
        instruction_set = {}
        color_set = {0 : [0, 0, 0]}
        instruction_value = 1
        instruction_set[instruction_value] = (self.handle_forward_loops, {'enter_loop_instruction_value' : instruction_value, 
                                                          'exit_loop_instruction_value' : instruction_value+1,
                                                          'enter_loop_condition' : lambda x: x == 0,
                                                          'debug' : False})
        color_set[instruction_value] = [0, 255, 0]
        instruction_value += 1
        instruction_set[instruction_value] = (self.handle_backward_loops, {'exit_loop_instruction_value' : instruction_value, 
                                                          'enter_loop_instruction_value' : instruction_value-1,
                                                          'exit_loop_condition' : lambda x: x != 0,
                                                          'debug' : False})
        color_set[instruction_value] = [255, 0, 0]
        instruction_value += 1
        
        for head_index in [0, 1]:
            for dim in [0, 1, 2]:
                for direction in [-1, 1]:
                    instruction_set[instruction_value] = (self.update_head_positions, {'instruction_value' : instruction_value, 
                                                                      'head_index' : head_index,
                                                                      'dim' : dim,
                                                                      'direction' : direction})
                    color_set[instruction_value] = [0, 255, 255] if not head_index else [0, 0, 255]
                    instruction_value += 1
        
        for head_index in [0, 1]:
            for increment in [-1, 1]:
                instruction_set[instruction_value] = (self.update_tape, {'instruction_value' : instruction_value,
                                                                         'head_index' : head_index, 
                                                                         'increment' : increment})
                color_set[instruction_value] = [255, 153, 153] if increment == -1 else [153, 255, 153]
                instruction_value += 1
       
        instruction_set[instruction_value] = (self.copy_tape_values, {'instruction_value' : instruction_value,
                                                                 'src_head_index' : 0, 
                                                                 'dest_head_index' : 1})
        color_set[instruction_value] = [255, 255, 0]
        instruction_value += 1
        instruction_set[instruction_value] = (self.copy_tape_values, {'instruction_value' : instruction_value,
                                                                 'src_head_index' : 1, 
                                                                 'dest_head_index' : 0})
        color_set[instruction_value] = [255, 0, 255]
        instruction_value += 1
        
        self.instruction_set = instruction_set
        self.color_set = torch.linspace(0, 255, self.num_instructions, dtype=torch.uint8)
        self.color_set = self.color_set.unsqueeze(1).repeat(1, 3)
        for instr, color in color_set.items():
            self.color_set[instr] = torch.tensor(color, dtype=torch.uint8)
        
    def update_head_positions(self, instructions,  **kwargs):
        
        # Extract key variables
        instruction_value = kwargs.get('instruction_value')
        head_index = kwargs.get('head_index')
        dim = kwargs.get('dim')
        direction = kwargs.get('direction')
        
        # Indentify positions with the corresponding instruction
        mask = instructions == instruction_value
        
        if mask.any():
            
            # Extract relevant head positions
            masked_heads = self.heads[mask]

            # Update each coordinate in the specified head_index
            masked_heads[:, head_index, dim] += direction
            
            # Set limits on head coordinates, allowing cycling
            mod = self.tape.shape[2] if dim == 2 else 3 
            masked_heads[:, head_index, dim] %= mod

            # Overwrite the relevant self.head coordinates with the new values
            self.heads[mask] = masked_heads

    def update_tape(self, instructions, **kwargs):
        
        # Extract key variables
        instruction_value = kwargs.get('instruction_value')
        head_index = kwargs.get('head_index')
        increment = kwargs.get('increment')

        # Indentify positions with the corresponding instruction
        mask = instructions == instruction_value
        
        if mask.any():
            
            # Extract relevant head positions
            masked_heads = self.heads[mask]
            
            # Calculate indices for height, width, and depth based on the masked heads
            height_indices = (mask.nonzero(as_tuple=True)[0] + masked_heads[:, head_index, 0] - 1) % self.tape.shape[0]
            width_indices = (mask.nonzero(as_tuple=True)[1] + masked_heads[:, head_index, 1] - 1) % self.tape.shape[1]
            depth_indices = masked_heads[:, head_index, 2]  # depth
        
            self.tape[height_indices, width_indices, depth_indices] += increment
    
            # Ensure values stay within the range [0, num_instructions)
            self.tape[height_indices, width_indices, depth_indices] %= self.num_instructions

    def copy_tape_values(self, instructions, **kwargs):
        
        # Extract key variables
        instruction_value = kwargs.get('instruction_value')
        src_head_index = kwargs.get('src_head_index')
        dest_head_index = kwargs.get('dest_head_index')
        
        # Indentify positions with the corresponding instruction
        mask = instructions == instruction_value
        
        if mask.any():
            
            # Extract relevant head positions
            masked_heads = self.heads[mask]
    
            # Calculate source and destination indices for height, width, and depth
            src_height = (mask.nonzero(as_tuple=True)[0] + masked_heads[:, src_head_index, 0] - 1) % self.tape.shape[0]  # height for source head
            dest_height = (mask.nonzero(as_tuple=True)[0] + masked_heads[:, dest_head_index, 0] - 1) % self.tape.shape[0]  # height for destination head
            src_width = (mask.nonzero(as_tuple=True)[1] + masked_heads[:, src_head_index, 1] - 1) % self.tape.shape[1]   # width for source head
            dest_width = (mask.nonzero(as_tuple=True)[1] + masked_heads[:, dest_head_index, 1] - 1) % self.tape.shape[1]  # width for destination head
            src_depth = masked_heads[:, src_head_index, 2]  # depth for source head
            dest_depth = masked_heads[:, dest_head_index, 2]  # depth for destination head
    
            # Copy values from the source head to the destination head
            self.tape[dest_height, dest_width, dest_depth] = self.tape[src_height, src_width, src_depth]

    # not working!
    def insert_tape_values(self, tape, indices_mask, heads, src_head_index, dest_head_index):
        if indices_mask.any():
            # Cache `nonzero` results
            nonzero_indices = indices_mask.nonzero(as_tuple=True)
            masked_heads = heads[indices_mask]
    
            # Calculate indices for height, width, and depth
            src_tape = (nonzero_indices[0] + masked_heads[:, src_head_index, 0] - 1) % tape.shape[0]
            dest_tape = (nonzero_indices[0] + masked_heads[:, dest_head_index, 0] - 1) % tape.shape[0]
            src_width = (nonzero_indices[1] + masked_heads[:, src_head_index, 1] - 1) % tape.shape[1]
            dest_width = (nonzero_indices[1] + masked_heads[:, dest_head_index, 1] - 1) % tape.shape[1]
            src_depth = masked_heads[:, src_head_index, 2]
            dest_depth = masked_heads[:, dest_head_index, 2]
    
            # Calculate slices with broadcasting, avoid `unsqueeze` wherever possible
            max_shift = tape.shape[2] - dest_depth.max().item() - 1
            source_slice = torch.arange(max_shift, device=tape.device).view(1, -1) + dest_depth.view(-1, 1)
            dest_slice = source_slice + 1
    
            # Perform shift in place, minimizing tensor expansion
            tape.index_put_(
                (dest_tape.unsqueeze(-1), dest_width.unsqueeze(-1), dest_slice),
                tape[dest_tape.unsqueeze(-1), dest_width.unsqueeze(-1), source_slice],
                accumulate=False,
            )
            print(tape.shape)
            # Insert the value from the source head to the destination head position
            tape[dest_tape, dest_width, dest_depth] = tape[src_tape, src_width, src_depth]
    
    # not working!
    def delete_tape_values(self, tape, indices_mask, heads, head_index):
        if indices_mask.any():
            # Cache `nonzero` results
            nonzero_indices = indices_mask.nonzero(as_tuple=True)
            masked_heads = heads[indices_mask]
    
            # Calculate indices
            tape_indices = (nonzero_indices[0] + masked_heads[:, head_index, 0] - 1) % tape.shape[0]
            width_indices = (nonzero_indices[1] + masked_heads[:, head_index, 1] - 1) % tape.shape[1]
            depth_indices = masked_heads[:, head_index, 2]
    
            # Create valid depth slices, reducing expansion
            depth_range = torch.arange(tape.shape[2] - 1, device=tape.device)
            depth_shifted = depth_range + depth_indices.unsqueeze(-1)
    
            # Ensure bounds and perform in-place delete operation
            valid_depth_shifted = depth_shifted[depth_shifted < tape.shape[2] - 1]
            tape.index_put_(
                (tape_indices.unsqueeze(-1), width_indices.unsqueeze(-1), valid_depth_shifted),
                tape[tape_indices.unsqueeze(-1), width_indices.unsqueeze(-1), valid_depth_shifted + 1],
                accumulate=False,
            )
            
    def handle_forward_loops(self, instructions, **kwargs):
        
        # Extract key variables
        enter_loop_instruction_value = kwargs.get('enter_loop_instruction_value')
        exit_loop_instruction_value = kwargs.get('exit_loop_instruction_value')
        enter_loop_condition = kwargs.get('enter_loop_condition')
        debug = kwargs.get('debug')
        
        # Create mask for height, width positions with enter loop instruction
        enter_mask = instructions == enter_loop_instruction_value
        
        if not enter_mask.any():
            return  # Exit early if no enter instructions are found
    
        # Create index using the exit_mask
        enter_indices = enter_mask.nonzero(as_tuple=True)
        
        # Identify the relevant heads
        enter_heads = self.heads[enter_indices]
    
        # Calculate the height, width, depth positions of relevant head0's
        enter_height_indices = (enter_indices[0] + enter_heads[:, 0, 0] - 1) % self.tape.shape[0]
        enter_width_indices = (enter_indices[1] + enter_heads[:, 0, 1] - 1) % self.tape.shape[1]
        enter_depth_indices = enter_heads[:, 0, 2]
    
        # Check loop conditions at the head positions
        enter_condition_met = enter_loop_condition(self.tape[enter_height_indices, enter_width_indices, enter_depth_indices])
    
        # Filter the positions where enter conditions are met
        valid_enter_positions = enter_mask.clone()
        valid_enter_positions[enter_indices] = enter_condition_met
    
        if not valid_enter_positions.any():
            return  # Exit early if no valid enter conditions are met
    
        # Create index of positions to search
        search_indices = valid_enter_positions.nonzero(as_tuple=True)
        search_area = self.tape[search_indices[0], search_indices[1]]
    
        # Find enter and exit positions within the search area
        enter_positions = search_area == enter_loop_instruction_value
        exit_positions = search_area == exit_loop_instruction_value
    
        # Compute the cumulative sum to track loop nesting levels
        nesting_levels = torch.cumsum(enter_positions.int() - exit_positions.int(), dim=1)
    
        # Identify the min depths of valid search positions
        enter_depths = self.ip[search_indices]
        
        # Match valid search positions with correct nesting level conditions
        valid_matches = (torch.arange(self.tape.shape[2], device=self.device).unsqueeze(0) > enter_depths.unsqueeze(1)) & \
                        (nesting_levels == (nesting_levels[torch.arange(len(enter_depths)), enter_depths].unsqueeze(1) - 1))
        valid_matches = valid_matches * exit_positions
        
        # Get the index of the first exit match
        first_exit_indices = valid_matches.int().argmax(dim=1)
        
        # Check if the zero positions actually correspond to exit loop instructions
        invalid_forward_matches = first_exit_indices == 0
        first_exit_indices[invalid_forward_matches] = -1
    
        # Update IPs based on valid forward matches
        self.ip[search_indices[0], search_indices[1]] = first_exit_indices
        
        # Debug: Verify the updated IP locations and corresponding instructions
        if debug:
            for h_idx, w_idx, d_idx in zip(search_indices[0], search_indices[1], first_exit_indices):
                if d_idx != -1:  # Check if the new IP is valid
                    instruction = self.tape[h_idx, w_idx, d_idx]
                    print(f"New IP Location (Forward): Height {h_idx}, Width {w_idx}, Depth {d_idx} - Instruction: {instruction.item()}")
                else:
                    print(f"No valid forward match found for Height {h_idx}, Width {w_idx}. IP set to -1.")
    
    def handle_backward_loops(self, instructions, **kwargs):
        
        # Initialize key variables
        enter_loop_instruction_value = kwargs.get('enter_loop_instruction_value')
        exit_loop_instruction_value = kwargs.get('exit_loop_instruction_value')
        exit_loop_condition = kwargs.get('exit_loop_condition')
        debug = kwargs.get('debug')
        
        # Create mask for height, width positions with exit loop instructions
        exit_mask = instructions == exit_loop_instruction_value
    
        if not exit_mask.any():
            return  # Exit early if no exit instructions are found
    
        # Create index using the exit_mask
        exit_indices = exit_mask.nonzero(as_tuple=True)
        
        # Identify the relevant heads
        exit_heads = self.heads[exit_indices]
    
        # Calculate the height, width, depth positions of relevant head0's
        exit_height_indices = (exit_indices[0] + exit_heads[:, 0, 0] - 1) % self.tape.shape[0]
        exit_width_indices = (exit_indices[1] + exit_heads[:, 0, 1] - 1) % self.tape.shape[1]
        exit_depth_indices = exit_heads[:, 0, 2]
    
        # Check loop conditions at the head positions
        exit_condition_met = exit_loop_condition(self.tape[exit_height_indices, exit_width_indices, exit_depth_indices])
    
        # Filter the positions where exit conditions are met
        valid_exit_positions = exit_mask.clone()
        valid_exit_positions[exit_indices] = exit_condition_met
    
        if not valid_exit_positions.any():
            return  # Exit early if no valid exit conditions are met
    
        # Create index of positions to search
        search_indices = valid_exit_positions.nonzero(as_tuple=True)
        search_area = self.tape[search_indices[0], search_indices[1]]
    
        # Flip depthwise to convert backward search to forward search
        flipped_search_area = search_area.flip(dims=[1])
    
        # Find enter and exit positions within the flipped search area
        enter_positions = flipped_search_area == enter_loop_instruction_value
        exit_positions = flipped_search_area == exit_loop_instruction_value
    
        # Compute the cumulative sum to track loop nesting levels in the flipped search area
        nesting_levels = torch.cumsum(exit_positions.int() - enter_positions.int(), dim=1)
    
        # Identify the min depths of valid search positions and adjust for flipped indices
        exit_depths = self.tape.shape[2] - self.ip[search_indices] - 1
    
        # Match valid search positions with correct nesting level conditions
        valid_matches = (torch.arange(self.tape.shape[2], device=self.device).unsqueeze(0) > exit_depths.unsqueeze(1)) & \
                        (nesting_levels == (nesting_levels[torch.arange(len(exit_depths)), exit_depths].unsqueeze(1) - 1))
    
        # Ensure matches point only to enter loop positions
        valid_matches = valid_matches * enter_positions
    
        # Get the first valid enter match for the search
        first_enter_indices = valid_matches.int().argmax(dim=1)
        
        # Identify positions where first_enter_indices points to zero
        zero_indices = first_enter_indices == 0
        
        # Check if these zero positions correspond to the enter loop instructions
        zero_valid = enter_positions[:, -1]
        
        # Convert the matched indices back to the unflipped depth dimension
        first_enter_indices = self.tape.shape[2] - first_enter_indices - 1
        
        # Update the invalid matches: set to -1 where the zero index does not correspond to a valid enter loop instruction
        first_enter_indices[zero_indices & ~zero_valid] = -1
    
        # Update IPs based on valid backward matches
        self.ip[search_indices[0], search_indices[1]] = first_enter_indices
    
        # Debug: Verify the updated IP locations and corresponding instructions
        if debug:
            for h_idx, w_idx, d_idx in zip(search_indices[0], search_indices[1], first_enter_indices):
                if d_idx != -1:  # Check if the new IP is valid
                    instruction = self.tape[h_idx, w_idx, d_idx]
                    print(f"New IP Location (Backward): Height {h_idx}, Width {w_idx}, Depth {d_idx} - Instruction: {instruction.item()}")
                else:
                    print(f"No valid backward match found for Height {h_idx}, Width {w_idx}. IP set to -1.")
                                                                    
    def iterate(self):
                
        # get the current instruction for each instruction sequence using self.ip (instruction position)
        instructions = self.tape[torch.arange(self.tape.shape[0], device=self.device).unsqueeze(1),
                                 torch.arange(self.tape.shape[1], device=self.device),
                                 self.ip]
        
        # iterate through each type of instruction, executing all sequences in parallel
        for instruction_value, (func, kwargs) in self.instruction_set.items():
            func(instructions, **kwargs)
        
        if self.reset_heads:
            # Reset head positions to 0 where the next instruction is position 0
            reset_mask = (self.ip == self.tape.shape[2] - 1) | (self.ip == -1)
        else:
            # Reset head positions only if the instruction sequence has reached the terminus
            reset_mask = (self.ip == self.tape.shape[2] - 1)
        self.heads[reset_mask, :] = torch.tensor([1, 1, 0], device=self.device)
        
        # Increment the instruction positions
        self.ip = (self.ip + 1) % self.tape.shape[2]
        
        # Increment iteration counter
        self.iters += 1
        
    def mutate(self, rate):
        
        # Generate a mask for mutation based on the rate
        mutation_mask = torch.rand(self.tape.shape, device=self.device) < rate
        
        # Count the number of mutations needed
        num_mutations = mutation_mask.sum()
        
        # Generate the required number of random values
        random_values = torch.randint(low=0, high=self.num_instructions, size=(num_mutations.item(),), dtype=torch.int32, device=self.device)
        
        # Apply the random values to the tape at positions where the mutation mask is True
        self.tape[mutation_mask] = random_values
            
    def save(self, path):
        # Save the state of the Abiogenesis class to disk
        state = {
            'tape': self.tape,
            'ip': self.ip,
            'heads': self.heads,
            'device': self.device,
            'num_instructions' : self.num_instructions,
            'reset_heads' : self.reset_heads,
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, path, device='cpu'):
        
        # Load the state of the Abiogenesis class from disk
        state = torch.load(path)
        
        # Extract the dimensions from the loaded state
        height, width, tape_len = state['tape'].shape
        
        # Create a new instance of the class with the loaded state
        obj = cls(height=height, width=width, tape_len=tape_len, device=device)
        
        # Restore the state to the desired device
        obj.tape = state['tape'].to(device)
        obj.ip = state['ip'].to(device)
        obj.heads = state['heads'].to(device)
        obj.num_instructions = state.get('num_instructions', 256)
        obj.reset_heads = state.get('reset_heads', True)
        obj.iters = state.get('iters', 0)
        
        return obj
        
    def visualize(self, path):
    
        height, width, tape_len = self.tape.shape
        
        # Flatten the entire tape to index the instruction color set
        flattened_tape = self.tape.flatten().cpu()
    
        # Create the color image array by directly indexing instruction_colors
        image_array = self.color_set[flattened_tape.long()]
        
        # Calculate the dimensions of a square instruction sequence
        grid_size = int(np.sqrt(tape_len))
        
        # Reshape image array to original height and width, but set the depth dimension
        # to square grids with 3 color channels
        image_array = image_array.view(height, width, grid_size, grid_size, 3)  
        
        # Rearrange the blocks into rows and columns
        # height * grid_size creates rows and width * grid_size creates columns of the tape
        image_array = image_array.permute(0, 2, 1, 3, 4).contiguous().view(height * grid_size, width * grid_size, 3)
        
        # Convert to numpy for saving with OpenCV
        image_array_np = image_array.numpy()
        
        # Convert BGR to RGB
        image_array_np = image_array_np[..., ::-1]
        
        # Save the image to disk using OpenCV
        cv2.imwrite(path, image_array_np)
        
    def to(self, device):
        assert str(device).isin(['cpu', 'cuda'])
        self.device = device
        self.tape = self.tape.to(device)
        self.ip = self.ip.to(device)
        self.heads = self.heads.to(device)
        
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Abiogenesis simulation with customizable parameters.")
    parser.add_argument('--height', type=int, default=128, help='Height of the tape (default: 128)')
    parser.add_argument('--width', type=int, default=256, help='Width of the tape (default: 256)')
    parser.add_argument('--depth', type=int, default=64, help='Depth of the tape (default: 64)')
    parser.add_argument('--num_instructions', type=int, default=64, help='Number of instructions (default: 64)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the simulation on (default: cuda)')
    parser.add_argument('--num_sims', type=int, default=int(1e7), help='Number of simulation iterations (default: 1e7)')
    parser.add_argument('--mutate_rate', type=float, default=0.0, help='Mutation rate (default: 0.0)')
    parser.add_argument('--results_path', type=str, default='results/run_0', help='Path to save results (default: results/run_0)')
    parser.add_argument('--image_save_interval', type=int, default=500, help='Interval to save images (default: 500 iterations)')
    parser.add_argument('--state_save_interval', type=int, default=100000, help='Interval to save states (default: 100000 iterations)')
    parser.add_argument('--stateful_heads', type=bool, default=False, help='Determines if heads maintain their state or reset upon invalid instruction')
    parser.add_argument('--load', type=str, default='', help='Path to saved simulation')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Create the results path if it does not exist
    os.makedirs(args.results_path, exist_ok=True)
    img_path = os.path.join(args.results_path, 'img')
    states_path = os.path.join(args.results_path, 'states')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(states_path, exist_ok=True)

    # Initialize the environment
    if args.load:
        if os.path.exists(args.load):
            env = Abiogenesis.load(args.load, device=args.device)
        else:
            print("Load path does not exist.")
            return
    else:
        env = Abiogenesis(args.height, args.width, args.depth, num_instructions=args.num_instructions, device=args.device, reset_heads=not args.stateful_heads)
        
    # Run the simulation
    start_iter = env.iters
    for i in range(start_iter, args.num_sims+1):
        env.iterate()
        if args.mutate_rate and not i % args.depth:
             env.mutate(args.mutate_rate)
        if not i % args.image_save_interval:
            print(f"Iteration {i}")
            env.visualize(os.path.join(img_path, f'{i:09}.png'))
        if not i % args.state_save_interval:
            env.save(os.path.join(states_path, f'{i:09}.p'))               
        
if __name__ == "__main__":
    main()