# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:57:17 2024

@author: ruhe
"""

import torch
import cv2
import numpy as np

num_sims = 1e7

class Abiogenesis(object):
    def __init__(self, height, width, tape_len, device='cpu'):
        assert tape_len < 256
        assert 256 % tape_len == 0
        self.device = device
        self.tape = torch.randint(low=0, high=255, size=(height, width, tape_len), dtype=torch.uint8, device=device)
        self.ip = torch.zeros((height, width), device=device).long()
        self.heads = torch.zeros((height, width, 2, 3), dtype=torch.int64, device=device)
        self.move_instructions = self.generate_move_instructions()

    def generate_move_instructions(self):
        # Possible movements in each dimension: -1 (decrement), 0 (no change), 1 (increment)
        movement_values = [-1, 0, 1]
        
        # Initialize an empty list to store movement instructions
        move_instructions = {}
        instruction_index = 1
    
        # Nested loops to create all combinations of movements in three dimensions
        for x in movement_values:  # Height movement
            for y in movement_values:  # Width movement
                for z in movement_values:  # Depth movement
                    # Exclude the (0, 0, 0) combination which represents no movement
                    if (x, y, z) != (0, 0, 0):
                        move_instructions[instruction_index] = [x, y, z]
                        instruction_index += 1
    
        return move_instructions

    def update_head_positions(self, heads, move_mask, head_index, movement):
        if move_mask.any():
            # Extract a reduced version of heads using the move_mask
            masked_heads = heads[move_mask]
            
            # Update each coordinate in the specified head_index
            masked_heads[:, head_index, 0] = (masked_heads[:, head_index, 0] + movement[0]) % 3  # height: cyclic in [0, 2]
            masked_heads[:, head_index, 1] = (masked_heads[:, head_index, 1] + movement[1]) % 3  # width: cyclic in [0, 2]
            masked_heads[:, head_index, 2] = (masked_heads[:, head_index, 2] + movement[2]) % heads.shape[3]  # depth: full range
    
            # Reassign the modified masked_heads back to the correct positions
            heads[move_mask] = masked_heads

    def update_tape(self, tape, indices_mask, heads, operation):
        if indices_mask.any():
            # Extract masked positions using the indices mask
            masked_heads = heads[indices_mask]
            
            # Calculate the indices for height, width, and depth based on the masked heads
            tape_indices = torch.clamp((indices_mask.nonzero(as_tuple=True)[0] + masked_heads[:, 0, 0] - 1), 0, tape.shape[0] - 1)
            width_indices = torch.clamp((indices_mask.nonzero(as_tuple=True)[1] + masked_heads[:, 0, 1] - 1), 0, tape.shape[1] - 1)
            depth_indices = masked_heads[:, 0, 2]  # depth
    
            # Perform the specified operation on the masked positions
            if operation == 'dec':
                tape[tape_indices, width_indices, depth_indices] -= 1
            elif operation == 'inc':
                tape[tape_indices, width_indices, depth_indices] += 1

    def copy_tape_values(self, tape, indices_mask, heads, src_head_index, dest_head_index):
        if indices_mask.any():
            # Extract masked positions using the indices mask
            masked_heads = heads[indices_mask]
    
            # Calculate source and destination indices for height, width, and depth
            src_tape = torch.clamp((indices_mask.nonzero(as_tuple=True)[0] + masked_heads[:, src_head_index, 0] - 1), 0, tape.shape[0] - 1)  # height for source head
            dest_tape = torch.clamp((indices_mask.nonzero(as_tuple=True)[0] + masked_heads[:, dest_head_index, 0] - 1), 0, tape.shape[0] - 1)  # height for destination head
            src_width = torch.clamp((indices_mask.nonzero(as_tuple=True)[1] + masked_heads[:, src_head_index, 1] - 1), 0, tape.shape[1] - 1)   # width for source head
            dest_width = torch.clamp((indices_mask.nonzero(as_tuple=True)[1] + masked_heads[:, dest_head_index, 1] - 1), 0, tape.shape[1] - 1)  # width for destination head
            src_depth = masked_heads[:, src_head_index, 2]  # depth for source head
            dest_depth = masked_heads[:, dest_head_index, 2]  # depth for destination head
    
            # Copy values from the source head to the destination head
            tape[dest_tape, dest_width, dest_depth] = tape[src_tape, src_width, src_depth]

    def handle_loops(self, tape, indices_mask, heads, ip, loop_mask, offset_matrix, match_value):
        if indices_mask.any():
            # Extract masked positions using the indices mask
            masked_heads = heads[indices_mask]
            masked_ip = ip[indices_mask]
    
            # Calculate height, width, and depth indices
            height_indices = torch.clamp(
                indices_mask.nonzero(as_tuple=True)[0] + masked_heads[:, 0, 0] - 1, 0, tape.shape[0] - 1
            )
            width_indices = torch.clamp(
                indices_mask.nonzero(as_tuple=True)[1] + masked_heads[:, 0, 1] - 1, 0, tape.shape[1] - 1
            )
            depth_indices = masked_heads[:, 0, 2]
    
            # Check loop condition based on the loop mask
            if loop_mask.any():
                # Perform a vectorized search for matching values
                search_offsets = offset_matrix[indices_mask]
                search_area = tape[height_indices, width_indices].unsqueeze(1)  # Add dimension for broadcasting
                match_found = search_area.gather(2, search_offsets.unsqueeze(1)) == match_value
                match_indices = match_found.int().argmax(dim=2).squeeze(1).type(torch.uint8)  # Index of the first occurrence
                found_mask = match_found.any(dim=2).squeeze(1)  # Check if any match found
    
                # Update instruction pointers based on search results
                if (loop_mask & found_mask).any():
                    ip[indices_mask[loop_mask & found_mask]] = (
                        masked_ip[loop_mask & found_mask] + match_indices[loop_mask & found_mask] - 1
                    )
                if (loop_mask & ~found_mask).any():
                    ip[indices_mask[loop_mask & ~found_mask]] = -1  # Reset if no match found


    def iterate(self):
        
        # Reset head positions to 0 where self.ip is zero
        reset_mask = self.ip == 0
        self.heads[reset_mask] = 0
        
        instructions = self.tape[torch.arange(self.tape.shape[0], device=self.device).unsqueeze(1),
                                 torch.arange(self.tape.shape[1], device=self.device),
                                 self.ip]
        
        # Update head0 positions based on the dynamically generated movement instructions
        for idx, movement in self.move_instructions.items():
            move_mask = instructions == idx
            self.update_head_positions(self.heads, move_mask, head_index=0, movement=movement)
        
        # Update head1 positions with an offset to ensure unique instructions for head1
        offset = len(self.move_instructions)  # Offset to differentiate instructions for head1
        
        for idx, movement in self.move_instructions.items():
            move_mask = instructions == (idx + offset)  # Offset the index for head1
            self.update_head_positions(self.heads, move_mask, head_index=1, movement=movement)
                
                
        # Use the offset to determine instruction values for head0
        dec_tape = instructions == offset * 2 + 1  # Decrement tape for head0
        inc_tape = instructions == offset * 2 + 2  # Increment tape for head0
        
        # Decrement tape values at positions specified by head0
        self.update_tape(self.tape, dec_tape, self.heads, operation='dec')
        
        # Increment tape values at positions specified by head0
        self.update_tape(self.tape, inc_tape, self.heads, operation='inc')      
        
        # Assign instruction values for copy operations
        copy_0_to_1 = instructions == offset * 2 + 3  # Copy from head0 to head1
        copy_1_to_0 = instructions == offset * 2 + 4  # Copy from head1 to head0
        
        # Perform copy from head0 to head1
        self.copy_tape_values(self.tape, copy_0_to_1, self.heads, src_head_index=0, dest_head_index=1)
        
        # Perform copy from head1 to head0
        self.copy_tape_values(self.tape, copy_1_to_0, self.heads, src_head_index=1, dest_head_index=0)
        
        
        forward_match_value = offset * 2 + 5
        reverse_match_value = offset * 2 + 6        
        enter_loop = instructions == offset * 2 + 5  # Loop entry
        exit_loop = instructions == offset * 2 + 6   # Loop exit
        
        # Generate forward and backward offset matrices for searching
        tape_height = self.tape.shape[0]
        tape_width = self.tape.shape[1]
        tape_depth = self.tape.shape[2]
        forward_offsets = (self.ip.unsqueeze(-1) + torch.arange(1, tape_depth, device=self.device).view(1, 1, -1)) % tape_depth
        backward_offsets = (self.ip.unsqueeze(-1) - torch.arange(1, tape_depth, device=self.device).view(1, 1, -1)) % tape_depth
        masked_heads = self.heads[enter_loop]
        self.handle_loops(
            tape=self.tape,
            indices_mask=enter_loop,
            heads=self.heads,
            ip=self.ip,
            loop_mask=(self.tape[
                masked_heads[:, 0, 0].clamp(0, tape_height - 1),  # height indices
                masked_heads[:, 0, 1].clamp(0, tape_width - 1),   # width indices
                masked_heads[:, 0, 2]                             # depth indices
            ] == 0),  # Mask for loop entry based on head value
            offset_matrix=forward_offsets,
            match_value=forward_match_value  # Matching value for the exit loop `]`
        )
        masked_heads = self.heads[exit_loop]
        self.handle_loops(
            tape=self.tape,
            indices_mask=exit_loop,
            heads=self.heads,
            ip=self.ip,
            loop_mask=(self.tape[
                masked_heads[:, 0, 0].clamp(0, tape_height - 1),  # height indices
                masked_heads[:, 0, 1].clamp(0, tape_width - 1),   # width indices
                masked_heads[:, 0, 2]                             # depth indices
            ] == 0),  # Mask for loop entry based on head value
            offset_matrix=backward_offsets,
            match_value=reverse_match_value  # Matching value for the exit loop `]`
        )
        
        # Update instruction pointer at the end
        self.ip = (self.ip + 1) % tape_depth
        
    def mutate(self, rate):
        # Generate a mask for mutation based on the rate
        mutation_mask = torch.rand(self.tape.shape, device=self.device) < rate
        # Count the number of mutations needed
        num_mutations = mutation_mask.sum()
        # Generate only the required number of random values
        random_values = torch.randint(low=0, high=255, size=(num_mutations.item(),), dtype=torch.uint8, device=self.device)
        # Apply the random values to the tape at positions where the mutation mask is True
        self.tape[mutation_mask] = random_values
            
    def save(self, path):
        # Save the state of the Abiogenesis class to disk
        state = {
            'tape': self.tape,
            'ip': self.ip,
            'heads': self.heads,
            'device': self.device
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, path):
        # Load the state of the Abiogenesis class from disk
        state = torch.load(path)
        
        # Extract the dimensions from the loaded state
        height, width, tape_len = state['tape'].shape
        
        # Create a new instance of the class with the loaded state
        obj = cls(height=height, width=width, tape_len=tape_len, device=state['device'])
        
        # Restore the state
        obj.tape = state['tape']
        obj.ip = state['ip']
        obj.heads = state['heads']
        obj.move_instructions = obj.generate_move_instructions()  # Regenerate the move instructions
        
        return obj
        
    def visualize(self, path):
        # Initialize a full color map with 256 colors, starting with shades of gray
        instruction_colors = torch.zeros((256, 3), dtype=torch.uint8)  # Default to black
    
        # Define shades of blue for head0 movements
        head0_blue_shades = torch.tensor([
            [0, 0, 255],    # Blue
            [0, 64, 255],   # Medium Blue
            [0, 128, 255],  # Light Blue
            [0, 192, 255],  # Lighter Blue
        ], dtype=torch.uint8)
    
        # Define shades of purple for head1 movements
        head1_purple_shades = torch.tensor([
            [128, 0, 128],  # Purple
            [153, 50, 204], # Medium Purple
            [75, 0, 130],   # Dark Purple
            [148, 0, 211],  # Light Purple
        ], dtype=torch.uint8)
    
        # Split the move instructions for head0 and head1 based on how they were generated
        num_move_instructions = len(self.move_instructions)
        head0_indices = torch.arange(1, 1 + num_move_instructions)  # Starting at 1 for head0
        head1_indices = torch.arange(1 + num_move_instructions, 1 + 2 * num_move_instructions)  # Continuing for head1
    
        # Assign colors to the move instructions for head0 and head1
        instruction_colors[head0_indices] = head0_blue_shades.repeat((num_move_instructions // len(head0_blue_shades)) + 1, 1)[:num_move_instructions]
        instruction_colors[head1_indices] = head1_purple_shades.repeat((num_move_instructions // len(head1_purple_shades)) + 1, 1)[:num_move_instructions]
    
        # Define remaining instruction colors
        remaining_instruction_colors = {
            2 * num_move_instructions + 1: [255, 0, 0],      # Red for decrement
            2 * num_move_instructions + 2: [0, 255, 0],      # Green for increment
            2 * num_move_instructions + 3: [255, 165, 0],    # Orange for copy head0 to head1
            2 * num_move_instructions + 4: [255, 140, 0],    # Dark Orange for copy head1 to head0
            2 * num_move_instructions + 5: [255, 255, 0],    # Yellow for loop entry
            2 * num_move_instructions + 6: [255, 255, 102],  # Light Yellow for loop exit
        }
    
        # Assign colors for the remaining instructions
        for instr, color in remaining_instruction_colors.items():
            instruction_colors[instr] = torch.tensor(color, dtype=torch.uint8)
    
        # Flatten the entire tape for processing
        flattened_tape = self.tape.flatten().cpu()
    
        # Create a color image array by directly indexing instruction_colors with flattened_tape
        image_array = instruction_colors[flattened_tape.long()]
    
        # For non-instruction values (from index 2 * num_move_instructions + 7 onward), map them to shades of gray
        non_instruction_mask = flattened_tape >= 2 * num_move_instructions + 7
        gray_values = flattened_tape[non_instruction_mask]
        image_array[non_instruction_mask] = torch.stack([gray_values, gray_values, gray_values], dim=-1)
    
        # Reshape the image array to the desired dimensions
        height, width, tape_len = self.tape.shape
        grid_size = int(np.sqrt(tape_len))
        image_array = image_array.view(height * grid_size, width * grid_size, 3)
    
        # Convert to numpy for saving with OpenCV
        image_array_np = image_array.numpy()
    
        # Save the image to disk using OpenCV
        cv2.imwrite(path, image_array_np)
                
        
env = Abiogenesis(128, 256, 64, device='cuda')
for i in range(int(num_sims)):
    env.iterate()
    if not i % 64:
        env.mutate(0.0001)
    if not i % 10000:
        print(f"iteration {i}")
        env.save(rf'D:\BFF\results\{i}.p')
        env.visualize(rf'D:\BFF\results\{i}.png')
