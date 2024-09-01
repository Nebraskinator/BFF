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
    def __init__(self, height, width, tape_len, num_instructions=256, device='cpu', seed=False):
        assert tape_len**0.5 % 1 == 0
        self.device = device
        self.num_instructions = num_instructions
        self.tape = torch.randint(low=0, high=num_instructions, size=(height, width, tape_len), dtype=torch.int32, device=device)
        self.ip = torch.zeros((height, width), device=device).long()
        self.heads = torch.ones((height, width, 2, 3), dtype=torch.int64, device=device)
        self.heads[:, :, :, -1] = 0
        self.move_instructions = self.generate_move_instructions()
        if seed:
            replicator_template = self.generate_replicator_template()
            self.tape[0, 0] = replicator_template.clone()
            self.tape[-1, -1] = replicator_template.clone()
            self.tape[0, -1] = replicator_template.clone()
            self.tape[-1, 0] = replicator_template.clone()
            

    def generate_move_instructions(self):
        # Possible movements in each dimension: -1 (decrement), 0 (no change), 1 (increment)
        movement_values = [-1, 1]
        
        # Initialize an empty list to store movement instructions
        move_instructions = {}
        instruction_index = 1
    
        # Nested loops to create all combinations of movements in three dimensions
        for x in movement_values:  # Height movement
            move_instructions[instruction_index] = [x, 0, 0]
            instruction_index += 1
        for x in movement_values:  # Height movement
            move_instructions[instruction_index] = [0, x, 0]
            instruction_index += 1
        for x in movement_values:  # Height movement
            move_instructions[instruction_index] = [0, 0, x]
            instruction_index += 1
                        
        return move_instructions

    def generate_replicator_template(self):
        offset = len(self.move_instructions)
        replicator = torch.zeros(self.tape.shape[2]) + self.num_instructions - 3
        replicator[2] = offset + 3 # shift head1 in dim 1
        replicator[3] = offset * 2 + 11 # enter loop
        replicator[4] = offset * 2 + 5 # copy head0 to head1
        replicator[5] = offset # increment head0 dim 2
        replicator[6] = offset * 2 # increment head1 dim 2
        replicator[7] = offset * 2  + 12 # exit loop
        return replicator
        
        
    def update_head_positions(self, heads, move_mask, head_index, movement):
        if move_mask.any():
            # Extract a reduced version of heads using the move_mask
            masked_heads = heads[move_mask]
            
            # Update each coordinate in the specified head_index
            masked_heads[:, head_index, 0] = (masked_heads[:, head_index, 0] + movement[0]) % 3  # height: cyclic in [0, 2]
            masked_heads[:, head_index, 1] = (masked_heads[:, head_index, 1] + movement[1]) % 3  # width: cyclic in [0, 2]
            masked_heads[:, head_index, 2] = (masked_heads[:, head_index, 2] + movement[2]) % self.tape.shape[2]  # depth: full range
    
            # Reassign the modified masked_heads back to the correct positions
            heads[move_mask] = masked_heads

    def update_tape(self, tape, indices_mask, heads, operation, head_index=0):
        """
        Updates the values on the tape based on the operation ('inc', 'dec', etc.) at the specified head position.
    
        Args:
            tape (torch.Tensor): The 3D tape tensor where values are updated.
            indices_mask (torch.Tensor): A mask indicating which tape positions to update.
            heads (torch.Tensor): The positions of the heads in the tape.
            operation (str): The operation to perform ('inc', 'dec', etc.).
            head_index (int): The index of the head (0 for head0, 1 for head1).
        """
        if indices_mask.any():
            # Extract masked positions using the indices mask
            masked_heads = heads[indices_mask]
            
            # Calculate indices for height, width, and depth based on the masked heads
            tape_indices = (indices_mask.nonzero(as_tuple=True)[0] + masked_heads[:, head_index, 0] - 1) % tape.shape[0]
            width_indices = (indices_mask.nonzero(as_tuple=True)[1] + masked_heads[:, head_index, 1] - 1) % tape.shape[1]
            depth_indices = masked_heads[:, head_index, 2]  # depth
        
            # Perform the specified operation on the masked positions
            if operation == 'dec':
                tape[tape_indices, width_indices, depth_indices] -= 1
            elif operation == 'inc':
                tape[tape_indices, width_indices, depth_indices] += 1
    
            # Ensure values stay within the range [0, num_instructions)
            tape[tape_indices, width_indices, depth_indices] %= self.num_instructions

    def copy_tape_values(self, tape, indices_mask, heads, src_head_index, dest_head_index):
        if indices_mask.any():
            # Extract masked positions using the indices mask
            masked_heads = heads[indices_mask]
    
            # Calculate source and destination indices for height, width, and depth
            src_tape = (indices_mask.nonzero(as_tuple=True)[0] + masked_heads[:, src_head_index, 0] - 1) % tape.shape[0]  # height for source head
            dest_tape = (indices_mask.nonzero(as_tuple=True)[0] + masked_heads[:, dest_head_index, 0] - 1) % tape.shape[0]  # height for destination head
            src_width = (indices_mask.nonzero(as_tuple=True)[1] + masked_heads[:, src_head_index, 1] - 1) % tape.shape[1]   # width for source head
            dest_width = (indices_mask.nonzero(as_tuple=True)[1] + masked_heads[:, dest_head_index, 1] - 1) % tape.shape[1]  # width for destination head
            src_depth = masked_heads[:, src_head_index, 2]  # depth for source head
            dest_depth = masked_heads[:, dest_head_index, 2]  # depth for destination head
    
            # Copy values from the source head to the destination head
            tape[dest_tape, dest_width, dest_depth] = tape[src_tape, src_width, src_depth]

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
                                                    
    def handle_loops(self, match_value, indices_mask, loop_condition, search_direction):
        # Extract positions where the loop conditions are checked
        if indices_mask.any():
            # Extract positions
            masked_heads = self.heads[indices_mask]
            masked_ip = self.ip[indices_mask]
            
            # Calculate indices based on head positions
            nonzero_indices = indices_mask.nonzero(as_tuple=True)
            height_indices = (nonzero_indices[0] + masked_heads[:, 0, 0] - 1) % self.tape.shape[0]
            width_indices = (nonzero_indices[1] + masked_heads[:, 0, 1] - 1) % self.tape.shape[1]
            depth_indices = masked_heads[:, 0, 2]
    
            # Apply the loop condition to the extracted head positions
            loop_met = loop_condition(self.tape[height_indices, width_indices, depth_indices])
            valid_indices = loop_met.nonzero(as_tuple=True)[0]
    
            if valid_indices.numel() > 0:
                # Get search indices corresponding to the loop met condition
                search_indices = (nonzero_indices[0][valid_indices], nonzero_indices[1][valid_indices])
                valid_ips = masked_ip[valid_indices]
                tape_depth = self.tape.shape[2]
    
                # Create offset search matrix
                search_offsets = (valid_ips.unsqueeze(-1) + torch.arange(tape_depth, device=self.device) * search_direction) % tape_depth
                search_area = self.tape[search_indices[0], search_indices[1], :]
    
                # Find the match positions
                match_found = (search_area == match_value)
    
                # Calculate distances and penalize unmatched positions
                match_distances = torch.abs(search_offsets - valid_ips.unsqueeze(-1))
                match_distances[~match_found] = tape_depth + 1  # Penalize non-matches
    
                # Find the closest match
                closest_match_indices = match_distances.argmin(dim=-1)
    
                # Determine if the closest match is valid based on the search direction
                if search_direction == 1:  # Forward search
                    invalid_matches = match_distances[torch.arange(len(closest_match_indices)), closest_match_indices] >= (tape_depth - valid_ips)
                else:  # Backward search
                    invalid_matches = match_distances[torch.arange(len(closest_match_indices)), closest_match_indices] > valid_ips
    
                # Set invalid matches to -1
                new_ips = closest_match_indices.clone().type(torch.long)
                new_ips[invalid_matches] = -1
    
                # Update the instruction pointer (ip) with the closest matching index directly
                self.ip[search_indices[0], search_indices[1]] = new_ips
                '''
                # Debug: Verify the updated IP locations and corresponding instructions
                for h_idx, w_idx, d_idx in zip(search_indices[0], search_indices[1], new_ips):
                    if d_idx != -1:  # Check if the new IP is valid
                        instruction = self.tape[h_idx, w_idx, d_idx % tape_depth]
                        print(f"New IP Location: Direction {search_direction} Height {h_idx}, Width {w_idx}, Depth {d_idx % tape_depth} - Instruction: {instruction.item()}")
                    else:
                        print(f"No valid match found for Height {h_idx}, Width {w_idx}. IP set to -1.")
                '''
                
    def iterate(self):
        
        # Reset head positions to 0 where self.ip is zero
        reset_mask = self.ip == 0
        self.heads[reset_mask, :] = torch.tensor([1, 1, 0], device=self.device)
        
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
        
        # Use the offset to determine instruction values for head1
        dec_tape = instructions == offset * 2 + 3  # Decrement tape for head1
        inc_tape = instructions == offset * 2 + 4  # Increment tape for head1
        
        # Decrement tape values at positions specified by head1
        self.update_tape(self.tape, dec_tape, self.heads, operation='dec', head_index=1)
        
        # Increment tape values at positions specified by head1
        self.update_tape(self.tape, inc_tape, self.heads, operation='inc', head_index=1)
        
        # Assign instruction values for copy operations
        copy_0_to_1 = instructions == offset * 2 + 5  # Copy from head0 to head1
        copy_1_to_0 = instructions == offset * 2 + 6  # Copy from head1 to head0
        
        # Perform copy from head0 to head1
        self.copy_tape_values(self.tape, copy_0_to_1, self.heads, src_head_index=0, dest_head_index=1)
        
        # Perform copy from head1 to head0
        self.copy_tape_values(self.tape, copy_1_to_0, self.heads, src_head_index=1, dest_head_index=0)
        
        '''
        # Insert operations
        insert_0_to_1 = instructions == offset * 2 + 7  # Insert from head0 to head1
        insert_1_to_0 = instructions == offset * 2 + 8  # Insert from head1 to head0
    
        self.insert_tape_values(self.tape, insert_0_to_1, self.heads, src_head_index=0, dest_head_index=1)
        self.insert_tape_values(self.tape, insert_1_to_0, self.heads, src_head_index=1, dest_head_index=0)
                
        # Delete operations
        delete_0 = instructions == offset * 2 + 9  # Delete at head0
        delete_1 = instructions == offset * 2 + 10  # Delete at head1
    
        self.delete_tape_values(self.tape, delete_0, self.heads, head_index=0)
        self.delete_tape_values(self.tape, delete_1, self.heads, head_index=1)        
        '''
        
        forward_match_value = offset * 2 + 12
        reverse_match_value = offset * 2 + 11        
        enter_loop = instructions == offset * 2 + 11  # Loop entry
        exit_loop = instructions == offset * 2 + 12   # Loop exit
        
        # Generate forward and backward offset matrices for searching
        tape_depth = self.tape.shape[2]
        self.handle_loops(forward_match_value, enter_loop, lambda x: x == 0, 1)
        self.handle_loops(reverse_match_value, exit_loop, lambda x: x != 0, -1)
        
        # Update instruction pointer at the end
        self.ip = (self.ip + 1) % tape_depth
        
    def mutate(self, rate):
        # Generate a mask for mutation based on the rate
        mutation_mask = torch.rand(self.tape.shape, device=self.device) < rate
        # Count the number of mutations needed
        num_mutations = mutation_mask.sum()
        # Generate only the required number of random values
        random_values = torch.randint(low=0, high=self.num_instructions, size=(num_mutations.item(),), dtype=torch.int32, device=self.device)
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
        instruction_colors = torch.zeros((self.num_instructions, 3), dtype=torch.uint8)  # Default to black
        # Define blue for head0 movements
        head0_blue = torch.tensor([0, 0, 255], dtype=torch.uint8)
    
        # Define purple for head1 movements
        head1_purple = torch.tensor([128, 0, 128], dtype=torch.uint8)
    
        # Split the move instructions for head0 and head1 based on how they were generated
        num_move_instructions = len(self.move_instructions)
        head0_indices = torch.arange(1, 1 + num_move_instructions)  # Starting at 1 for head0
        head1_indices = torch.arange(1 + num_move_instructions, 1 + 2 * num_move_instructions)  # Continuing for head1
    
        # Assign blue color to all head0 move instructions
        instruction_colors[head0_indices] = head0_blue
    
        # Assign purple color to all head1 move instructions
        instruction_colors[head1_indices] = head1_purple
        # Define remaining instruction colors
        remaining_instruction_colors = {
            2 * num_move_instructions + 1: [255, 165, 0],    # Orange for decrement
            2 * num_move_instructions + 2: [255, 165, 0],    # Orange for increment
            2 * num_move_instructions + 3: [255, 165, 0],    # Orange for decrement
            2 * num_move_instructions + 4: [255, 165, 0],    # Orange for increment
            2 * num_move_instructions + 5: [255, 255, 0],    # Yellow for copy head0 to head0
            2 * num_move_instructions + 6: [255, 255, 0],    # Yellow for copy head0 to head1
            #2 * num_move_instructions + 7: [0, 255, 0],      # Green for insert
            2 * num_move_instructions + 11: [0, 255, 0],      # Green for insert
            #2 * num_move_instructions + 9: [255, 0, 0],      # Red for delete
            2 * num_move_instructions + 12: [255, 0, 0],     # Red for delete
            #2 * num_move_instructions + 11: [135, 206, 250], # Light Blue for loop entry
            #2 * num_move_instructions + 12: [186, 85, 211],  # Light Purple for loop exit
        }
    
        # Assign colors for the remaining instructions
        for instr, color in remaining_instruction_colors.items():
            instruction_colors[instr] = torch.tensor(color, dtype=torch.uint8)
    
        # Flatten the entire tape for processing
        flattened_tape = self.tape.flatten().cpu()
    
        # Create a color image array by directly indexing instruction_colors with flattened_tape
        image_array = instruction_colors[flattened_tape.long()]
    
        # For non-instruction values (from index 2 * num_move_instructions + 7 onward), map them to shades of gray
        non_instruction_mask = flattened_tape >= 2 * num_move_instructions + 13
        gray_values = flattened_tape[non_instruction_mask].to(torch.uint8)
        image_array[non_instruction_mask] = torch.stack([gray_values, gray_values, gray_values], dim=-1)
    
        # Reshape the image array to the desired dimensions
        height, width, tape_len = self.tape.shape
        grid_size = int(np.sqrt(tape_len))
        # Each depthwise slice (tape_len) should correspond to a grid_size x grid_size block
        image_array = image_array.view(height, width, grid_size, grid_size, 3)
        
        # Rearrange the blocks to form the correct 2D representation
        # height * grid_size creates rows and width * grid_size creates columns of the tape
        image_array = image_array.permute(0, 2, 1, 3, 4).contiguous().view(height * grid_size, width * grid_size, 3)    
        # Convert to numpy for saving with OpenCV
        image_array_np = image_array.numpy()
        image_array_np = image_array_np[..., ::-1]
        # Save the image to disk using OpenCV
        cv2.imwrite(path, image_array_np)
                
        
env = Abiogenesis(128, 256, 64, num_instructions=64, device='cuda')
for i in range(int(num_sims)):
    env.iterate()
    if not i % 144:
        env.mutate(0.0001)
    if not i % 10000:
        print(f"iteration {i}")
        env.save(rf'D:\BFF\results\{i}.p')
        env.visualize(rf'D:\BFF\results\{i}.png')
