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
    def __init__(self, height, 
                 width, 
                 tape_len, 
                 num_instructions=64, 
                 device='cpu', 
                 reset_heads=True,
                 loop_condition='value',
                 loop_option=False,
                 no_copy=False,
                 seed=0,
                 color_scheme='default'):
        assert tape_len**0.5 % 1 == 0
        assert loop_condition in ['value', 'match', 'both']
        self.reset_heads = reset_heads
        self.loop_condition = loop_condition
        self.loop_option = loop_option
        self.no_copy = no_copy
        self.device = device
        self.seed = seed
        self.num_instructions = num_instructions
        self.tape = torch.randint(low=0, high=num_instructions, size=(height, width, tape_len), dtype=torch.int32, device=device)
        self.ip = torch.zeros((height, width), device=device).long()
        self.heads = torch.ones((height, width, 2, 3), dtype=torch.int64, device=device)
        self.heads[:, :, :, -1] = 0
        
        self.generate_instruction_sets()
        
        self.color_scheme = color_scheme
        self.default_color_scheme()
        if seed:
            self.seed_sim(seed)
        
        self.iters = 0
            
    
    def seed_sim(self, num_seeds):
        
        move_head1_xy = []
        enter_loop = []
        crement = []
        exit_loop = []
        copy = False
        move_head0_z = False
        move_head1_z = False
        enter_match = False
        exit_match = False
        
        for instruction, info in self.instruction_set.items():
            iv = info[1]['instruction_value']
            if 'head1_dim0' in instruction:
                move_head1_xy.append(iv)
            if 'head1_dim1' in instruction:
                move_head1_xy.append(iv)
            if 'skip' in instruction and '==' in instruction:
                enter_loop.append(iv)
                if 'head1' in instruction:
                    enter_match = iv
            if 'head0_dim2_move1' in instruction:
                move_head0_z = iv
            if 'head1_dim2_move1' in instruction:
                move_head1_z = iv
            if 'copy_head0' in instruction:
                copy = iv
            if 'repeat' in instruction and '!=' in instruction:
                exit_loop.append(iv)
                if 'head1' in instruction:
                    exit_match = iv
            if 'value' in instruction:
                crement.append(iv)
                    
        if self.no_copy and self.loop_condition in ['match', 'both']:
            replicators = np.zeros((num_seeds, 8)).astype('uint8')
            replicators[:, 0] = np.random.choice(move_head1_xy, num_seeds)
            replicators[:, 1] = np.random.choice(enter_loop, num_seeds)
            replicators[:, 2] = enter_match
            replicators[:, 3] = np.random.choice(crement, num_seeds)
            replicators[:, 4] = exit_match
            replicators[:, 5] = move_head1_z
            replicators[:, 6] = move_head0_z
            replicators[:, 7] = np.random.choice(exit_loop, num_seeds)
            
        elif not self.no_copy:
            replicators = np.zeros((num_seeds, 6)).astype('uint8')
            replicators[:, 0] = np.random.choice(move_head1_xy, num_seeds)
            replicators[:, 1] = np.random.choice(enter_loop, num_seeds)
            replicators[:, 2] = copy
            replicators[:, 3] = move_head1_z
            replicators[:, 4] = move_head0_z
            replicators[:, 5] = np.random.choice(exit_loop, num_seeds)
        else:
            print("Unable to create replicator when --no_copy is true and --loop_condition is not 'match' or 'both'")
            return
        
        replicators = torch.tensor(replicators, device=self.device)
        size = replicators.size(1)
        
        xs = torch.randint(low=0, high=self.tape.size(0), size=(num_seeds,), 
                               device=self.device)
        ys = torch.randint(low=0, high=self.tape.size(1), size=(num_seeds,), device=self.device)
        zs = torch.randint(low=0, high=self.tape.size(2)-size, size=(num_seeds,), device=self.device)
        print('seeding simulation...')
        cnt = 0
        for x, y, z in zip(xs, ys, zs):
            self.tape[x, y, z:z+size] = replicators[cnt]
            cnt += 1
        
        
    def default_color_scheme(self):
        
        if self.color_scheme == 'default':
            self.color_set = torch.linspace(0, 255, self.num_instructions, dtype=torch.uint8)
            self.color_set = self.color_set.unsqueeze(1).repeat(1, 3)
            for instruction, info in self.instruction_set.items():
                if 'skip' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([0, 255, 0], dtype=torch.uint8)
                elif 'repeat' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([255, 0, 0], dtype=torch.uint8)
                elif 'move' in instruction:
                    if 'head0' in instruction:
                        self.color_set[info[1]['instruction_value']] = torch.tensor([0, 255, 255], dtype=torch.uint8)
                    elif 'head1' in instruction:
                        self.color_set[info[1]['instruction_value']] = torch.tensor([0, 0, 255], dtype=torch.uint8)
                elif 'value-' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([255, 153, 153], dtype=torch.uint8)
                elif 'value' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([153, 255, 153], dtype=torch.uint8)
                elif 'copy_head0' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([255, 255, 0], dtype=torch.uint8)
                elif 'copy_head1' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([255, 0, 255], dtype=torch.uint8)
                    
        elif self.color_scheme == 'random':
            colors = np.array([[0,0,0],
                    [0,0,255],
                    [0,255,0],
                    [0,255,255],
                    [255,0,0],
                    [255,0,255],
                    [255,255,0],
                    [255,255,255],
                    [255, 153, 153],
                    [153, 255, 153],
                    [50,50,50],
                    [100,100,100],
                    [150,150,150],
                    [200,200,200],
                    ])
            
            colors = colors[np.random.choice(np.arange(len(colors)),
                                      size=self.num_instructions)]
            self.color_set = torch.tensor(colors, dtype=torch.uint8)
            
        elif self.color_scheme == 'dark':
            self.color_set = torch.linspace(50, 0, self.num_instructions, dtype=torch.uint8)
            self.color_set = self.color_set.unsqueeze(1).repeat(1, 3)
            self.color_set[0] = torch.tensor([255, 255, 255], dtype=torch.uint8)
            for instruction, info in self.instruction_set.items():
                if 'skip' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([0, 255, 0], dtype=torch.uint8)
                elif 'repeat' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([255, 69, 0], dtype=torch.uint8)
                elif 'move' in instruction:
                    if 'head0' in instruction:
                        self.color_set[info[1]['instruction_value']] = torch.tensor([0, 128, 128], dtype=torch.uint8)
                    elif 'head1' in instruction:
                        self.color_set[info[1]['instruction_value']] = torch.tensor([48, 25, 52], dtype=torch.uint8)
                elif 'value-' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([255, 105, 180], dtype=torch.uint8)
                elif 'value' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([144, 238, 144], dtype=torch.uint8)
                elif 'copy_head0' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([0, 255, 255], dtype=torch.uint8)
                elif 'copy_head1' in instruction:
                    self.color_set[info[1]['instruction_value']] = torch.tensor([138, 43, 226], dtype=torch.uint8)
        
        elif isinstance(self.color_scheme, dict):
            for key, color in self.color_scheme:
                if isinstance(key, str) and isinstance(color, list) and len(color) == 3 and all([isinstance(i, int) for i in color]):
                    for instruction, info in self.instruction_set.items():
                        if key in instruction:
                            self.color_set[info[1]['instruction_value']] = torch.tensor(color, dtype=torch.uint8) 
        
        
        

    def generate_instruction_sets(self):   

        # Set instruction logic kwargs and precompute colors for visualization
          
        instruction_set = {}
        
        instruction_value = 1
        num_loops = 1 + (self.loop_option or (self.loop_condition == 'both')) + 2*(self.loop_option and (self.loop_condition == 'both'))
        enter_loop_instruction_values = [i + instruction_value for i in range(num_loops)]
        exit_loop_instruction_values = [i + num_loops for i in enter_loop_instruction_values]
        loop_conditions = ['value', 'match'] if self.loop_condition == 'both' else [self.loop_condition]
        loop_conditions = loop_conditions * (1 + self.loop_option)
        loop_options = [False] * (1 + (self.loop_condition == 'both'))
        if self.loop_option:
            loop_options += [True] * (1 + (self.loop_condition == 'both'))
            
        for i in range(num_loops):
            loop_string = 'loop_skip_if_head0'
            loop_string += '_!=' if self.loop_option else '_=='
            loop_string += '_0' if loop_conditions[i] == 'value' else '_head1'

            instruction_set[loop_string] = (self.handle_forward_loops, {'instruction_value' : enter_loop_instruction_values[i], 
                                                              'loop_condition' : loop_conditions[i],
                                                              'enter_loop_instruction_values' : enter_loop_instruction_values, 
                                                              'exit_loop_instruction_values' : exit_loop_instruction_values,
                                                              'debug' : False,
                                                              'option' : loop_options[i]})
            instruction_value += 1
            loop_string = 'loop_repeat_if_head0'
            loop_string += '_==' if self.loop_option else '_!='
            loop_string += '_0' if loop_conditions[i] == 'value' else '_head1'
            
            instruction_set[loop_string] = (self.handle_backward_loops, {'instruction_value' : exit_loop_instruction_values[i],
                                                              'loop_condition' : loop_conditions[i],
                                                              'enter_loop_instruction_values' : enter_loop_instruction_values, 
                                                              'exit_loop_instruction_values' : exit_loop_instruction_values,
                                                              'debug' : False,
                                                              'option' : loop_options[i]})
            instruction_value += 1
        
        for head_index in [0, 1]:
            for dim in [0, 1, 2]:
                for direction in [-1, 1]:
                    instruction_set[f'head{head_index}_dim{dim}_move{direction}'] = (self.update_head_positions, {'instruction_value' : instruction_value, 
                                                                      'head_index' : head_index,
                                                                      'dim' : dim,
                                                                      'direction' : direction})
                    instruction_value += 1
        
        for head_index in [0, 1]:
            for increment in [-1, 1]:
                instruction_set[f'head{head_index}_value{increment}'] = (self.update_tape, {'instruction_value' : instruction_value,
                                                                         'head_index' : head_index, 
                                                                         'increment' : increment})
                instruction_value += 1
        
        if not self.no_copy:
            instruction_set['copy_head0_to_head1'] = (self.copy_tape_values, {'instruction_value' : instruction_value,
                                                                     'src_head_index' : 0, 
                                                                     'dest_head_index' : 1})
            instruction_value += 1
            instruction_set['copy_head1_to_head0'] = (self.copy_tape_values, {'instruction_value' : instruction_value,
                                                                     'src_head_index' : 1, 
                                                                     'dest_head_index' : 0})
            instruction_value += 1
        
        self.instruction_set = instruction_set
        
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
        instruction_value = kwargs.get('instruction_value')
        loop_condition = kwargs.get('loop_condition')
        enter_loop_instruction_values = kwargs.get('enter_loop_instruction_values')
        exit_loop_instruction_values = kwargs.get('exit_loop_instruction_values')
        debug = kwargs.get('debug')
        option = kwargs.get('option', False)
        
        # Create mask for height, width positions with enter loop instruction
        enter_mask = instructions == instruction_value
        
        if not enter_mask.any():
            return  # Exit early if no enter instructions are found
    
        # Create index using the exit_mask
        enter_indices = enter_mask.nonzero(as_tuple=True)
        
        # Identify the relevant heads
        enter_heads = self.heads[enter_indices]
        
        # Calculate the height, width, depth positions of relevant head0's
        head0_height_indices = (enter_indices[0] + enter_heads[:, 0, 0] - 1) % self.tape.shape[0]
        head0_width_indices = (enter_indices[1] + enter_heads[:, 0, 1] - 1) % self.tape.shape[1]
        head0_depth_indices = enter_heads[:, 0, 2]
        
        if loop_condition == 'value':
            if option:
                enter_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] != 0
            else:
                enter_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] == 0
        elif loop_condition == 'match':
            head1_height_indices = (enter_indices[0] + enter_heads[:, 1, 0] - 1) % self.tape.shape[0]
            head1_width_indices = (enter_indices[1] + enter_heads[:, 1, 1] - 1) % self.tape.shape[1]
            head1_depth_indices = enter_heads[:, 1, 2]
            if option:
                enter_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] != self.tape[head1_height_indices, head1_width_indices, head1_depth_indices]
            else:
                enter_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] == self.tape[head1_height_indices, head1_width_indices, head1_depth_indices]

        # Filter the positions where enter conditions are met
        valid_enter_positions = enter_mask.clone()
        valid_enter_positions[enter_indices] = enter_condition_met
    
        if not valid_enter_positions.any():
            return  # Exit early if no valid enter conditions are met
    
        # Create index of positions to search
        search_indices = valid_enter_positions.nonzero(as_tuple=True)
        search_area = self.tape[search_indices[0], search_indices[1]]
    
        # Find enter and exit positions within the search area
        enter_positions = torch.isin(search_area, torch.tensor(enter_loop_instruction_values, device=self.device))
        exit_positions = torch.isin(search_area, torch.tensor(exit_loop_instruction_values, device=self.device))
    
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
        instruction_value = kwargs.get('instruction_value')
        loop_condition = kwargs.get('loop_condition')
        enter_loop_instruction_values = kwargs.get('enter_loop_instruction_values')
        exit_loop_instruction_values = kwargs.get('exit_loop_instruction_values')
        debug = kwargs.get('debug')
        option = kwargs.get('loop_option', False)
        
        # Create mask for height, width positions with exit loop instructions
        exit_mask = instructions == instruction_value
    
        if not exit_mask.any():
            return  # Exit early if no exit instructions are found
    
        # Create index using the exit_mask
        exit_indices = exit_mask.nonzero(as_tuple=True)
        
        # Identify the relevant heads
        exit_heads = self.heads[exit_indices]
    
        # Calculate the height, width, depth positions of relevant head0's
        head0_height_indices = (exit_indices[0] + exit_heads[:, 0, 0] - 1) % self.tape.shape[0]
        head0_width_indices = (exit_indices[1] + exit_heads[:, 0, 1] - 1) % self.tape.shape[1]
        head0_depth_indices = exit_heads[:, 0, 2]
        
        if loop_condition == 'value':
            if option:
                exit_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] == 0
            else:
                exit_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] != 0
        elif loop_condition == 'match':
            head1_height_indices = (exit_indices[0] + exit_heads[:, 1, 0] - 1) % self.tape.shape[0]
            head1_width_indices = (exit_indices[1] + exit_heads[:, 1, 1] - 1) % self.tape.shape[1]
            head1_depth_indices = exit_heads[:, 1, 2]
            if option:
                exit_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] == self.tape[head1_height_indices, head1_width_indices, head1_depth_indices]
            else:
                exit_condition_met = self.tape[head0_height_indices, head0_width_indices, head0_depth_indices] != self.tape[head1_height_indices, head1_width_indices, head1_depth_indices]    

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
        enter_positions = torch.isin(flipped_search_area, torch.tensor(enter_loop_instruction_values, device=self.device))
        exit_positions = torch.isin(flipped_search_area, torch.tensor(exit_loop_instruction_values, device=self.device))
    
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
        
        invalid_backward_matches = first_enter_indices == 0
        first_enter_indices[invalid_backward_matches] = self.tape.shape[2]
        
        # Convert the matched indices back to the unflipped depth dimension
        first_enter_indices = self.tape.shape[2] - first_enter_indices - 1
    
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
            'iters' : self.iters,
            'reset_heads' : self.reset_heads,
            'loop_condition' : self.loop_condition,
            'loop_option' : self.loop_option,
            'no_copy' : self.no_copy
        }
        torch.save(state, path)
                
    @classmethod
    def load(cls, path, device='cpu'):
        
        # Load the state of the Abiogenesis class from disk
        state = torch.load(path)
        
        # Extract the dimensions from the loaded state
        height, width, tape_len = state['tape'].shape
        
        # Create a new instance of the class with the loaded state
        obj = cls(height=height, 
                  width=width, 
                  tape_len=tape_len, 
                  num_instructions=state.get('num_instructions', 64),
                  device=device,
                  reset_heads=state.get('reset_heads', True),
                  loop_condition=state.get('loop_condition', 'value'),
                  loop_option=state.get('loop_option', False),
                  no_copy=state.get('no_copy', False))
        
        # Restore the state to the desired device
        for key in ['tape', 'ip', 'heads']:
            setattr(obj, key, state[key].to(device))
        setattr(obj, 'iters', state.get('iters', 0))
        
        return obj
    
    def reset(self):
        self.tape[:] = torch.randint_like(self.tape, low=0, high=self.num_instructions)
        self.ip[:] = 0
        self.heads[:] = 1
        self.heads[:, :, :, -1] = 0
        
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
        assert str(device) in ['cpu', 'cuda']
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
    parser.add_argument('--stateful_heads', action='store_true', help='Determines if heads maintain their state or reset upon invalid instruction')
    parser.add_argument('--load', type=str, default='', help='Path to saved simulation')
    parser.add_argument('--loop_condition', type=str, default='value', help='Loops are conditioned on 0 (value) or matching head values (match)')
    parser.add_argument('--loop_option', action='store_true', help='Adds an additional set of loop instructions with opposite conditions')
    parser.add_argument('--no_copy', action='store_true', help='Removes copy operations from instruction set')
    parser.add_argument('--seed', type=int, default=0, help='Number of hand-coded replicators to seed into the simulation')
    parser.add_argument('--color_scheme', type=str, default='default', help='Color scheme for visualizations')
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
        env = Abiogenesis(args.height, 
                          args.width, 
                          args.depth, 
                          num_instructions=args.num_instructions, 
                          device=args.device, 
                          reset_heads=not args.stateful_heads,
                          loop_condition=args.loop_condition,
                          loop_option=args.loop_option,
                          no_copy=args.no_copy,
                          seed=args.seed,
                          color_scheme=args.color_scheme)
    print(env.instruction_set)

    # Run the simulation
    start_iter = env.iters
    with torch.no_grad():
        for i in range(start_iter, args.num_sims+1):
            
            if args.mutate_rate and not i % args.depth:
                 env.mutate(args.mutate_rate)
            if not i % args.image_save_interval:
                print(f"Iteration {i}")
                env.visualize(os.path.join(img_path, f'{i:09}.png'))
            if not i % args.state_save_interval:
                env.save(os.path.join(states_path, f'{i:09}.p'))    
                
            env.iterate()
        
if __name__ == "__main__":
    main()