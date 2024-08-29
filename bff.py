# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:57:17 2024

@author: ruhe
"""

import torch

num_sims = 1e7

class Abiogenesis(object):
    def __init__(self, num_tapes, tape_len, device='cpu'):
        assert tape_len < 256
        assert 256 % tape_len == 0
        self.device = device
        self.tape = torch.randint(low=0, high=255, size=(num_tapes, tape_len), dtype=torch.uint8, device=device)
        self.ip = torch.zeros(num_tapes, device=device).long()
        self.heads = torch.zeros((num_tapes, 2, 2), dtype=torch.int64, device=device)

    def iterate(self):
        
        # Reset head positions to 0 where self.ip is zero
        reset_mask = self.ip == 0
        self.heads[reset_mask] = 0
        
        # Correctly indexing the instructions using torch.gather and casting to int64
        instructions = self.tape[torch.arange(self.tape.shape[0], device=self.device), self.ip]
        
        # Boolean masks for instruction types
        dec_head0_dim0 = torch.where(instructions == 1)[0]
        inc_head0_dim0 = torch.where(instructions == 2)[0]
        dec_head0_dim1 = torch.where(instructions == 3)[0]
        inc_head0_dim1 = torch.where(instructions == 4)[0]
        dec_head1_dim0 = torch.where(instructions == 5)[0]
        inc_head1_dim0 = torch.where(instructions == 6)[0]
        dec_head1_dim1 = torch.where(instructions == 7)[0]
        inc_head1_dim1 = torch.where(instructions == 8)[0]
        dec_tape = torch.where(instructions == 9)[0]
        inc_tape = torch.where(instructions == 10)[0]
        copy_0_to_1 = torch.where(instructions == 11)[0]
        copy_1_to_0 = torch.where(instructions == 12)[0]
        enter_loop = torch.where(instructions == 13)[0]
        exit_loop = torch.where(instructions == 14)[0]

        self.heads[dec_head0_dim0, 0, 0] = (self.heads[dec_head0_dim0, 0, 0] - 1) % 3
        self.heads[inc_head0_dim0, 0, 0] = (self.heads[inc_head0_dim0, 0, 0] + 1) % 3
        self.heads[dec_head0_dim1, 0, 1] = (self.heads[dec_head0_dim1, 0, 1] - 1) % self.tape.shape[1]
        self.heads[inc_head0_dim1, 0, 1] = (self.heads[inc_head0_dim1, 0, 1] + 1) % self.tape.shape[1]
        self.heads[dec_head1_dim0, 1, 0] = (self.heads[dec_head1_dim0, 1, 0] - 1) % 3
        self.heads[inc_head1_dim0, 1, 0] = (self.heads[inc_head1_dim0, 1, 0] + 1) % 3
        self.heads[dec_head1_dim1, 1, 1] = (self.heads[dec_head1_dim1, 1, 1] - 1) % self.tape.shape[1]
        self.heads[inc_head1_dim1, 1, 1] = (self.heads[inc_head1_dim1, 1, 1] + 1) % self.tape.shape[1]

        # Correctly calculate tape indices for the operations based on the heads
        tape_indices = torch.clamp(dec_tape + self.heads[dec_tape, 0, 0] - 1, 0, self.tape.shape[0] - 1)        
        positions = self.heads[dec_tape, 0, 1]

        self.tape[tape_indices, positions] -= 1
        
        tape_indices = torch.clamp(inc_tape + self.heads[inc_tape, 0, 0] - 1, 0, self.tape.shape[0] - 1)        
        positions = self.heads[inc_tape, 0, 1]

        self.tape[tape_indices, positions] += 1
        
                # Directly use indices to extract the relevant positions for the copy operations
        src_positions_0_for_copy_0_to_1 = self.heads[copy_0_to_1, 0, 1]  # Positions from head0 for copy 0 to 1
        src_positions_1_for_copy_0_to_1 = self.heads[copy_0_to_1, 1, 1]  # Positions from head1 for copy 0 to 1
        
        src_positions_0_for_copy_1_to_0 = self.heads[copy_1_to_0, 0, 1]  # Positions from head0 for copy 1 to 0
        src_positions_1_for_copy_1_to_0 = self.heads[copy_1_to_0, 1, 1]  # Positions from head1 for copy 1 to 0
        
        # Directly adjust indices based on self.heads with proper clamping
        src_tape_0 = torch.clamp(copy_0_to_1 + self.heads[copy_0_to_1, 0, 0] - 1, 0, self.tape.shape[0] - 1)
        src_tape_1 = torch.clamp(copy_1_to_0 + self.heads[copy_1_to_0, 1, 0] - 1, 0, self.tape.shape[0] - 1)
        
        # Copy values from head0 to head1
        self.tape[copy_0_to_1, src_positions_1_for_copy_0_to_1] = self.tape[src_tape_0, src_positions_0_for_copy_0_to_1]
        
        # Copy values from head1 to head0
        self.tape[copy_1_to_0, src_positions_0_for_copy_1_to_0] = self.tape[src_tape_1, src_positions_1_for_copy_1_to_0]

        # Dimensions
        num_tapes = self.tape.shape[0]
        tape_len = self.tape.shape[1]
        
        # Generate a matrix of offsets for forward and backward searches
        forward_offsets = (self.ip[:, None] + torch.arange(1, tape_len).to(self.device)) % tape_len
        backward_offsets = (self.ip[:, None] - torch.arange(1, tape_len).to(self.device)) % tape_len
        
        # Correct tape indices using enter_loop and exit_loop
        tape_indices = (enter_loop + self.heads[enter_loop, 0, 0] - 1).clamp(0, num_tapes - 1)
        positions = self.heads[enter_loop, 0, 1]  # Positions within the tape
        
        # Handle loop entry: Find matching exit `]` if head value is 0
        enter_mask = (self.tape[tape_indices, positions] == 0)  # Check if head values are 0
        if enter_mask.any():
            # Vectorized search for next `]`
            forward_search = self.tape[enter_loop].unsqueeze(1)  # Add dimension for broadcasting
            exits_found = forward_search.gather(2, forward_offsets[enter_loop].unsqueeze(1)) == 14  # Find `]` in forward search
            next_exits = exits_found.int().argmax(dim=2).squeeze(1).type(torch.uint8)  # Index of the first occurrence
            found_mask = exits_found.any(dim=2).squeeze(1)  # Check if any match found
            if (enter_mask & found_mask).any():
                self.ip[enter_loop[enter_mask & found_mask]] = (self.ip[enter_loop[enter_mask & found_mask]] + next_exits[enter_mask & found_mask] - 1) % tape_len
            if (enter_mask & ~found_mask).any():
                self.ip[enter_loop[enter_mask & ~found_mask]] = -1  # Reset if no match found
        
        
        # Correct tape indices for exit_loop
        tape_indices = (exit_loop + self.heads[exit_loop, 0, 0] - 1).clamp(0, num_tapes - 1)
        positions = self.heads[exit_loop, 0, 1]  # Positions within the tape
        
        # Handle loop exit: Find matching enter `[` if head value is not 0
        exit_mask = (self.tape[tape_indices, positions] != 0)  # Check if head values are not 0
        if exit_mask.any():
            # Vectorized search for previous `[`
            backward_search = self.tape[exit_loop].unsqueeze(1)  # Add dimension for broadcasting
            enters_found = backward_search.gather(2, backward_offsets[exit_loop].unsqueeze(1)) == 13  # Find `[` in backward search
            prev_enters = enters_found.int().argmax(dim=2).squeeze(1).type(torch.uint8)  # Index of the first occurrence
            found_mask = enters_found.any(dim=2).squeeze(1)  # Check if any match found
            self.ip[exit_loop[exit_mask & found_mask]] = (self.ip[exit_loop[exit_mask & found_mask]] - prev_enters[exit_mask & found_mask] - 1) % tape_len
            self.ip[exit_loop[exit_mask & ~found_mask]] = -1  # Reset if no match found
        
        # Update instruction pointer at the end
        self.ip = (self.ip + 1) % tape_len
        
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
        # Create a new instance of the class with the loaded state
        obj = cls(num_tapes=state['tape'].shape[0], tape_len=state['tape'].shape[1], device=state['device'])
        obj.tape = state['tape']
        obj.ip = state['ip']
        obj.heads = state['heads']
        return obj
            
        
env = Abiogenesis(1024, 64, device='cuda')
for i in range(int(num_sims)):
    env.iterate()
    if not i % 10:
        env.mutate(0.001)
    if not i % 10000:
        print(f"iteration {i}")
        print("tape[0] values: ")
        print(env.tape[0])
        env.save(rf'D:\BFF\results\{i}.p')
