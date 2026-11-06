"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START
def compute_reset_length(a, b):
    """
    Computes the length of the shortest reset word for the automaton (a, b).
    Returns -1 if the automaton is not synchronizing.
    """
    start_mask = (1 << N) - 1
    
    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        if mask & (mask - 1) == 0:
            return dist
            
        # Apply 'a'
        next_mask_a = 0
        m = mask
        while m:
            lsb = m & -m
            state = lsb.bit_length() - 1
            next_mask_a |= 1 << a[state]
            m -= lsb
        if next_mask_a not in visited:
            visited.add(next_mask_a)
            queue.append((next_mask_a, dist + 1))
            
        # Apply 'b'
        next_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            state = lsb.bit_length() - 1
            next_mask_b |= 1 << b[state]
            m -= lsb
        if next_mask_b not in visited:
            visited.add(next_mask_b)
            queue.append((next_mask_b, dist + 1))
            
    return -1

def construct():
    deadline = time.time() + 80.0
    random.seed(42)
    
    # Initial best: Cycle + Sink
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_len = compute_reset_length(best_a, best_b)
    
    current_a = list(best_a)
    current_b = list(best_b)
    current_len = best_len
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Mutation
        if random.random() < 0.5:
            state = random.randint(0, N - 1)
            dest = random.randint(0, N - 1)
            new_a = current_a[:]
            new_b = current_b
            new_a[state] = dest
        else:
            state = random.randint(0, N - 1)
            dest = random.randint(0, N - 1)
            new_a = current_a
            new_b = current_b[:]
            new_b[state] = dest
            
        new_len = compute_reset_length(new_a, new_b)
        
        if new_len > current_len:
            current_a = new_a
            current_b = new_b
            current_len = new_len
            no_improve_count = 0
            
            if current_len > best_len:
                best_a = current_a
                best_b = current_b
                best_len = current_len
        else:
            no_improve_count += 1
            # Random walk or restart?
            # If stuck, restart from best
            if no_improve_count > max_no_improve:
                current_a = list(best_a)
                current_b = list(best_b)
                current_len = best_len
                no_improve_count = 0
                # Maybe perturb best_a/b slightly to escape?
                # Or just restart search from best.
                
    return [best_a, best_b]
# EVOLVE-BLOCK-END
