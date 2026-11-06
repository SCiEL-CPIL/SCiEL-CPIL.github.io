import numpy as np
import time
import random

N = 18

# Pre-compute constants for verification
STATES = np.arange(2**N, dtype=np.uint32).reshape(-1, 1)
SHIFTS = np.arange(N, dtype=np.uint32)

def _is_valid(net):
    """
    Checks if the network sorts all boolean inputs.
    Uses vectorized NumPy operations.
    """
    # Initialize bits from states
    # bits shape: (2**N, N)
    bits = (STATES >> SHIFTS) & 1
    
    # Apply comparators
    for i, j in net:
        a = bits[:, i]
        b = bits[:, j]
        mask = a > b
        # Swap if a > b
        # To avoid temporary arrays if possible, but np.where is fine
        bits[:, i] = np.where(mask, b, a)
        bits[:, j] = np.where(mask, a, b)
        
    # Check sortedness
    # A sequence is sorted if bits[:, k] <= bits[:, k+1] for all k
    # We can check if the sum of differences is 0? No, differences can be negative.
    # Check if any difference is negative.
    diffs = np.diff(bits, axis=1)
    return not np.any(diffs < 0)

def construct():
    # Seed network: Odd-Even Transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # 1. Greedy Removal
    i = len(network) - 1
    while i >= 0:
        temp_net = network[:i] + network[i+1:]
        if _is_valid(temp_net):
            network = temp_net
        else:
            i -= 1
            
    best_net = network
    best_len = len(network)
    
    # 2. Local Search (Hill Climbing)
    start_time = time.time()
    timeout = 80.0
    current_net = best_net
    
    # Use a random seed for reproducibility in case of ties, but randomness is key for escape
    random.seed(42)
    
    while time.time() - start_time < timeout:
        # Attempt to remove more comparators
        i = len(current_net) - 1
        while i >= 0:
            temp_net = current_net[:i] + current_net[i+1:]
            if _is_valid(temp_net):
                current_net = temp_net
            else:
                i -= 1
                
        if len(current_net) < best_len:
            best_len = len(current_net)
            best_net = current_net
            
        # If we can't remove more, try to rearrange
        if len(current_net) > 1:
            i = random.randint(0, len(current_net) - 1)
            j = random.randint(0, len(current_net) - 1)
            
            # Strategy 1: Swap
            new_net = current_net[:]
            new_net[i], new_net[j] = new_net[j], new_net[i]
            if _is_valid(new_net):
                current_net = new_net
                continue
                
            # Strategy 2: Move i to j
            comp = current_net[i]
            new_net = current_net[:i] + current_net[i+1:]
            insert_pos = j if i < j else j + 1
            new_net = new_net[:insert_pos] + [comp] + new_net[insert_pos:]
            if _is_valid(new_net):
                current_net = new_net
                continue

            # Strategy 3: Move i to a nearby position (i-1 or i+1)
            if i > 0:
                new_net = current_net[:]
                comp = new_net.pop(i)
                new_net.insert(i-1, comp)
                if _is_valid(new_net):
                    current_net = new_net
                    continue
            if i < len(current_net) - 1:
                new_net = current_net[:]
                comp = new_net.pop(i)
                new_net.insert(i+1, comp)
                if _is_valid(new_net):
                    current_net = new_net
                    continue

    return best_net
