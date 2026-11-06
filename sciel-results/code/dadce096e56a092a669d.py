import numpy as np
import time
import random
import math

N = 17
DEADLINE = 82.0  # 8s safety margin before 90s kill

def _verify(net):
    """
    Fast bit-parallel verification using the zero-one principle.
    For each of the 2^N boolean inputs, simulate the network using bit operations.
    Returns True if valid, False otherwise.
    """
    total = 1 << N
    # For each input, we represent the wire state as a single integer
    # where bit k corresponds to wire k (wire 0 = LSB, wire N-1 = MSB)
    # A compare-exchange (i, j): min goes to wire i, max goes to wire j
    #   If wire_i < wire_j: no change (i=0, j=1)
    #   If wire_i > wire_j: swap (i=1, j=0)
    #   If equal: no change
    # In terms of bits: if bit_i=1 and bit_j=0, we need to clear bit_i and set bit_j.
    
    # We'll process all inputs in parallel using vectorized operations
    # Actually, let's use a smarter approach:
    # For a given network, we can check if it sorts by verifying that
    # for every input pattern, the output is sorted.
    
    # Fast approach: represent each input as an integer, apply comparators
    # using bitwise operations, then check if result is sorted.
    
    # Precompute masks
    mask_i = 1 << 0  # placeholder
    mask_j = 1 << 1  # placeholder
    
    # We'll iterate through all 2^N inputs
    # For speed, we can use numpy arrays
    inputs = np.arange(total, dtype=np.uint32)
    
    for i, j in net:
        # For each input, check if wire i has 1 and wire j has 0
        # If so, swap them
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        
        # Need to swap when bit_i=1 and bit_j=0
        swap_mask = bit_i & (~bit_j) & 1
        # Clear bit i and set bit j where swap_mask is 1
        # Clear bit i: inputs & ~(1 << i)
        # Set bit j: inputs | (1 << j)
        # But only where swap_mask is 1
        
        # Vectorized: 
        # new_inputs = inputs.copy()
        # new_inputs[swap_mask==1] = (inputs[swap_mask==1] & ~(1<<i)) | (1<<j)
        # But this involves boolean indexing which can be slow.
        
        # Alternative: use arithmetic
        # Where swap_mask=1: subtract (1<<i) and add (1<<j)
        # delta = swap_mask * ((1<<j) - (1<<i))
        delta = swap_mask.astype(np.uint32) * ((1 << j) - (1 << i))
        inputs = (inputs + delta).astype(np.uint32)
    
    # Check if sorted: for all k, wire k <= wire k+1
    # A state is sorted iff for all k, not (bit_k=1 and bit_{k+1}=0)
    # i.e., for all k, ~(bit_k & ~bit_{k+1})
    # Unsorted if exists k such that bit_k=1 and bit_{k+1}=0
    
    for k in range(N - 1):
        bit_k = (inputs >> k) & 1
        bit_k1 = (inputs >> (k + 1)) & 1
        unsorted = bit_k & (~bit_k1) & 1
        if np.any(unsorted):
            return False
    
    return True

def _verify_single(net, input_val):
    """Verify a single input through the network. Returns True if sorted."""
    state = input_val
    for i, j in net:
        bit_i = (state >> i) & 1
        bit_j = (state >> j) & 1
        if bit_i == 1 and bit_j == 0:
            state = (state & ~(1 << i)) | (1 << j)
    # Check sorted
    for k in range(N - 1):
        if ((state >> k) & 1) == 1 and ((state >> (k+1)) & 1) == 0:
            return False
    return True

def _generate_seed():
    """
    Generates an Odd-Even Transposition network for N wires.
    """
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _sa_search(net, deadline):
    """
    Simulated annealing with add, remove, and swap moves.
    """
    start_time = time.time()
    best_net = net[:]
    best_size = len(net)
    current_net = net[:]
    current_size = len(net)
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling = 0.995
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Choose a move
        move_type = random.random()
        if move_type < 0.4:
            # Remove a random comparator
            if len(current_net) > 0:
                idx = random.randint(0, len(current_net) - 1)
                removed = current_net[idx]
                current_net.pop(idx)
                new_size = len(current_net)
                
                if _verify(current_net):
                    # Accept
                    if new_size < best_size:
                        best_net = current_net[:]
                        best_size = new_size
                else:
                    # Reject, put back
                    current_net.insert(idx, removed)
        elif move_type < 0.6:
            # Add a random comparator
            comp = random.choice(all_comparators)
            if comp not in current_net:
                current_net.append(comp)
                new_size = len(current_net)
                
                if _verify(current_net):
                    # Accept (even if larger, to explore)
                    if new_size < best_size:
                        best_net = current_net[:]
                        best_size = new_size
                else:
                    # Reject
                    current_net.pop()
        else:
            # Swap two comparators in the network
            if len(current_net) >= 2:
                idx1 = random.randint(0, len(current_net) - 1)
                idx2 = random.randint(0, len(current_net) - 1)
                if idx1 != idx2:
                    current_net[idx1], current_net[idx2] = current_net[idx2], current_net[idx1]
                    new_size = len(current_net)
                    
                    if _verify(current_net):
                        if new_size < best_size:
                            best_net = current_net[:]
                            best_size = new_size
                    else:
                        # Reject, swap back
                        current_net[idx1], current_net[idx2] = current_net[idx2], current_net[idx1]
    
    return best_net

def construct():
    """
    Constructs a size-efficient sorting network for N=17 wires.
    """
    start_time = time.time()
    
    # 1. Generate Seed
    net = _generate_seed()
    
    # 2. Verify Seed (should be valid)
    if not _verify(net):
        return net
    
    # 3. Simulated Annealing
    best_net = _sa_search(net, DEADLINE)
    
    return best_net
