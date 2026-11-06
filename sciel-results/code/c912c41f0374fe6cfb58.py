"""
Size-efficient sorting networks on 17 wires – Enhanced Simulated Annealing approach.
Uses bitmask verification with multi-phase search: aggressive deletion, then fine-tuning.
"""

import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute initial state bitmasks for verification
INITIAL_STATE = [0] * N
for i in range(N):
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    INITIAL_STATE[i] = val

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    state = list(INITIAL_STATE)
    st = state
    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj
        
    for i in range(N - 1):
        if st[i] & ~st[i + 1]:
            return False
            
    return True

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _greedy_delete(network):
    """
    Iteratively remove redundant comparators.
    Returns the reduced network.
    """
    current = list(network)
    improved = True
    while improved:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(len(current)))
        random.shuffle(indices)
        for idx in indices:
            if len(current) <= 10:
                break
            candidate = current[:idx] + current[idx+1:]
            if _is_valid(candidate):
                current = candidate
                improved = True
                break # Restart loop with new list
    return current

def construct():
    """
    Construct a sorting network with as few comparators as possible.
    Uses Greedy Deletion with Random Kicks to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Start with Odd-Even Transposition
    current_network = _construct_odd_even_transposition()
    best_network = list(current_network)
    best_len = len(current_network)
    
    if not _is_valid(current_network):
        return best_network
    
    # Initial greedy reduction
    current_network = _greedy_delete(current_network)
    if len(current_network) < best_len:
        best_network = list(current_network)
        best_len = len(current_network)
    
    # Main search loop: Kick and Reduce
    while time.time() < deadline:
        # Check if we can break early if we hit the theoretical lower bound? 
        # Lower bound is 63, but we are far from it.
        
        # 1. Kick: Add 1-3 random comparators to escape local minimum
        # Adding comparators preserves validity.
        num_kicks = random.randint(1, 3)
        for _ in range(num_kicks):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            # Insert at random position
            idx = random.randint(0, len(current_network))
            current_network.insert(idx, (i, j))
        
        # 2. Reduce: Greedy deletion
        current_network = _greedy_delete(current_network)
        
        # 3. Update best
        if len(current_network) < best_len:
            best_network = list(current_network)
            best_len = len(current_network)
            
        # Optional: If we get stuck at the same length for too long, 
        # we might try a more aggressive kick or restart from best.
        # For now, the random insertion should provide enough diversity.

    return best_network
