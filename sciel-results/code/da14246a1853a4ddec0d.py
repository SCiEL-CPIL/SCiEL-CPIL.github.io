import numpy as np
import time
import random

N = 16
NUM_INPUTS = 2 ** N

# Precompute all boolean inputs
def get_all_inputs():
    inputs = np.arange(NUM_INPUTS, dtype=np.uint8)
    bits = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for k in range(N):
        bits[:, k] = (inputs >> k) & 1
    return bits

ALL_INPUTS = get_all_inputs()
# Sample for fast evaluation
SAMPLE_SIZE = 2048
SAMPLE_INPUTS = ALL_INPUTS[:SAMPLE_SIZE]

def simulate_sample(network):
    """
    Simulate network on a sample of inputs.
    Returns the number of unsorted inputs in the sample.
    """
    if not network:
        return SAMPLE_SIZE
        
    x = SAMPLE_INPUTS.copy()
    
    for i, j in network:
        min_val = np.minimum(x[:, i], x[:, j])
        max_val = np.maximum(x[:, i], x[:, j])
        x[:, i] = min_val
        x[:, j] = max_val
        
    # Count unsorted
    # Sorted if x[:, :-1] <= x[:, 1:]
    # Unsorted if any adjacent pair is 1, 0
    # diff = x[:, 1:] - x[:, :-1]
    # unsorted = np.sum(diff < 0)
    diff = x[:, 1:].astype(np.int8) - x[:, :-1].astype(np.int8)
    return np.sum(diff < 0)

def simulate_full(network):
    """
    Simulate network on all inputs.
    Returns True if the network sorts all inputs.
    """
    if not network:
        return False
        
    x = ALL_INPUTS.copy()
    
    for i, j in network:
        min_val = np.minimum(x[:, i], x[:, j])
        max_val = np.maximum(x[:, i], x[:, j])
        x[:, i] = min_val
        x[:, j] = max_val
        
    diff = x[:, 1:].astype(np.int8) - x[:, :-1].astype(np.int8)
    return np.sum(diff < 0) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial network: Odd-even transposition (120 comparators)
    current_network = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
    
    best_network = current_network.copy()
    best_len = len(current_network)
    
    # Verify initial
    if simulate_full(current_network):
        best_network = current_network.copy()
        best_len = len(current_network)
    else:
        return current_network

    candidates = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Greedy Pruning
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            test_network = current_network[:idx] + current_network[idx+1:]
            
            # Check on sample first
            if simulate_sample(test_network) == 0:
                # Check on full
                if simulate_full(test_network):
                    current_network = test_network
                    improved = True
                    best_network = current_network.copy()
                    best_len = len(current_network)
                    break
        
        if improved:
            continue
        
        # Phase 2: Swapping
        if len(current_network) > 0:
            idx = random.randint(0, len(current_network) - 1)
            old_comp = current_network[idx]
            
            attempts = 0
            while attempts < 10 and time.time() < deadline:
                new_comp = random.choice(candidates)
                if new_comp == old_comp:
                    attempts += 1
                    continue
                
                test_network = current_network.copy()
                test_network[idx] = new_comp
                
                if simulate_sample(test_network) == 0:
                    if simulate_full(test_network):
                        current_network = test_network
                        improved = True
                        best_network = current_network.copy()
                        best_len = len(current_network)
                        break
                attempts += 1
        
        # If stuck, add a comparator and prune
        if not improved and time.time() < deadline - 2.0:
            new_comp = random.choice(candidates)
            if new_comp not in current_network:
                current_network.append(new_comp)

    return best_network
