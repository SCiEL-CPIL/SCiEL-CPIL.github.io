import numpy as np
import time
import random
import math

N = 18

def _odd_even_transposition_network(n):
    """Generates the standard Odd-Even Transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _is_valid_network_full(network, inputs):
    """
    Checks if the network sorts all inputs in the provided 'inputs' array.
    'inputs' is a numpy array of shape (2^N, N).
    """
    state = inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return np.all(state[:, :-1] <= state[:, 1:])

def _is_valid_network_sample(network, inputs, sample_size=2048):
    """
    Checks if the network sorts a random sample of inputs.
    """
    # Randomly select sample_size rows
    indices = np.random.choice(len(inputs), size=sample_size, replace=False)
    sampled_inputs = inputs[indices]
    
    state = sampled_inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return np.all(state[:, :-1] <= state[:, 1:])

def construct():
    start_time = time.time()
    deadline = 55.0
    
    # Generate inputs
    bits = np.arange(2**N, dtype=np.uint32)
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        inputs[:, k] = (bits >> k) & 1

    # Start with Odd-Even Transposition
    current_network = _odd_even_transposition_network(N)
    
    # Verify initial
    if not _is_valid_network_full(current_network, inputs):
        return current_network

    best_network = list(current_network)
    best_size = len(best_network)
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    current_size = len(current_network)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Choose a random move
        move_type = random.choice(['delete', 'swap', 'insert'])
        candidate = list(current_network)
        
        if move_type == 'delete' and len(candidate) > 0:
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif move_type == 'swap' and len(candidate) > 1:
            idx = random.randint(0, len(candidate) - 2)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
        elif move_type == 'insert':
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
        else:
            continue
            
        # Fast check
        if not _is_valid_network_sample(candidate, inputs):
            continue
            
        # Full check
        if not _is_valid_network_full(candidate, inputs):
            continue
            
        # Acceptance
        new_size = len(candidate)
        delta = new_size - current_size
        
        if delta < 0:
            # Improvement, always accept
            current_network = candidate
            current_size = new_size
            if new_size < best_size:
                best_size = new_size
                best_network = list(current_network)
        else:
            # Deterioration or neutral, accept with probability
            # delta is 0 for swaps, positive for inserts
            if delta == 0:
                # Swaps are neutral, accept with high probability to explore
                if random.random() < 0.5:
                    current_network = candidate
            else:
                # Insertions, accept with lower probability
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_network = candidate
                    current_size = new_size
                    
        T *= cooling_rate

    return best_network
