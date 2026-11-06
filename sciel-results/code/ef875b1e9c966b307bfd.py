import numpy as np
import time
import random
import math

N = 18
NUM_INPUTS = 1 << N

def _generate_inputs():
    """Generate all 2^N Boolean inputs as a numpy array of shape (2^N, N)."""
    X = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for k in range(N):
        X[:, k] = (np.arange(NUM_INPUTS, dtype=np.uint64) >> k) & 1
    return X

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _verify(network, X):
    """
    Verify if the network sorts all inputs.
    Returns True if sorted, False otherwise.
    """
    state = X.copy()
    for i, j in network:
        a = state[:, i].copy()
        b = state[:, j].copy()
        state[:, i] = a & b
        state[:, j] = a | b
    
    # Check if sorted: for all k, state[k] <= state[k+1]
    return not np.any(state[:, :-1] > state[:, 1:])

def construct():
    X = _generate_inputs()
    current_net = _odd_even_transposition()
    best_net = current_net.copy()
    best_cost = len(current_net)
    
    # Pre-verify seed
    if not _verify(current_net, X):
        return current_net

    start_time = time.time()
    deadline = start_time + 85.0
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    # Move probabilities
    p_remove = 0.4
    p_add = 0.3
    p_swap = 0.2
    p_mutate = 0.1
    
    while time.time() < deadline:
        # Generate move
        move_type = random.random()
        new_net = current_net.copy()
        
        if move_type < p_remove:
            if len(new_net) > 0:
                idx = random.randrange(len(new_net))
                new_net.pop(idx)
            else:
                continue
        elif move_type < p_remove + p_add:
            i = random.randrange(N)
            j = random.randrange(N)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            new_net.append((i, j))
        elif move_type < p_remove + p_add + p_swap:
            if len(new_net) >= 2:
                i = random.randrange(len(new_net))
                j = random.randrange(len(new_net))
                if i != j:
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                else:
                    continue
            else:
                continue
        else: # mutate
            if len(new_net) > 0:
                idx = random.randrange(len(new_net))
                i, j = new_net[idx]
                if random.random() < 0.5:
                    new_i = random.randrange(N)
                    if new_i == j:
                        continue
                    if new_i > j:
                        new_net[idx] = (j, new_i)
                    else:
                        new_net[idx] = (new_i, j)
                else:
                    new_j = random.randrange(N)
                    if new_j == i:
                        continue
                    if new_j < i:
                        new_net[idx] = (new_j, i)
                    else:
                        new_net[idx] = (i, new_j)
            else:
                continue

        # Verify
        if _verify(new_net, X):
            new_cost = len(new_net)
            if new_cost < best_cost:
                best_cost = new_cost
                best_net = new_net.copy()
            
            delta = new_cost - len(current_net)
            if delta < 0:
                current_net = new_net
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                if random.random() < prob:
                    current_net = new_net
        else:
            pass
            
        T *= cooling
        if T < min_T:
            T = 1.0
            cooling = 0.9995

    return best_net
