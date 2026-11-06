import random
import numpy as np
from collections import deque

N = 12
ITERATIONS = 100000
START_TEMP = 5.0
COOLING = 0.9999
MIN_TEMP = 0.01
RANDOM_SEED = 42

def _eval_transitions(transitions):
    """
    Returns (is_synchronizing, reset_length)
    Uses bitmask BFS for speed.
    """
    full_set = (1 << N) - 1
    
    # Precompute transition bitmasks
    # trans_bits[l][s] is the bitmask for the single state transitions[l][s]
    trans_bits = [[1 << transitions[l][s] for s in range(N)] for l in range(2)]
    
    visited = {full_set}
    queue = deque([(full_set, 0)])
    
    while queue:
        subset, depth = queue.popleft()
        # Check if singleton: subset has exactly one bit set
        if subset & (subset - 1) == 0:
            return True, depth
        
        for l in range(2):
            next_subset = 0
            # Iterate through bits of subset
            # N=12, so unrolling or simple loop is fine
            for s in range(N):
                if subset & (1 << s):
                    next_subset |= trans_bits[l][s]
            
            if next_subset not in visited:
                visited.add(next_subset)
                queue.append((next_subset, depth + 1))
                
    return False, 0

def construct():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Phase 1: Find an initial synchronizing automaton
    current = [[random.randint(0, N - 1) for _ in range(N)] for _ in range(2)]
    is_sync, depth = _eval_transitions(current)
    
    attempts = 0
    while not is_sync and attempts < 1000:
        # Mutate
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        current[letter][state] = random.randint(0, N - 1)
        is_sync, depth = _eval_transitions(current)
        attempts += 1
        
    if not is_sync:
        # Fallback to a known simple sync automaton (e.g., a chain)
        # State 0 -> 1 -> 2 ... -> 11 -> 0 on '0'
        # State i -> i on '1' (identity)
        # This synchronizes in 11 steps (word 0...0)
        current = [[(i + 1) % N for i in range(N)], [i for i in range(N)]]
        is_sync, depth = _eval_transitions(current)
        
    current_score = depth
    best = [row[:] for row in current]
    best_score = current_score
    
    temp = START_TEMP
    iteration = 0
    
    while temp > MIN_TEMP and iteration < ITERATIONS:
        # Create neighbor
        neighbor = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_val = neighbor[letter][state]
        new_val = random.randint(0, N - 1)
        
        if new_val == old_val:
            iteration += 1
            temp *= COOLING
            continue
            
        neighbor[letter][state] = new_val
        
        n_is_sync, n_depth = _eval_transitions(neighbor)
        
        if not n_is_sync:
            # Reject non-synchronizing neighbors
            iteration += 1
            temp *= COOLING
            continue
        
        # Neighbor is synchronizing
        neighbor_score = n_depth
        
        if neighbor_score >= current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = [row[:] for row in current]
                best_score = current_score
        else:
            # Metropolis criterion
            delta = current_score - neighbor_score
            # delta > 0
            accept_prob = np.exp(-delta / temp)
            if random.random() < accept_prob:
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best = [row[:] for row in current]
                    best_score = current_score
        
        iteration += 1
        temp *= COOLING
        
    return best
