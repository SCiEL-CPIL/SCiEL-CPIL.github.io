"""Combined program for sorting networks on 17 wires.
Uses pre-computed global inputs for fast validation and a hybrid pruning/local search strategy.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute the input matrix and sorted reference once.
# Shape: (2^N, N)
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a subset of indices for fast sampling
_SAMPLE_SIZE = 2048
_SAMPLE_INDICES = np.random.choice(1 << N, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses pre-computed _INPUTS and _SORTED_INPUTS for speed.
    """
    if len(comparators) > 4096:
        return False
    
    # If no comparators, it's only valid if inputs are already sorted (false for N>1)
    if not comparators:
        return False

    # Copy the input matrix to avoid modifying the global constant
    state = _INPUTS.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return np.array_equal(state, _SORTED_INPUTS)

def is_sorted_network_sample(comparators):
    """
    Fast check using a subset of inputs. Returns True if it passes the sample.
    False negatives are possible (if it fails sample, it's definitely invalid),
    but False positives are also possible (passes sample, might fail full).
    So we only use this as a filter: if False, reject immediately. If True, need full check.
    """
    if len(comparators) > 4096:
        return False
    if not comparators:
        return False

    state = _SAMPLE_INPUTS.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return np.array_equal(state, _SAMPLE_SORTED)

def try_remove_comparator(network, idx):
    """Try removing comparator at index idx. Returns True if still valid."""
    if len(network) <= 1:
        return False
    candidate = network[:idx] + network[idx+1:]
    # Use sample check as a fast filter to avoid expensive full simulation
    if not is_sorted_network_sample(candidate):
        return False
    return is_sorted_network(candidate)

def construct():
    start_time = time.time()
    # Safety margin: aim to finish well before 60s
    deadline = start_time + 55.0
    
    # 1. Generate Seed Network (Odd-Even Transposition)
    # Standard OET for N=17: 17 phases, ~8-9 comparators per phase -> ~136 total
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    # Verify seed (should be true)
    if not is_sorted_network(seed_network):
        # Fallback: return seed anyway, though it shouldn't happen
        return seed_network

    best_network = seed_network
    best_len = len(seed_network)
    current = list(seed_network)
    
    # 2. Aggressive Pruning Phase
    # Try to remove any comparator that is redundant.
    # We do multiple passes because removing one can make another redundant.
    # We iterate backwards to avoid index shifting issues during the pass,
    # but since we break and restart on success, we can also just do a full scan.
    
    pruning_deadline = deadline - 20.0 # Reserve more time for local search
    
    changed = True
    while changed and time.time() < pruning_deadline:
        changed = False
        # Iterate from end to start to handle deletions without index issues in a single pass
        # However, if we delete, we break and restart the while loop to catch new redundancies.
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > pruning_deadline:
                break
            
            if try_remove_comparator(current, idx):
                current.pop(idx)
                best_network = current.copy()
                best_len = len(current)
                changed = True
                # Break the inner loop to restart the outer while loop
                # This ensures we check the new list from the beginning (or end) again
                break
        
    # 3. Local Search Phase
    # Try to improve further by swapping, moving, or replacing comparators.
    # We focus on small perturbations to the current best.
    # Use sample-based validation to speed up the search loop.
    
    local_search_deadline = deadline - 2.0
    current = best_network.copy()
    current_len = best_len
    
    # Pre-generate a pool of random comparators for speed
    random_comparators = set()
    for _ in range(200):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        random_comparators.add((i, j))
    random_comparator_list = list(random_comparators)
    
    while time.time() < local_search_deadline:
        # Randomly select a move type with weights favoring removal and replacement
        # We increase weight of removal to find smaller networks
        move_type = random.choices([0, 1, 2, 3], weights=[0.4, 0.2, 0.2, 0.2])[0]
        neighbor = current.copy()
        
        if move_type == 0 and neighbor:
            # Strategy: Remove a random comparator
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
            
        elif move_type == 1 and len(neighbor) >= 2:
            # Strategy: Swap adjacent comparators
            # This is often more effective than random swaps for sorting networks
            idx = random.randint(0, len(neighbor) - 2)
            neighbor[idx], neighbor[idx+1] = neighbor[idx+1], neighbor[idx]
                
        elif move_type == 2:
            # Strategy: Replace a random comparator with a random new one
            if not neighbor:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            new_comp = random_comparator_list[random.randint(0, len(random_comparator_list) - 1)]
            neighbor[idx] = new_comp
            
        else:
            # Strategy: Move a comparator to a different position
            if len(neighbor) < 2:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            new_idx = random.randint(0, len(neighbor) - 1)
            if idx != new_idx:
                comp = neighbor.pop(idx)
                neighbor.insert(new_idx, comp)
            else:
                continue
        
        # Check validity using sample first
        if not is_sorted_network_sample(neighbor):
            continue
            
        # If sample passes, do full check
        if is_sorted_network(neighbor):
            neighbor_len = len(neighbor)
            if neighbor_len < current_len:
                current = neighbor
                current_len = neighbor_len
                if current_len < best_len:
                    best_network = current.copy()
                    best_len = current_len
            # Accept equal-length valid networks with higher probability to explore structure
            elif neighbor_len == current_len:
                if random.random() < 0.7:
                    current = neighbor

    # Final Check: Ensure we return a valid network
    if is_sorted_network(best_network):
        return best_network
    else:
        # Fallback to seed if something went wrong
        return seed_network
