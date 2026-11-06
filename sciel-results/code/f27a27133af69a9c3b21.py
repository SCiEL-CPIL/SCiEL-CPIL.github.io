"""
Sorting network construction for N=17 wires.
Uses a combination of:
1. Fast NumPy-based validation over all 2^17 Boolean inputs (zero-one principle).
2. Aggressive greedy pruning from a seed odd-even transposition network.
3. Local search with diverse moves (remove, swap, move, replace) and simulated annealing
   to escape local minima.
4. A final verification pass.

The goal is to minimize the number of comparators while ensuring correctness.
"""

import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136
MAX_COMPARATORS = 4096
EVALUATION_TIMEOUT = 90.0  # Hard limit from evaluator
SAFETY_MARGIN = 5.0        # Reserve time for final checks and return
DEADLINE = time.time() + EVALUATION_TIMEOUT - SAFETY_MARGIN

# Precompute all 2^N Boolean inputs and their sorted versions
# Shape: (2^N, N)
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)


def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^N inputs using the zero-one principle.
    Returns True if the network is valid.
    """
    if len(comparators) > MAX_COMPARATORS:
        return False
    if not comparators:
        return False

    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val

    return np.array_equal(state, _SORTED_INPUTS)


def generate_seed_network():
    """
    Generate the standard odd-even transposition network for N wires.
    """
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network


def prune_network(network, deadline):
    """
    Aggressively prune redundant comparators from the network.
    Uses a backward scan, restarting after each removal to catch cascading redundancies.
    """
    current = list(network)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = current.copy()
            candidate.pop(idx)
            if is_sorted_network(candidate):
                current = candidate
                changed = True
                break  # Restart scan from the end
    return current


def local_search(network, deadline):
    """
    Local search with simulated annealing to minimize comparator count.
    Moves: remove, swap, move, replace.
    Accepts worse solutions with probability based on simulated annealing.
    """
    current = list(network)
    best = list(network)
    best_len = len(current)
    
    # Simulated annealing parameters
    temperature = 0.5
    cooling_rate = 0.995
    min_temperature = 0.01
    
    # Precompute all possible comparator pairs for random replacement
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    step = 0
    while time.time() < deadline and temperature > min_temperature:
        step += 1
        move_type = random.randint(0, 3)
        neighbor = current.copy()
        
        if move_type == 0 and neighbor:
            # Remove a random comparator
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
            
        elif move_type == 1 and len(neighbor) >= 2:
            # Swap two comparators
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            else:
                continue
                
        elif move_type == 2 and len(neighbor) >= 2:
            # Move a comparator to a new position
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                comp = neighbor.pop(i1)
                neighbor.insert(i2, comp)
            else:
                continue
                
        else:
            # Replace a random comparator with a random new one
            if not neighbor:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            new_comp = random.choice(all_pairs)
            neighbor[idx] = new_comp
        
        # Validate
        if not is_sorted_network(neighbor):
            continue
        
        neighbor_len = len(neighbor)
        current_len = len(current)
        
        # Acceptance criterion
        if neighbor_len < current_len:
            # Always accept improvement
            current = neighbor
            if neighbor_len < best_len:
                best = neighbor.copy()
                best_len = neighbor_len
                # Re-prune after finding a better solution
                pruned = prune_network(best, min(time.time() + 2.0, deadline))
                if len(pruned) < best_len:
                    best = pruned
                    best_len = len(best)
                    current = best.copy()
        else:
            # Simulated annealing: accept worse or equal with probability
            delta = neighbor_len - current_len
            if delta == 0:
                # Accept equal-length moves with some probability to explore structure
                if random.random() < 0.3:
                    current = neighbor
            else:
                # Accept worse moves with probability exp(-delta / temperature)
                prob = np.exp(-delta / temperature)
                if random.random() < prob:
                    current = neighbor
        
        # Cool down
        temperature *= cooling_rate
    
    return best, best_len


def construct():
    start_time = time.time()
    deadline = start_time + EVALUATION_TIMEOUT - SAFETY_MARGIN
    
    # 1. Generate seed network
    seed_network = generate_seed_network()
    
    # Verify seed
    if not is_sorted_network(seed_network):
        return seed_network  # Fallback
    
    # 2. Initial aggressive pruning
    pruning_deadline = deadline - 20.0  # Reserve 20s for local search
    pruned_network = prune_network(seed_network, pruning_deadline)
    
    # 3. Local search with simulated annealing
    ls_deadline = deadline - 2.0  # Reserve 2s for final verification
    best_network, best_len = local_search(pruned_network, ls_deadline)
    
    # 4. Final verification and cleanup
    if not is_sorted_network(best_network):
        # Fallback to seed if something went wrong
        return seed_network
    
    # One more quick pruning pass on the final best
    final_pruned = prune_network(best_network, deadline)
    if len(final_pruned) < len(best_network):
        best_network = final_pruned
    
    return best_network


if __name__ == "__main__":
    result = construct()
    print(f"Constructing sorting network for N={N}")
    print(f"Number of comparators: {len(result)}")
    print(f"Is valid: {is_sorted_network(result)}")
