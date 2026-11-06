import time
import numpy as np
import random

N = 17
DEADLINE = 55.0  # Strict safety margin
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _make_all_inputs():
    """All 2^N boolean inputs as uint8 array shape (2^N, N)."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def simulate_network(comps, initial_data):
    """Simulate network on given data. Returns final uint8 array."""
    data = initial_data.copy()
    for a, b in comps:
        lo = np.minimum(data[:, a], data[:, b])
        hi = np.maximum(data[:, a], data[:, b])
        data[:, a] = lo
        data[:, b] = hi
    return data

def is_sorted_network(comps):
    """Check if network sorts all boolean inputs (zero-one principle)."""
    final = simulate_network(comps, _get_all_inputs())
    return np.all(final[:, :-1] <= final[:, 1:])

def count_inversions(data):
    """Count total inversions in all rows of data."""
    n_rows, n_cols = data.shape
    inv = 0
    for i in range(n_cols - 1):
        for j in range(i + 1, n_cols):
            inv += np.sum(data[:, i] > data[:, j])
    return inv

def generate_oetn():
    """Generate Odd-Even Transposition Network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

def construct():
    """Construct a sorting network using Simulated Annealing."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with OETN
    comps = generate_oetn()
    current_comps = comps[:]
    
    # Pre-generate all possible pairs for replacement moves
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Simulated Annealing parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    best_comps = current_comps[:]
    best_score = len(current_comps)
    
    # Use a random subset of inputs for fast cost evaluation
    n_test_inputs = 1024
    test_indices = np.random.choice(1 << N, n_test_inputs, replace=False)
    test_inputs = _get_all_inputs()[test_indices]
    
    while _time_left() > 1.0:
        # Generate a random move
        move_type = random.choice(['remove', 'swap', 'replace'])
        
        new_comps = current_comps[:]
        
        if move_type == 'remove' and len(new_comps) > 0:
            idx = random.randint(0, len(new_comps) - 1)
            new_comps.pop(idx)
        elif move_type == 'swap' and len(new_comps) > 1:
            i, j = random.sample(range(len(new_comps)), 2)
            new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
        elif move_type == 'replace' and len(new_comps) > 0:
            idx = random.randint(0, len(new_comps) - 1)
            new_comps[idx] = random.choice(all_pairs)
        else:
            continue
        
        # Fast cost evaluation using test inputs
        final_test = simulate_network(new_comps, test_inputs)
        cost = count_inversions(final_test)
        
        # Acceptance probability
        if cost < best_score:
            current_comps = new_comps
            best_comps = new_comps[:]
            best_score = cost
        else:
            # Accept with probability exp(-delta/T)
            delta = cost - best_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_comps = new_comps
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final verification and optimization
    if is_sorted_network(best_comps):
        # Try to remove more comparators
        improved = True
        while improved and _time_left() > 0.5:
            improved = False
            indices = list(range(len(best_comps)))
            random.shuffle(indices)
            for idx in indices:
                if _time_left() < 0.5:
                    break
                new_comps = best_comps[:idx] + best_comps[idx + 1:]
                if is_sorted_network(new_comps):
                    best_comps = new_comps
                    improved = True
                    break
    
    return best_comps

if __name__ == "__main__":
    result = construct()
    print(f"Comparators: {len(result)}")
