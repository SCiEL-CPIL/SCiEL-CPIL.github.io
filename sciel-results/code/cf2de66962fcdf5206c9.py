"""Improved sorting network search for 17 wires using efficient inversion-based verification."""
import time
import numpy as np
import random

N = 17
DEADLINE = 85.0  # Safety margin before 90s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _make_all_inputs():
    """All 2^N boolean inputs as uint8 array shape (2^N, N)."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)

_ALL_INPUTS = None
_START_TIME = time.time()

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def count_inversions(data):
    """Count total inversions across all rows. data shape (M, N)."""
    # For each row, count pairs (i,j) with i<j and data[i] > data[j]
    # Since data is binary, inversion means data[i]=1, data[j]=0 for i<j
    # Use prefix sums: for each position j, count of 1s before j
    # inversions = sum over j of (data[:,j] == 0) * (sum of data[:, 0:j])
    # More efficiently: for each row, inversions = sum_{j} (1-data[:,j]) * prefix_sum_ones[:,j]
    # where prefix_sum_ones[:,j] = sum_{i<j} data[:,i]
    
    # Actually, let's use a simpler approach with cumulative sum
    # For binary data, inversions in a row = number of (1,0) pairs with 1 before 0
    # = sum over each 0 of the number of 1s before it
    
    # Efficient vectorized approach:
    # Compute cumulative sum of 1s up to each position (exclusive)
    cumsum_ones = np.cumsum(data, axis=1)
    # For position j, the number of 1s before j is cumsum_ones[:, j] - data[:, j]
    # Inversions contributed by position j = (1 - data[:, j]) * (cumsum_ones[:, j] - data[:, j])
    # But we need to be careful with the last column
    
    # Let's compute: for each j from 0 to N-1, inversions_j = (1-data[:,j]) * (number of 1s in positions 0..j-1)
    # number of 1s in positions 0..j-1 = cumsum_ones[:, j] - data[:, j] if j < N, but cumsum_ones includes position j
    # So: ones_before_j = cumsum_ones[:, j] - data[:, j]
    # But for j=0, ones_before_0 = 0, and cumsum_ones[:,0] - data[:,0] = 0. Correct.
    
    ones_before = cumsum_ones - data  # This gives ones in positions 0..j-1 for each j
    # Wait, cumsum_ones[:, j] = sum_{i=0}^{j} data[:, i]
    # So cumsum_ones[:, j] - data[:, j] = sum_{i=0}^{j-1} data[:, i] = ones before position j
    # Yes, that's correct.
    
    inv_per_pos = (1 - data).astype(np.int32) * ones_before.astype(np.int32)
    return int(np.sum(inv_per_pos))

def simulate_network(comps, initial_data=None):
    """Simulate network on all boolean inputs. Returns final uint8 array (2^N, N)."""
    if initial_data is None:
        data = _get_all_inputs().copy()
    else:
        data = initial_data.copy()
    for a, b in comps:
        lo = np.minimum(data[:, a], data[:, b])
        hi = np.maximum(data[:, a], data[:, b])
        data[:, a] = lo
        data[:, b] = hi
    return data

def is_sorted_network(comps):
    """Check if network sorts all boolean inputs (zero-one principle)."""
    final = simulate_network(comps)
    # Check if each row is sorted (non-decreasing)
    # For binary, sorted means all 0s come before all 1s
    # So final[:, i] <= final[:, i+1] for all i
    return bool(np.all(final[:, :-1] <= final[:, 1:]))

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
    """Construct a sorting network using local search on OETN."""
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with OETN
    comps = generate_oetn()
    
    # Pre-generate all possible pairs for replacement moves
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # First, do a quick verification
    if not is_sorted_network(comps):
        # Should not happen, but just in case
        pass
    
    # Local Search Loop
    # Priority: removals first (most impactful), then swaps, then replacements
    
    while _time_left() > 1.0:
        improved = False
        
        # 1. Try removals - iterate through indices in random order
        indices = list(range(len(comps)))
        random.shuffle(indices)
        
        for idx in indices:
            if _time_left() < 0.5:
                break
            # Try removing comps[idx]
            new_comps = comps[:idx] + comps[idx + 1:]
            if is_sorted_network(new_comps):
                comps = new_comps
                improved = True
                break
        
        if improved:
            continue

        # 2. Try swaps
        if len(comps) > 1:
            for _ in range(20):
                if _time_left() < 0.5:
                    break
                i, j = random.sample(range(len(comps)), 2)
                comps[i], comps[j] = comps[j], comps[i]
                if is_sorted_network(comps):
                    improved = True
                    break
                else:
                    comps[i], comps[j] = comps[j], comps[i]
            
            if improved:
                continue

        # 3. Try replacements
        if len(comps) > 0:
            for _ in range(20):
                if _time_left() < 0.5:
                    break
                idx = random.randint(0, len(comps) - 1)
                new_comp = random.choice(all_pairs)
                if comps[idx] == new_comp:
                    continue
                old_comp = comps[idx]
                comps[idx] = new_comp
                if is_sorted_network(comps):
                    improved = True
                    break
                else:
                    comps[idx] = old_comp
            
            if improved:
                continue
        
        # If no improvement, continue with new random shuffles

    return comps

if __name__ == "__main__":
    result = construct()
    print(f"Comparators: {len(result)}")
