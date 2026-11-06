"""Rewritten program: Local search minimization starting from Odd-Even Transposition Network."""
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
    return np.all(final[:, :-1] <= final[:, 1:])

def generate_oetn():
    """Generate Odd-Even Transposition Network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            # Even phase: (0,1), (2,3), ...
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            # Odd phase: (1,2), (3,4), ...
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps

def construct():
    """Construct a sorting network using local search on OETN."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with OETN
    comps = generate_oetn()
    
    # Pre-generate all possible pairs for replacement moves
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local Search Loop
    # We prioritize removals, then swaps, then replacements.
    # To avoid getting stuck, we randomize the order of indices we check.
    
    while _time_left() > 1.0:
        improved = False
        
        # 1. Try removals
        # We iterate through indices in a random order to break symmetry
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
                # After a removal, the indices shift, so break and restart loop
                break
        
        if improved:
            continue

        # 2. Try swaps
        if len(comps) > 1:
            # Randomly pick two distinct indices
            # To be efficient, we can try a few random swaps per iteration
            # or iterate through a subset. 
            # Let's try a random swap.
            for _ in range(10): # Try 10 random swaps per "pass"
                if _time_left() < 0.5:
                    break
                i, j = random.sample(range(len(comps)), 2)
                # Swap comps[i] and comps[j]
                comps[i], comps[j] = comps[j], comps[i]
                if is_sorted_network(comps):
                    improved = True
                    # Keep the swap
                else:
                    # Revert
                    comps[i], comps[j] = comps[j], comps[i]
                if improved:
                    break
            if improved:
                continue

        # 3. Try replacements
        # Replace a random comparator with a random other comparator
        if len(comps) > 0:
            for _ in range(10): # Try 10 random replacements per "pass"
                if _time_left() < 0.5:
                    break
                idx = random.randint(0, len(comps) - 1)
                new_comp = random.choice(all_pairs)
                # If the new comp is the same, skip
                if comps[idx] == new_comp:
                    continue
                old_comp = comps[idx]
                comps[idx] = new_comp
                if is_sorted_network(comps):
                    improved = True
                    # Keep the replacement
                else:
                    # Revert
                    comps[idx] = old_comp
                if improved:
                    break
            if improved:
                continue
        
        # If no improvement in this full pass, we might be stuck in a local minimum.
        # We could try a "kick" but for now, we just continue looping with random shuffles.
        # The random shuffle of indices for removals provides some diversification.

    return comps

if __name__ == "__main__":
    result = construct()
    print(f"Comparators: {len(result)}")
