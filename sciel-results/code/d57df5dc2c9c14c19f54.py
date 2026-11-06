"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N  # 262144

def _generate_all_inputs():
    """Generate all 2^N Boolean inputs as a (2^N, N) uint8 array."""
    arr = np.arange(NUM_INPUTS, dtype=np.uint32)
    bits = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = ((arr >> (N - 1 - i)) & 1).astype(np.uint8)
    return bits

def _verify(comparators, all_inputs):
    """Verify network sorts all Boolean inputs. Returns True if valid."""
    state = all_inputs.copy()
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # Check sorted: each row non-decreasing
    for i in range(N - 1):
        if not np.all(state[:, i] <= state[:, i + 1]):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 84.0  # safety margin
    
    # Build odd-even transposition network (153 comparators)
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    
    all_inputs = _generate_all_inputs()
    
    # Verify initial
    assert _verify(comparators, all_inputs), "Initial network should be valid"
    
    best_comparators = list(comparators)
    best_count = len(comparators)
    
    # Phase 1: Aggressive greedy removal
    # Multiple passes, trying to remove each comparator
    for p in range(100):
        if time.time() - start_time > DEADLINE:
            break
        removed = False
        # Try removing from end to start (order matters less)
        for idx in range(len(comparators) - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                break
            new_comps = comparators[:idx] + comparators[idx + 1:]
            if _verify(new_comps, all_inputs):
                comparators = new_comps
                removed = True
                if len(comparators) < best_count:
                    best_count = len(comparators)
                    best_comparators = list(comparators)
        if not removed:
            break
    
    # Phase 2: Swap-based local search
    if time.time() - start_time < DEADLINE - 3.0:
        current = list(comparators)
        rng = np.random.default_rng(42)
        stall = 0
        max_iter = 500
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
            n_c = len(current)
            if n_c < 2:
                break
            # Try a random swap
            i1 = rng.integers(0, n_c)
            i2 = rng.integers(0, n_c)
            if i1 == i2:
                continue
            new_comps = list(current)
            new_comps[i1], new_comps[i2] = new_comps[i2], new_comps[i1]
            
            if _verify(new_comps, all_inputs):
                current = new_comps
                # After swap, try to remove
                for idx in range(len(current) - 1, -1, -1):
                    if time.time() - start_time > DEADLINE:
                        break
                    nc = current[:idx] + current[idx + 1:]
                    if _verify(nc, all_inputs):
                        current = nc
                        if len(current) < best_count:
                            best_count = len(current)
                            best_comparators = list(current)
                stall = 0
            else:
                stall += 1
                if stall > 200:
                    break
    
    return best_comparators
# EVOLVE-BLOCK-END
