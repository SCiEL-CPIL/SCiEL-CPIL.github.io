"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
SEED_DEADLINE = 45.0  # safety margin before 60s kill

# Pre-compute the input matrix for verification to avoid regeneration overhead
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    """
    # Copy the input matrix
    x = _INPUT_MATRIX.copy()
    
    # Apply network
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    
    # Verify sorted
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _count_comparators(network):
    return len(network)

def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    
    n = len(network)
    wire_to_last_comp = {}  # wire -> last comparator index that touched it
    depths = [0] * n
    
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    
    return max(depths) if depths else 0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires.
    
    Uses Batcher's odd-even merge sort as a starting point,
    then aggressively prunes redundant comparators via greedy removal
    and randomized local search with swaps.
    """
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE - 3.0  # safety margin
    
    # --- Generate Batcher's sort network ---
    def _batcher_sort_network(n):
        """Return Batcher's sort network for n wires (0-indexed)."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n == 3:
            return [(0, 1), (1, 2), (0, 1)]
        if n == 4:
            return [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
        
        half = n // 2
        net = []
        net.extend([(i, j) for i, j in _batcher_sort_network(half)])
        net.extend([(i + half, j + half) for i, j in _batcher_sort_network(n - half)])
        
        def _merge_strided(k, start_even, start_odd, step):
            if k == 1:
                return [(start_even, start_odd)]
            net2 = []
            for i in range(k):
                net2.append((start_even + i * step, start_odd + i * step))
            net2.extend(_merge_strided(k // 2, start_even, start_odd, 2 * step))
            net2.extend(_merge_strided(k // 2, start_even + step, start_odd + step, 2 * step))
            return net2
        
        net.extend(_merge_strided(half, 0, half, 1))
        return net
    
    net = _batcher_sort_network(N)
    
    # Verify it's correct
    if not _verify_network_fastest(net):
        # Fallback: odd-even transposition (120 comparators)
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_network = list(net)
    best_count = len(best_network)
    
    # --- Greedy removal: try to remove each comparator and check if still valid ---
    # Process from the end backwards (later comparators are more likely removable)
    changed = True
    while changed:
        changed = False
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best_network[:idx] + best_network[idx + 1:]
            if _verify_network_fastest(candidate):
                best_network = candidate
                changed = True
        if time.time() > deadline:
            break
    
    # --- Randomized local search: swap comparator orderings to enable more removals ---
    for _iter in range(20000):
        if time.time() > deadline:
            break
        idx1 = random.randint(0, len(best_network) - 1)
        idx2 = random.randint(0, len(best_network) - 1)
        if idx1 == idx2:
            continue
        candidate = list(best_network)
        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
        if _verify_network_fastest(candidate):
            # Try to remove a comparator after the swap
            removed = False
            for r in range(len(candidate) - 1, -1, -1):
                if time.time() > deadline:
                    break
                test = candidate[:r] + candidate[r + 1:]
                if _verify_network_fastest(test):
                    candidate = test
                    removed = True
                    break
            if removed:
                best_network = candidate
    
    return best_network
# EVOLVE-BLOCK-END
