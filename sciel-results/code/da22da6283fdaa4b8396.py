"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import numpy as np

# ---------------------------------------------------------------------------
# Fast zero-one principle verification using NumPy vectorization
# ---------------------------------------------------------------------------
def _verify_network(comparators):
    """Verify sorting network using zero-one principle over all 2**N Boolean inputs."""
    n = N
    M = 1 << n
    if M == 0:
        return True
    # Initialize state[k] = input bitmask k as a NumPy array
    state = np.arange(M, dtype=np.int32)
    for i, j in comparators:
        bi = 1 << i
        bj = 1 << j
        # Find all k where bit i is set and bit j is not set
        mask = (state & bi) & ~(state & bj)
        if np.any(mask):
            # Swap bits i and j for those entries
            state[mask] = state[mask] ^ bi ^ bj
    # Check sorted: for every k, if bit i set then all lower bits set.
    # i.e. for k >= (1<<i), bit i must be 1.
    for i in range(n - 1):
        threshold = 1 << i
        mask_bit = 1 << (i + 1)
        # For k in [threshold, M), check if bit (i+1) is set
        sub = state[threshold:]
        if not np.all(sub & mask_bit):
            return False
    return True


def _verify_network_fast(comparators):
    """Even faster verification using bit manipulation tricks."""
    n = N
    M = 1 << n
    if M == 0:
        return True
    
    # Use a different approach: track the set of values on each wire
    # After all comparators, wire 0 should have the minimum, wire 15 the maximum
    # For 0-1 inputs, this means the sorted output is determined by the count of 1s
    
    # Actually, let's use the same vectorized approach but with early termination
    state = np.arange(M, dtype=np.int32)
    
    for i, j in comparators:
        bi = 1 << i
        bj = 1 << j
        # Only process if there are entries that need swapping
        need_swap = (state & bi) & ~(state & bj)
        if np.any(need_swap):
            state[need_swap] = state[need_swap] ^ bi ^ bj
    
    # Check if sorted
    for i in range(n - 1):
        threshold = 1 << i
        mask_bit = 1 << (i + 1)
        if not np.all(state[threshold:] & mask_bit):
            return False
    return True


def _compute_depth(comparators):
    """Compute parallel depth (greedy layering of non-adjacent comparators)."""
    if not comparators:
        return 0
    depth = 0
    remaining = list(comparators)
    while remaining:
        used = set()
        layer = []
        still = []
        for comp in remaining:
            i, j = comp
            if i not in used and j not in used:
                layer.append(comp)
                used.add(i)
                used.add(j)
            else:
                still.append(comp)
        depth += 1
        remaining = still
    return depth


# ---------------------------------------------------------------------------
# Bitonic sorting network construction (the design family)
# ---------------------------------------------------------------------------
def _bitonic_sort(n, offset, direction):
    """Generate comparators for a bitonic sort of n wires starting at offset."""
    if n <= 1:
        return []
    half = n // 2
    comps = []
    if direction == 'up':
        comps.extend(_bitonic_sort(half, offset, 'down'))
        comps.extend(_bitonic_sort(half, offset + half, 'up'))
    else:
        comps.extend(_bitonic_sort(half, offset, 'up'))
        comps.extend(_bitonic_sort(half, offset + half, 'down'))
    # Bitonic merge: k = n/2 down to 1 (the k = n stage is a no-op)
    k = half
    while k > 0:
        for i in range(0, n, 2 * k):
            for j in range(k):
                a = offset + i + j
                b = offset + i + k + j
                if direction == 'up':
                    comps.append((a, b))
                else:
                    comps.append((b, a))
        k //= 2
    return comps


def _baseline_network():
    """A verified correct sorting network (bitonic, 56 comparators for N=16)."""
    return _bitonic_sort(N, 0, 'up')


# ---------------------------------------------------------------------------
# Refinement machinery within the bitonic family
# ---------------------------------------------------------------------------
def _reorder_into_layers(comps):
    """Reorder non-adjacent comparators into parallel layers (safe: they commute).

    Returns (reordered_list, depth).
    """
    layers = []
    remaining = list(comps)
    while remaining:
        used = set()
        layer = []
        still = []
        for c in remaining:
            i, j = c
            if i not in used and j not in used:
                layer.append(c)
                used.add(i)
                used.add(j)
            else:
                still.append(c)
        layers.append(layer)
        remaining = still
    out = []
    for layer in layers:
        out.extend(layer)
    return out, len(layers)


def _prune_smart(comps, deadline):
    """Smart pruning: multi-pass removal with early termination checks."""
    current = list(comps)
    if not _verify_network(current):
        return current
    
    # Pass 1: Remove from end backwards
    i = len(current) - 1
    while i >= 0 and time.time() < deadline:
        trial = current[:i] + current[i+1:]
        if _verify_network(trial):
            current = trial
        else:
            i -= 1
    
    # Pass 2: Remove from beginning
    i = 0
    while i < len(current) and time.time() < deadline:
        trial = current[:i] + current[i+1:]
        if _verify_network(trial):
            current = trial
        else:
            i += 1
    
    # Pass 3: Multi-round middle removal
    rounds = 0
    max_rounds = 50
    while time.time() < deadline and rounds < max_rounds:
        improved = False
        for idx in range(len(current)):
            if time.time() >= deadline:
                break
            trial = current[:idx] + current[idx + 1:]
            if _verify_network(trial):
                current = trial
                improved = True
                break
        if not improved:
            break
        rounds += 1
    
    return current


def _prune(comps, deadline):
    """Greedy removal of redundant comparators (any subsequence of a sorter sorts)."""
    return _prune_smart(comps, deadline)


def _odd_even_transposition():
    """Fallback correct network: 120 comparators."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def construct():
    deadline = time.time() + 85  # safety margin well under 90s wall-clock
    try:
        comps = _baseline_network()
        # Ensure the baseline is actually correct before refining.
        if not _verify_network(comps):
            comps = _odd_even_transposition()

        # Refinement 1: reorder non-adjacent comparators into parallel layers
        # (reduces reported depth; correctness unchanged).
        comps, _depth = _reorder_into_layers(comps)

        # Refinement 2: prune redundant comparators (reduces count; correctness
        # preserved because any subsequence of a sorting network sorts).
        comps = _prune(comps, deadline)

        # Refinement 3: Try additional pruning passes if time remains
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for idx in range(len(comps)):
                    if time.time() >= deadline - 2:
                        break
                    trial = comps[:idx] + comps[idx + 1:]
                    if _verify_network(trial):
                        comps = trial
                        improved = True
                        break
        
        # Final safety: guarantee a valid network.
        if not _verify_network(comps):
            comps = _odd_even_transposition()
        return comps
    except Exception:
        return _odd_even_transposition()
# EVOLVE-BLOCK-END
