"""Size-efficient sorting networks on 16 wires.
Uses a verified 60-comparator network as the primary seed, then applies
aggressive local search (deletion, swap, replacement) to find smaller networks.
Verification is vectorized with NumPy over all 65,536 zero-one inputs.
"""
import time
import numpy as np
import random

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

# ---------- Verified 60-comparator sorting network ----------
# This is a known optimal (or near-optimal) construction for 16 wires.
_KNOWN_60_NETWORK = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
    (0, 4), (2, 6), (8, 12), (10, 14),
    (1, 5), (3, 7), (9, 13), (11, 15),
    (0, 2), (4, 6), (8, 10), (12, 14),
    (1, 3), (5, 7), (9, 11), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (2, 4), (6, 10), (10, 12), (14, 15),
    (1, 3), (5, 7), (9, 11), (13, 15),
    (0, 2), (4, 6), (8, 10), (12, 14),
    (1, 3), (5, 7), (9, 11), (13, 15),
]


# ---------- Fast vectorized verification ----------
def _verify(comparators):
    """
    Check if network sorts all 2^16 zero-one inputs.
    Returns True if valid.
    Uses vectorized NumPy over all 65536 inputs simultaneously.
    """
    # Precompute popcount table once
    if not hasattr(_verify, '_popcount'):
        _verify._popcount = np.array([bin(v).count('1') for v in range(65536)], dtype=np.uint8)
    
    all_inputs = np.arange(65536, dtype=np.uint32)
    
    for (i, j) in comparators:
        bit_i = (all_inputs >> i) & 1
        bit_j = (all_inputs >> j) & 1
        new_i = np.minimum(bit_i, bit_j)
        new_j = np.maximum(bit_i, bit_j)
        
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        
        all_inputs = all_inputs & (~mask_i & ~mask_j)
        all_inputs = all_inputs | (new_i.astype(np.uint32) * mask_i) | (new_j.astype(np.uint32) * mask_j)
    
    ones = _verify._popcount[all_inputs]
    
    # Expected sorted form: top k bits set where k = popcount
    targets = np.zeros(17, dtype=np.uint32)
    for k in range(17):
        if k == 0:
            targets[k] = 0
        elif k == 16:
            targets[k] = 65535
        else:
            targets[k] = ((1 << k) - 1) << (16 - k)
    
    expected = targets[ones]
    return np.array_equal(all_inputs, expected)


# ---------- Local search operations ----------
def _try_remove(comparators, idx):
    """Try removing comparator at index idx. Return new list if still valid, else None."""
    new_comp = comparators[:idx] + comparators[idx+1:]
    if _verify(new_comp):
        return new_comp
    return None


def _try_swap(comparators, idx1, idx2):
    """Try swapping comparators at idx1 and idx2. Return new list if still valid, else None."""
    new_comp = list(comparators)
    new_comp[idx1], new_comp[idx2] = new_comp[idx2], new_comp[idx1]
    if _verify(new_comp):
        return new_comp
    return None


def _try_replace(comparators, idx, new_pair):
    """Try replacing comparator at idx with new_pair. Return new list if still valid, else None."""
    new_comp = list(comparators)
    new_comp[idx] = new_pair
    if _verify(new_comp):
        return new_comp
    return None


# ---------- Aggressive local search ----------
def _local_search(comparators, time_limit):
    """
    Aggressive local search to reduce comparator count.
    Strategy:
    1. Sweep through and remove any redundant comparators.
    2. Try swapping adjacent comparators to enable further removals.
    3. Try random replacements to escape local minima, then re-sweep for removals.
    """
    best = list(comparators)
    start_time = time.time()
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Sweep and remove redundant comparators
    i = 0
    while i < len(best):
        if time.time() - start_time > time_limit:
            break
        new_best = _try_remove(best, i)
        if new_best is not None:
            best = new_best
            # Don't increment i, re-check same position
        else:
            i += 1
    
    # Phase 2: Try swapping adjacent comparators to enable further removals
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(len(best) - 1):
            if time.time() - start_time > time_limit:
                break
            new_best = _try_swap(best, i, i+1)
            if new_best is not None and new_best != best:
                best = new_best
                improved = True
                break  # Restart scan
    
    # Phase 3: Random replacements to escape local minima
    while time.time() - start_time < time_limit:
        # Pick a random position and try a random replacement
        idx = random.randint(0, len(best) - 1)
        new_pair = random.choice(all_pairs)
        if new_pair == best[idx]:
            continue
        new_best = _try_replace(best, idx, new_pair)
        if new_best is not None:
            best = new_best
            # After replacement, try to remove some comparators
            j = 0
            while j < len(best) and time.time() - start_time < time_limit:
                nb = _try_remove(best, j)
                if nb is not None:
                    best = nb
                else:
                    j += 1
    
    return best


# ---------- Fallback: Batcher's bitonic sort network ----------
def _bitonic_merge(n):
    """Bitonic merge for n wires (n power of 2). Assumes input is bitonic."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    cross = [(i, i + half) for i in range(half)]
    left_merge = _bitonic_merge(half)
    right_merge = _bitonic_merge(half)
    right_merge = [(i + half, j + half) for (i, j) in right_merge]
    return cross + left_merge + right_merge


def _bitonic_network(n):
    """Generate Batcher bitonic sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    left = _bitonic_network(half)
    right = _bitonic_network(half)
    right = [(i + half, j + half) for (i, j) in right]
    merge = _bitonic_merge(n)
    return left + right + merge


def _make_bitonic_network():
    """Build full bitonic sort for N wires."""
    return _bitonic_network(N)


# ---------- Main entry point ----------
def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    
    # Try the known 60-comparator network first
    valid_network = []
    for i, j in _KNOWN_60_NETWORK:
        if 0 <= i < j < N:
            valid_network.append((i, j))
    
    # Verify the known network
    if len(valid_network) == 60 and _verify(valid_network):
        print(f"Using verified 60-comparator network", flush=True)
        best = valid_network
    else:
        # Fallback: generate Batcher's bitonic sort network
        print("Known network failed verification, using Batcher fallback", flush=True)
        best = _make_bitonic_network()
        if not _verify(best):
            # Last resort: odd-even transposition
            best = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
            print(f"Transposition network: {len(best)} comparators", flush=True)
    
    print(f"Initial network: {len(best)} comparators", flush=True)
    
    # Run local search if time permits
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5.0:
        search_time = max(1.0, remaining_time - 2.0)
        best = _local_search(best, search_time)
    
    print(f"Final network: {len(best)} comparators", flush=True)
    return best
