"""
Improved local search approach for sorting networks on 17 wires.

Structural change: Abandon the genetic algorithm and layer-based representation.
Use a flat list representation with aggressive local search operators:
1. Greedy deletion (remove valid comparators).
2. Swap-based perturbation (swap comparator positions to unlock new deletion paths).
3. Random restarts with insertion (add a random comparator, then delete greedily).

This approach is more deterministic and focused on minimizing the comparator count
directly via local moves that are verified by fast sampling followed by full check.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute the input matrix and expected sorted outputs at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute sample sets for fast validation
_SAMPLE_SIZE_LARGE = 16384
_SAMPLE_SIZE_SMALL = 4096
_rng_mod = np.random.RandomState(42)
_SAMPLE_INDICES_LARGE = _rng_mod.choice(_NUM_INPUTS, size=_SAMPLE_SIZE_LARGE, replace=False)
_SAMPLE_INPUTS_LARGE = _INPUTS[_SAMPLE_INDICES_LARGE]
_SAMPLE_SORTED_LARGE = _SORTED_INPUTS[_SAMPLE_INDICES_LARGE]
_SAMPLE_INDICES_SMALL = _rng_mod.choice(_NUM_INPUTS, size=_SAMPLE_SIZE_SMALL, replace=False)
_SAMPLE_INPUTS_SMALL = _INPUTS[_SAMPLE_INDICES_SMALL]
_SAMPLE_SORTED_SMALL = _SORTED_INPUTS[_SAMPLE_INDICES_SMALL]


def _simulate_on_inputs(comparators, inputs, sorted_inputs):
    """Simulate network on given inputs, return number of mismatches."""
    if len(comparators) > 4096:
        return len(inputs) * N
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return int(np.sum(state != sorted_inputs))


def is_sorted_network_full(comparators):
    """Check if the network sorts all 2^17 inputs using the zero-one principle."""
    return _simulate_on_inputs(comparators, _INPUTS, _SORTED_INPUTS) == 0


def is_sorted_network_sample(comparators):
    """Fast check on a sample of inputs."""
    return _simulate_on_inputs(comparators, _SAMPLE_INPUTS_LARGE, _SAMPLE_SORTED_LARGE) == 0


# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)


def _odd_even_transposition(n):
    """Standard odd-even transposition sort network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _odd_even_merge_sort(n):
    """
    Odd-even merge sort network for n wires.
    Returns a valid sorting network.
    """
    def _merge(left_start, left_len, right_start, right_len):
        """Merge two sorted sequences using odd-even merge."""
        if left_len == 0 or right_len == 0:
            return []
        if left_len == 1 and right_len == 1:
            return [(left_start, right_start)]
        
        # Split into odd and even positions
        left_odd_start = left_start
        left_odd_len = (left_len + 1) // 2
        left_even_start = left_start + 1
        left_even_len = left_len // 2
        right_odd_start = right_start
        right_odd_len = (right_len + 1) // 2
        right_even_start = right_start + 1
        right_even_len = right_len // 2
        
        comparators = []
        # Merge odd parts
        comparators.extend(_merge(left_odd_start, left_odd_len, right_odd_start, right_odd_len))
        # Merge even parts
        comparators.extend(_merge(left_even_start, left_even_len, right_even_start, right_even_len))
        
        # Interleave: compare adjacent odd-even pairs
        # The merged odd elements are at positions: left_odd_start, left_odd_start+2, ...
        # The merged even elements are at positions: left_even_start, left_even_start+2, ...
        # We need to compare (pos, pos+1) for the interleaved result.
        # The total length is left_len + right_len.
        # The odd merged stream has length left_odd_len + right_odd_len.
        # The even merged stream has length left_even_len + right_even_len.
        
        # Standard Odd-Even Merge interleaving step:
        # Compare (2i, 2i+1) for i in range(total_len // 2)
        total_len = left_len + right_len
        for i in range(total_len // 2):
            pos = left_start + 2 * i
            # Ensure we don't go out of bounds
            if pos + 1 < left_start + total_len:
                comparators.append((pos, pos + 1))
        
        return comparators
    
    def _sort(seq_start, seq_len):
        """Sort a sequence of length seq_len starting at seq_start."""
        if seq_len <= 1:
            return []
        half = seq_len // 2
        comparators = []
        comparators.extend(_sort(seq_start, half))
        comparators.extend(_sort(seq_start + half, seq_len - half))
        comparators.extend(_merge(seq_start, half, seq_start + half, seq_len - half))
        return comparators
    
    return _sort(0, n)


def _bitonic_sort_padded(n):
    """
    Bitonic sort network for n wires (pads to next power of 2, then prunes).
    """
    if n <= 1:
        return []
    
    p = 1
    while p < n:
        p *= 2
    
    def bitonic_network(k, ascending=True):
        if k == 1:
            return []
        if k == 2:
            return [(0, 1)] if ascending else [(1, 0)]
        
        half = k // 2
        left = bitonic_network(half, ascending)
        right = bitonic_network(half, not ascending)
        right = [(i + half, j + half) if i < j else (j + half, i + half) for i, j in right]
        
        merge = []
        for i in range(half):
            j = i + half
            if ascending:
                merge.append((i, j))
            else:
                merge.append((j, i))
        
        return left + right + merge
    
    full_network = bitonic_network(p, True)
    normalized = []
    for i, j in full_network:
        if i < j:
            normalized.append((i, j))
        else:
            normalized.append((j, i))
    
    filtered = [(i, j) for i, j in normalized if i < n and j < n]
    return filtered


def _construct_initial_network():
    """
    Construct a good initial network.
    """
    candidates = []
    
    oet = _odd_even_transposition(N)
    if is_sorted_network_full(oet):
        candidates.append(oet)
    
    oems = _odd_even_merge_sort(N)
    if is_sorted_network_full(oems):
        candidates.append(oems)
    
    bitonic = _bitonic_sort_padded(N)
    if is_sorted_network_full(bitonic):
        candidates.append(bitonic)
    
    if not candidates:
        return oet
    
    return min(candidates, key=len)


def _greedy_delete(comparators, deadline):
    """
    Greedily delete comparators if the network remains valid on a sample.
    Returns the reduced list.
    """
    current = comparators[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = current[:i] + current[i+1:]
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    changed = True
    return current


def _local_search(comparators, deadline, rng):
    """
    Perform local search: try swaps and moves to find a better network.
    """
    best = comparators[:]
    best_len = len(best)
    
    while time.time() < deadline:
        # Try to improve via greedy deletion
        best = _greedy_delete(best, deadline)
        if len(best) < best_len:
            best_len = len(best)
        
        # If we can't delete, try swapping to unlock deletions
        improved = False
        n = len(best)
        if n < 2:
            break
            
        # Randomly select a few swaps to try
        for _ in range(50):
            if time.time() > deadline:
                break
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            candidate = best[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    # Try to delete from this new state
                    new_candidate = _greedy_delete(candidate, deadline)
                    if len(new_candidate) < best_len:
                        best = new_candidate
                        best_len = len(best)
                        improved = True
                        break
        
        if not improved:
            # If no improvement from swaps, try a random restart with insertion
            if best_len < 4096:
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                candidate = best[:]
                insert_pos = rng.randint(0, len(candidate))
                candidate.insert(insert_pos, (i, j))
                new_candidate = _greedy_delete(candidate, deadline)
                if len(new_candidate) < best_len:
                    best = new_candidate
                    best_len = len(new_candidate)
                else:
                    # If this restart didn't help, and we've been stuck, break?
                    # Or just continue?
                    pass
            else:
                break
                
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    initial = _construct_initial_network()
    current = initial[:]
    best_network = initial[:]
    best_len = len(current)
    
    rng = random.Random(12345)
    
    # Step 1: Aggressive initial deletion
    current = _greedy_delete(current, deadline)
    if len(current) < best_len:
        best_network = current[:]
        best_len = len(current)
    
    # Step 2: Local Search
    if time.time() < deadline - 5.0:
        current = _local_search(current, deadline - 5.0, rng)
        if len(current) < best_len:
            best_network = current[:]
            best_len = len(current)
    
    # Step 3: Final cleanup
    current = _greedy_delete(current, deadline)
    if len(current) < best_len:
        best_network = current[:]
        best_len = len(current)
    
    # Final validation
    if not is_sorted_network_full(best_network):
        if is_sorted_network_full(initial):
            return initial
        return _odd_even_transposition(N)
    
    return best_network
