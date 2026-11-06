"""Size-efficient sorting networks on 16 wires.
Uses Batcher bitonic sort (63 comparators) as seed, then aggressive local search
to reduce comparator count toward the known optimum of 60.
"""
import numpy as np
import time
import random
from collections import deque

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

# ---------- Batcher Bitonic Sort Network ----------
def _bitonic_network(n):
    """Generate Batcher bitonic sort network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    half = n // 2
    # Sort first half ascending, second half descending, then merge
    left = _bitonic_network(half)
    right = _bitonic_network(half)
    # Shift right comparators
    right = [(i + half, j + half) for (i, j) in right]
    # Merge: odd-even merge for bitonic
    merge = _bitonic_merge(n)
    return left + right + merge

def _bitonic_merge(n):
    """Bitonic merge for n wires (n power of 2). Assumes input is bitonic."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # First do cross comparisons
    cross = [(i, i + half) for i in range(half)]
    # Then recursively merge both halves
    left_merge = _bitonic_merge(half)
    right_merge = _bitonic_merge(half)
    right_merge = [(i + half, j + half) for (i, j) in right_merge]
    return cross + left_merge + right_merge

def _make_bitonic_network():
    """Build full bitonic sort: split, sort halves opposite, merge."""
    half = N // 2
    left = _bitonic_network(half)
    right = _bitonic_network(half)
    right = [(i + half, j + half) for (i, j) in right]
    merge = _bitonic_merge(N)
    return left + right + merge

# ---------- Verification ----------
def _verify(comparators):
    """Check if network sorts all 2^16 zero-one inputs. Returns True if valid."""
    n_comp = len(comparators)
    # Build permutation tracking: for each wire, track which original position it holds
    # Use the zero-one principle: check all 2^16 inputs
    # Fast method: represent state as 16-bit integer, apply comparators
    # Actually, for 16 wires we can use the "16-bit mask" approach
    # Each input is a 16-bit number. After sorting, it must be of form (2^k - 1) << (16-k)
    
    # More efficient: track the set of positions of 1s
    # For a given mask, after all comparators, the 1s must be in the top positions
    # i.e., the result mask must be of form ((1 << n_ones) - 1) << (16 - n_ones)
    
    # We'll use a vectorized approach with numpy
    # Represent all 65536 states as a column of 16-bit values
    all_inputs = np.arange(65536, dtype=np.uint32)
    
    for (i, j) in comparators:
        # For each state, extract bits i and j
        bit_i = (all_inputs >> i) & 1
        bit_j = (all_inputs >> j) & 1
        # min goes to i, max goes to j
        new_i = np.minimum(bit_i, bit_j)
        new_j = np.maximum(bit_i, bit_j)
        # Update: clear old bits, set new
        mask_i = np.uint32(1 << i)
        mask_j = np.uint32(1 << j)
        all_inputs = (all_inputs & ~mask_i & ~mask_j) | (new_i.astype(np.uint32) * mask_i) | (new_j.astype(np.uint32) * mask_j)
    
    # Check: for each input, count ones, then verify the sorted form
    # Count ones in each 16-bit value
    # Use lookup or bit counting
    # For 65536 values, we can precompute popcount
    # Actually let's just check: the sorted result for k ones is ((1<<k)-1) << (16-k)
    # We need to verify that all_inputs[k] == target[k] for each k
    
    # Compute popcount for each of the 65536 values
    # Use a simple loop or numpy trick
    # np.unpackbits approach:
    # Actually, let's just verify by checking that the value equals the expected sorted form
    
    # Compute number of ones in each value
    # For efficiency, use a precomputed table
    popcount_table = np.zeros(65536, dtype=np.uint8)
    for v in range(65536):
        popcount_table[v] = bin(v).count('1')
    
    ones = popcount_table[all_inputs.astype(np.uint32)]
    # Expected sorted form: ((1<<k)-1) << (16-k)
    expected = np.zeros(65536, dtype=np.uint32)
    for k in range(17):
        mask = (ones == k)
        if np.any(mask):
            val = ((1 << k) - 1) << (16 - k) if k < 16 else 0
            # Wait, if k=16, all ones, value = 0xFFFF = 65535
            # if k=0, value = 0
            if k == 0:
                val = 0
            elif k == 16:
                val = 65535
            else:
                val = ((1 << k) - 1) << (16 - k)
            expected[mask] = val
    
    return np.array_equal(all_inputs.astype(np.uint32), expected.astype(np.uint32))

def _verify_fast(comparators):
    """Faster verification using the property that we only need to check
    inputs that are 'critical'. Actually for 16 wires, 65536 checks is fast enough
    with numpy. Let's just use _verify."""
    return _verify(comparators)

# ---------- Local Search ----------
def _try_remove(comparators, idx):
    """Try removing comparator at index idx. Return new list if still valid, else None."""
    new_comp = comparators[:idx] + comparators[idx+1:]
    if _verify(new_comp):
        return new_comp
    return None

def _try_swap(comparators, idx1, idx2):
    """Try swapping comparators at idx1 and idx2."""
    new_comp = list(comparators)
    new_comp[idx1], new_comp[idx2] = new_comp[idx2], new_comp[idx1]
    if _verify(new_comp):
        return new_comp
    return None

def _try_replace(comparators, idx, new_comp):
    """Try replacing comparator at idx with new_comp."""
    new_comp_list = list(comparators)
    new_comp_list[idx] = new_comp
    if _verify(new_comp_list):
        return new_comp_list
    return None

def _local_search(comparators, time_limit):
    """Aggressive local search to reduce comparator count."""
    best = list(comparators)
    start_time = time.time()
    
    # All possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # First pass: try to remove each comparator
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
    
    # Second pass: try swapping adjacent comparators to enable further removals
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
    
    # Third pass: try replacing comparators with different ones to enable removals
    # This is more expensive, so do it selectively
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
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
            while j < len(best):
                if time.time() - start_time > time_limit:
                    break
                nb = _try_remove(best, j)
                if nb is not None:
                    best = nb
                else:
                    j += 1
    
    return best

def _anneal(comparators, time_limit):
    """Simulated annealing-style search."""
    best = list(comparators)
    current = list(comparators)
    start_time = time.time()
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        temperature *= cooling
        if temperature < 0.01:
            break
        
        # Random move: add, remove, swap, or replace
        move_type = random.choice(['remove', 'swap', 'replace', 'add'])
        
        new_comp = list(current)
        valid = False
        
        if move_type == 'remove' and len(new_comp) > 0:
            idx = random.randint(0, len(new_comp) - 1)
            new_comp.pop(idx)
            valid = _verify(new_comp)
        
        elif move_type == 'swap' and len(new_comp) >= 2:
            i, j = random.sample(range(len(new_comp)), 2)
            new_comp[i], new_comp[j] = new_comp[j], new_comp[i]
            valid = _verify(new_comp)
        
        elif move_type == 'replace' and len(new_comp) > 0:
            idx = random.randint(0, len(new_comp) - 1)
            new_comp[idx] = random.choice(all_pairs)
            valid = _verify(new_comp)
        
        elif move_type == 'add' and len(new_comp) < 100:
            new_comp.append(random.choice(all_pairs))
            valid = _verify(new_comp)
        
        if not valid:
            continue
        
        # Accept if better or with probability exp(-delta/T)
        delta = len(new_comp) - len(current)
        if delta < 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
            current = new_comp
            if len(current) < len(best):
                best = list(current)
    
    return best

# ---------- Main construct function ----------
def construct():
    start_time = time.time()
    
    # Start with Batcher bitonic sort network (63 comparators)
    network = _make_bitonic_network()
    
    # Verify it's correct
    if not _verify(network):
        # Fallback to odd-even transposition
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Phase 1: Greedy removal
    deadline1 = start_time + 15
    network = _local_search(network, deadline1 - time.time())
    
    # Phase 2: More aggressive search with restarts
    deadline2 = start_time + 70
    best_network = network
    best_len = len(network)
    
    while time.time() < deadline2:
        # Restart from best with random perturbation
        perturbed = list(best_network)
        # Remove a random comparator
        if len(perturbed) > 0:
            idx = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
        
        # Local search from perturbed state
        remaining_time = deadline2 - time.time() - 5  # 5s safety margin
        if remaining_time < 1:
            break
        
        candidate = _local_search(perturbed, remaining_time)
        if len(candidate) < best_len:
            best_network = candidate
            best_len = len(candidate)
    
    # Final: try one more round of removals on the best
    final_deadline = start_time + 82
    if time.time() < final_deadline:
        best_network = _local_search(best_network, final_deadline - time.time())
    
    # Verify final result
    if not _verify(best_network):
        # Should not happen, but fallback
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return best_network
