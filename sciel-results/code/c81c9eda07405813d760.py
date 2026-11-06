"""Sorting network optimization for 17 wires - restructured approach."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None
_SAMPLE_SMALL = None
_SAMPLE_MED = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_small(n=512):
    global _SAMPLE_SMALL
    if _SAMPLE_SMALL is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_inputs), size=n, replace=False)
        _SAMPLE_SMALL = all_inputs[idx]
    return _SAMPLE_SMALL

def _get_sample_med(n=4096):
    global _SAMPLE_MED
    if _SAMPLE_MED is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(all_inputs), size=n, replace=False)
        _SAMPLE_MED = all_inputs[idx]
    return _SAMPLE_MED

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def _check_sorted(output):
    """Check if output rows are sorted (non-decreasing)."""
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def quick_check_small(comparators):
    """Very fast check on 512 random inputs."""
    inputs = _get_sample_small()
    output = simulate_network(comparators, inputs)
    return _check_sorted(output)

def check_medium(comparators):
    """Medium check on 4096 random inputs."""
    inputs = _get_sample_med()
    output = simulate_network(comparators, inputs)
    return _check_sorted(output)

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    return _check_sorted(output)

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Uses the recursive odd-even merge construction.
    Returns list of (i, j) comparator pairs.
    """
    if n == 1:
        return []
    
    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        merged = _odd_even_merge(n)
        return left + right + merged
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comparators = []
        
        if n % 2 == 0:
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        
        return comparators
    
    return _build(n)

def bitonic_sort_network(n):
    """Generate a bitonic sort network for n wires (must be power of 2, pad if needed)."""
    if n == 1:
        return []
    
    def _bitonic(k, offset=0, ascending=True):
        """Bitonic merge for k elements starting at offset."""
        if k <= 1:
            return []
        
        comps = []
        half = k // 2
        
        # Compare elements k/2 apart
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if ascending:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        # Recursively sort each half
        comps.extend(_bitonic(half, offset, ascending))
        comps.extend(_bitonic(half, offset + half, ascending))
        
        return comps
    
    def _sort(k, offset=0, ascending=True):
        """Sort k elements starting at offset."""
        if k <= 1:
            return []
        
        comps = []
        half = k // 2
        
        # Sort first half ascending, second half descending
        comps.extend(_sort(half, offset, True))
        comps.extend(_sort(half, offset + half, False))
        
        # Merge
        comps.extend(_bitonic(k, offset, ascending))
        
        return comps
    
    # Find next power of 2 >= n
    p = 1
    while p < n:
        p *= 2
    
    if p == n:
        return _sort(n)
    else:
        # Pad with extra wires that don't affect the first n wires
        comps = _sort(p)
        # Filter to only comparators involving wires < n
        filtered = [(i, j) for (i, j) in comps if i < n and j < n]
        return filtered

def compute_redundancy_scores(comparators, n_samples=2048):
    """
    For each comparator, estimate how 'redundant' it is by checking
    how often it actually changes the output on random inputs.
    Lower score = more redundant = better candidate for removal.
    """
    inputs = _get_all_inputs()
    idx = np.random.default_rng(99).choice(len(inputs), size=n_samples, replace=False)
    sample = inputs[idx]
    
    n_comps = len(comparators)
    scores = np.zeros(n_comps, dtype=np.float64)
    
    state = sample.copy()
    for ci, (i, j) in enumerate(comparators):
        a = state[:, i]
        b = state[:, j]
        swap_mask = a > b
        scores[ci] = np.sum(swap_mask)
        state[:, i] = np.where(swap_mask, b, a)
        state[:, j] = np.where(swap_mask, a, b)
    
    return scores

def batch_removal_search(comparators, deadline):
    """
    Structural approach: identify groups of comparators that are likely redundant
    and try removing them in batches. Uses redundancy scores to guide which
    comparators to group together.
    """
    best = list(comparators)
    best_len = len(best)
    
    while time.time() < deadline - 3:
        if len(best) <= 63:
            break
        
        # Compute redundancy scores
        scores = compute_redundancy_scores(best, 2048)
        
        # Find the most redundant comparator
        min_idx = np.argmin(scores)
        min_score = scores[min_idx]
        
        # If the least redundant comparator still has low activity, try batch removal
        # Group comparators with similar scores (low activity)
        threshold = max(0.0, min_score * 2)
        candidates = [i for i in range(len(best)) if scores[i] <= threshold]
        
        if len(candidates) < 2:
            # Just try removing the single most redundant
            if quick_check_small(best[:min_idx] + best[min_idx+1:]):
                if check_medium(best[:min_idx] + best[min_idx+1:]):
                    if is_sorted_network(best[:min_idx] + best[min_idx+1:]):
                        best = best[:min_idx] + best[min_idx+1:]
                        best_len = len(best)
                        continue
            break
        
        # Try removing subsets of the candidate group (up to size 3)
        found_improvement = False
        candidates_sorted = sorted(candidates, key=lambda x: scores[x])
        
        # Try removing 1 at a time from the most redundant
        for idx in candidates_sorted[:min(5, len(candidates_sorted))]:
            if time.time() > deadline - 3:
                break
            candidate = best[:idx] + best[idx+1:]
            if quick_check_small(candidate):
                if check_medium(candidate):
                    if is_sorted_network(candidate):
                        best = candidate
                        best_len = len(best)
                        found_improvement = True
                        break
        
        if found_improvement:
            continue
        
        # Try removing pairs of the most redundant
        tried_pairs = set()
        for i in range(min(4, len(candidates_sorted))):
            for j in range(i + 1, min(4, len(candidates_sorted))):
                if time.time() > deadline - 3:
                    break
                pair = frozenset([candidates_sorted[i], candidates_sorted[j]])
                if pair in tried_pairs:
                    continue
                tried_pairs.add(pair)
                
                # Remove both (adjust indices)
                ci, cj = sorted([candidates_sorted[i], candidates_sorted[j]])
                candidate = best[:ci] + best[ci+1:cj] + best[cj+1:]
                if quick_check_small(candidate):
                    if check_medium(candidate):
                        if is_sorted_network(candidate):
                            best = candidate
                            best_len = len(best)
                            found_improvement = True
                            break
            if found_improvement:
                break
        
        if not found_improvement:
            # Try triple removals
            found_improvement = False
            for i in range(min(3, len(candidates_sorted))):
                for j in range(i + 1, min(3, len(candidates_sorted))):
                    for k in range(j + 1, min(3, len(candidates_sorted))):
                        if time.time() > deadline - 3:
                            break
                        cands = sorted([candidates_sorted[i], candidates_sorted[j], candidates_sorted[k]])
                        ci, cj, ck = cands
                        candidate = best[:ci] + best[ci+1:cj] + best[cj+1:ck] + best[ck+1:]
                        if quick_check_small(candidate):
                            if check_medium(candidate):
                                if is_sorted_network(candidate):
                                    best = candidate
                                    best_len = len(best)
                                    found_improvement = True
                                    break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if not found_improvement:
                break
    
    return best

def sequential_removal_pass(comparators, deadline):
    """
    Sequential pass: try removing each comparator in order.
    Uses fast pre-filtering.
    """
    best = list(comparators)
    
    while time.time() < deadline - 3:
        removed_any = False
        i = 0
        while i < len(best):
            if time.time() > deadline - 3:
                break
            candidate = best[:i] + best[i+1:]
            if quick_check_small(candidate):
                if check_medium(candidate):
                    if is_sorted_network(candidate):
                        best = candidate
                        removed_any = True
                        # Don't increment i, re-check same position
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        if not removed_any:
            break
    
    return best

def reorder_and_remove(comparators, deadline):
    """
    Try reordering comparators (swapping adjacent ones that don't share wires)
    and then removing redundancies. This can escape local minima.
    """
    best = list(comparators)
    best_len = len(best)
    
    while time.time() < deadline - 5:
        # Try swapping adjacent comparators that don't share wires
        improved = False
        for i in range(len(best) - 1):
            if time.time() > deadline - 5:
                break
            i1, j1 = best[i]
            i2, j2 = best[i+1]
            
            # Only swap if they don't share wires
            if i1 == i2 or i1 == j2 or j1 == i2 or j1 == j2:
                continue
            
            # Try the swap
            swapped = best[:i] + [best[i+1], best[i]] + best[i+2:]
            
            # After swap, try to remove a comparator
            for remove_idx in range(len(swapped)):
                if time.time() > deadline - 5:
                    break
                candidate = swapped[:remove_idx] + swapped[remove_idx+1:]
                if quick_check_small(candidate):
                    if check_medium(candidate):
                        if is_sorted_network(candidate):
                            best = candidate
                            best_len = len(best)
                            improved = True
                            break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def random_reorder_search(comparators, deadline):
    """
    Try random reordering of the comparator list and then remove redundancies.
    Different orderings can make different comparators redundant.
    """
    best = list(comparators)
    best_len = len(best)
    
    attempts = 0
    max_attempts = 30
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Create a random permutation of the comparators
        shuffled = list(best)
        random.shuffle(shuffled)
        
        # Try removing comparators from the shuffled version
        candidate = list(shuffled)
        i = 0
        while i < len(candidate) and time.time() < deadline - 5:
            test = candidate[:i] + candidate[i+1:]
            if quick_check_small(test):
                if check_medium(test):
                    if is_sorted_network(test):
                        candidate = test
                        # Don't increment i
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use odd-even merge sort as the base network (109 comparators vs 136 for transposition)
    base = odd_even_merge_sort_network(N)
    
    # Verify base is correct
    if not is_sorted_network(base):
        # Fallback to odd-even transposition
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Sequential removal (most effective, do it first)
    if time.time() < deadline - 30:
        best = sequential_removal_pass(best, deadline)
        best_len = len(best)
    
    # Phase 2: Batch removal using redundancy scores
    if time.time() < deadline - 20:
        best = batch_removal_search(best, deadline)
        best_len = len(best)
    
    # Phase 3: Another sequential pass to catch remaining redundancies
    if time.time() < deadline - 15:
        best = sequential_removal_pass(best, deadline)
        best_len = len(best)
    
    # Phase 4: Reorder and remove to escape local minima
    if time.time() < deadline - 10:
        best = reorder_and_remove(best, deadline)
        best_len = len(best)
    
    # Phase 5: Random reorder search
    if time.time() < deadline - 8:
        best = random_reorder_search(best, deadline)
        best_len = len(best)
    
    # Phase 6: Final sequential removal
    if time.time() < deadline - 3:
        best = sequential_removal_pass(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
