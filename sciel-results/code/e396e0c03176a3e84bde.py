"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

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
_SAMPLE_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_inputs(n_samples=8192):
    global _SAMPLE_INPUTS
    if _SAMPLE_INPUTS is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
        _SAMPLE_INPUTS = all_inputs[idx]
    return _SAMPLE_INPUTS

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

def is_sorted_fast(comparators, inputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs = _get_sample_inputs()
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

def compute_redundancy_scores(comparators):
    """
    For each comparator, estimate how 'redundant' it is by checking
    how often it actually changes the output on random inputs.
    Lower score = more redundant = better candidate for removal.
    Uses vectorized simulation for speed.
    """
    sample = _get_sample_inputs(4096)
    n_comps = len(comparators)
    scores = np.zeros(n_comps, dtype=np.float64)
    
    # Vectorized simulation tracking swaps
    state = sample.copy()
    for ci, (i, j) in enumerate(comparators):
        a = state[:, i]
        b = state[:, j]
        swap_mask = a > b
        scores[ci] = np.sum(swap_mask)
        # Apply swap where needed
        state[:, i] = np.where(swap_mask, b, a)
        state[:, j] = np.where(swap_mask, a, b)
    
    return scores

def improve_by_removal_smart(comparators, deadline):
    """
    Remove comparators in order of redundancy score (most redundant first).
    Multiple passes with re-scoring.
    """
    best = list(comparators)
    
    while time.time() < deadline - 5:
        if len(best) == 0:
            break
        
        # Score all comparators
        scores = compute_redundancy_scores(best)
        
        # Sort by score ascending (most redundant first)
        order = sorted(range(len(best)), key=lambda x: scores[x])
        
        removed_any = False
        for idx in order:
            if time.time() > deadline - 5:
                break
            if idx >= len(best):
                break
            
            candidate = best[:idx] + best[idx+1:]
            if is_sorted_fast(candidate):
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                    # Re-score since indices changed
                    break
        
        if not removed_any:
            break
    
    return best

def improve_by_removal_sequential(comparators, deadline):
    """
    Sequential pass: try removing each comparator in order.
    Multiple passes. Uses a quick pre-filter with fewer samples.
    """
    best = list(comparators)
    
    # Use a smaller sample for fast pre-filtering
    pre_sample = _get_sample_inputs(2048)
    
    def quick_check(candidate):
        """Very fast check on a small sample."""
        state = pre_sample.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    while time.time() < deadline - 5:
        removed_any = False
        i = 0
        while i < len(best):
            if time.time() > deadline - 5:
                break
            candidate = best[:i] + best[i+1:]
            if quick_check(candidate):
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                    # Don't increment i
                else:
                    i += 1
            else:
                i += 1
        
        if not removed_any:
            break
    
    return best

def try_insert_and_remove(best, deadline):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    max_attempts = 200
    
    # Pre-compute a small sample for quick checks
    pre_sample = _get_sample_inputs(1024)
    
    def quick_check(candidate):
        state = pre_sample.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Try to remove comparators from the candidate
        k = 0
        improved = True
        while k < len(candidate) and improved:
            improved = False
            if time.time() > deadline - 3:
                break
            test = candidate[:k] + candidate[k+1:]
            if quick_check(test):
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                else:
                    k += 1
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

def build_layers(comparators):
    """
    Build depth-parallel layers from a sequential comparator list.
    Each layer contains disjoint comparators (no shared wires).
    Returns list of layers, where each layer is a list of (i,j) pairs.
    """
    layers = []
    current_layer = []
    used_wires = set()
    
    for (i, j) in comparators:
        if i in used_wires or j in used_wires:
            layers.append(current_layer)
            current_layer = []
            used_wires = set()
        current_layer.append((i, j))
        used_wires.add(i)
        used_wires.add(j)
    
    if current_layer:
        layers.append(current_layer)
    
    return layers

def flatten_layers(layers):
    """Flatten layers back to a sequential list."""
    result = []
    for layer in layers:
        result.extend(layer)
    return result

def optimize_layer_structure(comparators, deadline):
    """
    Use layer structure to find redundancies.
    Within a layer, if two comparators are on disjoint wires,
    we can try removing one and see if the other compensates.
    Also try reordering comparators across layers.
    """
    best = list(comparators)
    
    pre_sample = _get_sample_inputs(2048)
    
    def quick_check(candidate):
        state = pre_sample.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    while time.time() < deadline - 5:
        layers = build_layers(best)
        removed_any = False
        
        # Try removing comparators, prioritizing those in larger layers
        # (more parallelism = more likely redundant)
        for layer_idx in range(len(layers)):
            if time.time() > deadline - 5:
                break
            for comp_idx in range(len(layers[layer_idx])):
                if time.time() > deadline - 5:
                    break
                # Find global index of this comparator
                global_idx = sum(len(layers[l]) for l in range(layer_idx)) + comp_idx
                if global_idx >= len(best):
                    continue
                
                candidate = best[:global_idx] + best[global_idx+1:]
                if quick_check(candidate):
                    if is_sorted_network(candidate):
                        best = candidate
                        removed_any = True
                        break
            if removed_any:
                break
        
        if not removed_any:
            break
    
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use odd-even merge sort as the base network
    base = odd_even_merge_sort_network(N)
    
    # Verify base is correct
    if not is_sorted_network(base):
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Sequential removal (most effective, do it first)
    if time.time() < deadline - 20:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Phase 2: Smart removal based on redundancy scores
    if time.time() < deadline - 15:
        best = improve_by_removal_smart(best, deadline)
        best_len = len(best)
    
    # Phase 3: Another sequential pass to catch remaining redundancies
    if time.time() < deadline - 10:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Phase 4: Try insert-then-remove to escape local minima
    if time.time() < deadline - 8:
        best = try_insert_and_remove(best, deadline)
        best_len = len(best)
    
    # Phase 5: Final sequential removal
    if time.time() < deadline - 3:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END
