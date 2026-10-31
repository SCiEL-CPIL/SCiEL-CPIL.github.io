"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 50
DELTA = 1.0 / (2 * N)

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) with proper scaling."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h ** 2)
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    # The continuous supremum of the autoconvolution of piecewise-constant functions
    # on intervals of width DELTA is max(conv) * DELTA
    conv_inf = np.max(conv) * DELTA
    if conv_inf < 1e-15:
        return 1e15  # Arbitrarily large for zero off-peak
    return l2_sq / (l1 * conv_inf)

def _is_sidon_set(indices, max_diff=None):
    """Check if the set of indices forms a Sidon set (unique positive differences)."""
    n = len(indices)
    if max_diff is None:
        max_diff = max(indices) - min(indices)
    diffs = set()
    for i in range(n):
        for j in range(i):
            d = indices[i] - indices[j]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _generate_sidon_candidates(n_indices, max_idx):
    """Generate multiple Sidon set candidates using greedy with randomization."""
    candidates = []
    
    # Try different greedy strategies
    for trial in range(5):
        random.seed(42 + trial * 1000)
        indices = [random.randint(0, max_idx // 4)]
        for _ in range(n_indices - 1):
            # Try adding the largest possible index first, then work down
            best = None
            for candidate in range(max_idx, indices[-1], -1):
                new_indices = indices + [candidate]
                if _is_sidon_set(new_indices, max_idx):
                    best = candidate
                    break
            if best is None:
                # Try smaller indices
                for candidate in range(indices[-1] - 1, -1, -1):
                    if candidate in indices:
                        continue
                    new_indices = sorted(indices + [candidate])
                    if _is_sidon_set(new_indices, max_idx):
                        best = candidate
                        break
            if best is None:
                break
            indices.append(best)
        if len(indices) >= n_indices:
            candidates.append(sorted(indices))
    
    return candidates

def _optimize_weights_for_support(support, time_budget=10.0):
    """Optimize weights on a given support set to maximize the ratio."""
    indices = support
    n_support = len(indices)
    
    # Start with equal weights
    weights = np.ones(n_support)
    
    # Use local search: try perturbing weights
    best_weights = weights.copy()
    best_ratio = _compute_ratio_from_weights(weights, indices)
    
    start_time = time.time()
    step_size = 0.1
    iterations = 0
    
    while time.time() - start_time < time_budget and step_size > 1e-6:
        improved = False
        for i in range(n_support):
            if time.time() - start_time > time_budget:
                break
            for delta in [step_size, -step_size]:
                new_weights = best_weights.copy()
                new_weights[i] = max(0.01, best_weights[i] + delta)
                ratio = _compute_ratio_from_weights(new_weights, indices)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_weights = new_weights
                    improved = True
        if not improved:
            step_size *= 0.7
        iterations += 1
    
    return best_weights, best_ratio

def _compute_ratio_from_weights(weights, indices):
    """Compute ratio given weights and their support indices."""
    h = np.zeros(N)
    for idx, w in zip(indices, weights):
        h[idx] = w
    return _compute_ratio(h)

def _local_search_swap(h, time_budget=5.0):
    """Local search: try swapping or adjusting individual heights."""
    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    
    start_time = time.time()
    for iteration in range(1000):
        if time.time() - start_time > time_budget:
            break
        
        # Try adjusting a random element
        i = random.randint(0, N - 1)
        new_h = best_h.copy()
        
        # Random perturbation
        if best_h[i] > 0.1:
            new_h[i] = max(0.0, best_h[i] + random.gauss(0, 0.05))
        else:
            new_h[i] = abs(random.gauss(0, 0.05))
        
        ratio = _compute_ratio(new_h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = new_h
    
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_h = None
    best_ratio = 0.0
    
    # Strategy 1: Sidon set with optimized weights
    # Try different sizes of Sidon sets
    for n_indices in [10, 9, 8, 7]:
        if time.time() > deadline:
            break
        
        candidates = _generate_sidon_candidates(n_indices, N - 1)
        
        for support in candidates:
            if time.time() > deadline:
                break
            
            # Optimize weights on this support
            weights, ratio = _optimize_weights_for_support(support, time_budget=15.0)
            
            if ratio > best_ratio:
                best_ratio = ratio
                h = np.zeros(N)
                for idx, w in zip(support, weights):
                    h[idx] = w
                best_h = h.copy()
    
    # Strategy 2: Local search from best found so far
    if best_h is not None:
        if time.time() < deadline - 20:
            best_h, ratio = _local_search_swap(best_h, time_budget=20.0)
            if ratio > best_ratio:
                best_ratio = ratio
    
    # Fallback: single spike
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        best_ratio = _compute_ratio(best_h)
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
