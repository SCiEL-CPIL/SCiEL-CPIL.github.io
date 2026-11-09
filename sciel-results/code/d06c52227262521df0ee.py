# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using O(n^2 log n) searchsorted.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    if n == 1:
        return 0.0
    
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_probs = np.outer(w, w)
    
    sums_flat = pair_sums.flatten()
    probs_flat = pair_probs.flatten()
    
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
            cum_probs[-1] = 1.0
    
    thresholds = 2.0 * s_row - s_col
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    return float(P)

def _normalize(s, w):
    """Normalize weights and clean up."""
    w = np.maximum(w, 0.0)
    s = np.maximum(s, 0.0)
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return s, w
    w = w / w_sum
    return s, w

def _ga_mutation(s, w, rate=0.1, sigma_s=0.1, sigma_w=0.1):
    """Apply Gaussian mutation to support and weights."""
    n = len(s)
    s_new = s.copy()
    w_new = w.copy()
    
    mask_s = np.random.random(n) < rate
    mask_w = np.random.random(n) < rate
    
    if np.any(mask_s):
        s_new[mask_s] += np.random.normal(0, sigma_s, np.sum(mask_s))
    if np.any(mask_w):
        w_new[mask_w] += np.random.normal(0, sigma_w, np.sum(mask_w))
    
    return s_new, w_new

def _ga_crossover(s1, w1, s2, w2, alpha=0.5):
    """Uniform crossover between two individuals."""
    n1, n2 = len(s1), len(s2)
    n = max(n1, n2)
    
    s_new = np.zeros(n)
    w_new = np.zeros(n)
    
    for i in range(n):
        if np.random.random() < alpha:
            if i < n1:
                s_new[i] = s1[i]
                w_new[i] = w1[i]
            else:
                s_new[i] = s2[i - n1] if i - n1 < n2 else np.random.uniform(0, 1)
                w_new[i] = w1[i - n1] if i - n1 < n1 else np.random.uniform(0.01, 0.1)
        else:
            if i < n2:
                s_new[i] = s2[i]
                w_new[i] = w2[i]
            else:
                s_new[i] = s1[i - n2] if i - n2 < n1 else np.random.uniform(0, 1)
                w_new[i] = w2[i - n2] if i - n2 < n2 else np.random.uniform(0.01, 0.1)
    
    return s_new, w_new

def _add_atom(s, w, t_start, deadline):
    """Try adding a new atom at a random position."""
    if time.time() > deadline - 5:
        return None, None
    
    n = len(s)
    if n >= 20:  # Cap to avoid O(n^2) explosion
        return None, None
    
    # Try adding at a random position
    s_new_pos = np.random.uniform(0, 2)
    w_new_val = np.random.uniform(0.01, 0.1)
    
    s_new = np.concatenate([s, [s_new_pos]])
    w_new = np.concatenate([w, [w_new_val]])
    s_new, w_new = _normalize(s_new, w_new)
    
    P_new = _compute_P_fast(s_new, w_new)
    return s_new, w_new, P_new

def _remove_atom(s, w):
    """Remove the atom with the smallest weight."""
    n = len(s)
    if n <= 2:
        return None, None
    
    idx = np.argmin(w)
    s_new = np.delete(s, idx)
    w_new = np.delete(w, idx)
    s_new, w_new = _normalize(s_new, w_new)
    
    P_new = _compute_P_fast(s_new, w_new)
    return s_new, w_new, P_new

def _evolve_population(pop_s, pop_w, pop_P, n_gen=50, t_start=None, deadline=None):
    """
    Evolve a population of measures using genetic operators.
    """
    if t_start is None:
        t_start = time.time()
    if deadline is None:
        deadline = t_start + 600
    
    best_idx = np.argmax(pop_P)
    best_s = pop_s[best_idx].copy()
    best_w = pop_w[best_idx].copy()
    best_P = pop_P[best_idx]
    
    pop_size = len(pop_s)
    
    for gen in range(n_gen):
        if time.time() > deadline - 30:
            break
        
        new_pop_s = []
        new_pop_w = []
        new_pop_P = []
        
        # Elitism: keep top 2
        top_idx = np.argsort(pop_P)[-2:]
        for idx in top_idx:
            new_pop_s.append(pop_s[idx].copy())
            new_pop_w.append(pop_w[idx].copy())
            new_pop_P.append(pop_P[idx])
        
        # Generate offspring
        while len(new_pop_s) < pop_size:
            if time.time() > deadline - 30:
                break
            
            op = np.random.random()
            
            if op < 0.4:
                # Crossover
                i = np.random.randint(0, pop_size)
                j = np.random.randint(0, pop_size)
                if i == j:
                    continue
                s_child, w_child = _ga_crossover(pop_s[i], pop_w[i], pop_s[j], pop_w[j])
            elif op < 0.7:
                # Mutation
                i = np.random.randint(0, pop_size)
                s_child, w_child = _ga_mutation(pop_s[i], pop_w[i])
            elif op < 0.85:
                # Add atom
                i = np.random.randint(0, pop_size)
                result = _add_atom(pop_s[i], pop_w[i], t_start, deadline)
                if result is None:
                    continue
                s_child, w_child, _ = result
            else:
                # Remove atom
                i = np.random.randint(0, pop_size)
                result = _remove_atom(pop_s[i], pop_w[i])
                if result is None:
                    continue
                s_child, w_child, _ = result
            
            s_child, w_child = _normalize(s_child, w_child)
            
            if len(s_child) < 1 or len(w_child) < 1:
                continue
            
            P_child = _compute_P_fast(s_child, w_child)
            
            if P_child > best_P:
                best_P = P_child
                best_s = s_child.copy()
                best_w = w_child.copy()
            
            new_pop_s.append(s_child)
            new_pop_w.append(w_child)
            new_pop_P.append(P_child)
        
        if len(new_pop_s) == 0:
            break
        
        pop_s = new_pop_s
        pop_w = new_pop_w
        pop_P = np.array(new_pop_P)
        
        # Selection: keep top pop_size
        if len(pop_P) > pop_size:
            keep_idx = np.argsort(pop_P)[-pop_size:]
            pop_s = [pop_s[i] for i in keep_idx]
            pop_w = [pop_w[i] for i in keep_idx]
            pop_P = pop_P[keep_idx]
    
    return best_s, best_w, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Adaptive variable-length evolutionary search with add/remove/mutate/crossover.
    """
    t_start = time.time()
    deadline = t_start + 590.0
    
    np.random.seed(42)
    
    # Initialize population with diverse starting points
    pop_s = []
    pop_w = []
    pop_P = []
    
    # Start with various n values
    for n in [2, 3, 4, 5, 6]:
        s0 = np.sort(np.random.uniform(0, 1.5, n))
        w0 = np.random.uniform(0.1, 1.0, n)
        s0, w0 = _normalize(s0, w0)
        P0 = _compute_P_fast(s0, w0)
        pop_s.append(s0)
        pop_w.append(w0)
        pop_P.append(P0)
    
    # Add some structured starts
    s_struct = np.array([0.0, 0.5, 1.0, 2.0])
    w_struct = np.array([0.4, 0.3, 0.2, 0.1])
    s_struct, w_struct = _normalize(s_struct, w_struct)
    P_struct = _compute_P_fast(s_struct, w_struct)
    pop_s.append(s_struct)
    pop_w.append(w_struct)
    pop_P.append(P_struct)
    
    s_struct2 = np.array([0.0, 0.1, 0.3, 0.7, 1.5])
    w_struct2 = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    s_struct2, w_struct2 = _normalize(s_struct2, w_struct2)
    P_struct2 = _compute_P_fast(s_struct2, w_struct2)
    pop_s.append(s_struct2)
    pop_w.append(w_struct2)
    pop_P.append(P_struct2)
    
    pop_P = np.array(pop_P)
    
    # Run evolution
    best_s, best_w, best_P = _evolve_population(pop_s, pop_w, pop_P, n_gen=100, t_start=t_start, deadline=deadline)
    
    # If no valid solution found, return a baseline
    if best_s is None or len(best_s) == 0:
        s_base = np.array([0.0, 1.0])
        w_base = np.array([0.5, 0.5])
        return s_base, w_base
    
    # Final polish: try local refinement on the best solution
    if time.time() < deadline - 30:
        s_curr = best_s.copy()
        w_curr = best_w.copy()
        P_curr = best_P
        
        # Coordinate ascent refinement
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            improved = False
            n = len(s_curr)
            
            # Optimize support positions
            for i in range(n):
                if w_curr[i] < 1e-8:
                    continue
                s_old = s_curr[i]
                for delta in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
                    s_test = s_curr.copy()
                    s_test[i] = max(0, s_old + delta)
                    P_test = _compute_P_fast(s_test, w_curr)
                    if P_test > P_curr + 1e-10:
                        s_curr = s_test
                        P_curr = P_test
                        improved = True
            
            # Optimize weights
            for i in range(n):
                if w_curr[i] < 1e-8:
                    continue
                w_old = w_curr[i]
                for factor in [0.7, 0.85, 1.15, 1.3]:
                    w_test = w_curr.copy()
                    w_test[i] = w_old * factor
                    w_test = np.maximum(w_test, 0)
                    w_sum = np.sum(w_test)
                    if w_sum <= 1e-9:
                        continue
                    w_test = w_test / w_sum
                    P_test = _compute_P_fast(s_curr, w_test)
                    if P_test > P_curr + 1e-10:
                        w_curr = w_test
                        P_curr = P_test
                        improved = True
            
            if not improved:
                break
        
        if P_curr > best_P:
            best_P = P_curr
            best_s = s_curr
            best_w = w_curr
    
    # Validation and formatting
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    if len(best_s) == 0 or len(best_w) == 0:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    if len(best_s) != len(best_w):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_w = best_w / w_sum
    
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    return best_s, best_w

# EVOLVE-BLOCK-END
