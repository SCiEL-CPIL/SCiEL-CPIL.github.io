# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support value
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    # Efficient method using sorted pair sums - works for all n
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    cum_probs = np.cumsum(sorted_probs)
    
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * inner_sums)
    return float(P)

def _precompute_support(support):
    """Precompute sorted pair sums for a fixed support."""
    s = np.asarray(support, dtype=np.float64)
    order = np.argsort(s)
    s_sorted = s[order]
    n = len(s_sorted)
    
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    flat_sums = pair_sums.flatten()
    sort_order = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_order]
    return s_sorted, sorted_sums, sort_order

def _compute_P_fixed_support(s_sorted, weights, sorted_sums, sort_order):
    """Fast P computation when support is fixed and sorted_sums precomputed."""
    w = np.asarray(weights, dtype=np.float64)
    n = len(s_sorted)
    
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    pair_probs = w[:, None] * w[None, :]
    flat_probs = pair_probs.flatten()
    sorted_probs = flat_probs[sort_order]
    cum_probs = np.cumsum(sorted_probs)
    
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w[:, None] * w[None, :] * inner_sums)
    return float(P)

def _coordinate_descent(s_sorted, weights, sorted_sums, sort_order, timeout=60):
    """Custom coordinate descent for weight optimization with fixed support."""
    n = len(s_sorted)
    w = np.asarray(weights, dtype=np.float64).copy()
    w_sum = w.sum()
    w = w / w_sum
    
    best_P = _compute_P_fixed_support(s_sorted, w, sorted_sums, sort_order)
    start_time = time.time()
    
    step_size = 0.1
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try adjusting each weight
        for i in range(n):
            for delta in [step_size, -step_size, step_size*0.1, -step_size*0.1]:
                new_w = w.copy()
                new_w[i] += delta
                if new_w[i] < 0:
                    continue
                new_w_sum = new_w.sum()
                if new_w_sum <= 1e-9:
                    continue
                new_w = new_w / new_w_sum
                P = _compute_P_fixed_support(s_sorted, new_w, sorted_sums, sort_order)
                if P > best_P + 1e-12:
                    w = new_w
                    best_P = P
                    improved = True
        
        # Try adjusting pairs of weights
        for i in range(n):
            for j in range(i+1, n):
                for delta in [step_size, -step_size]:
                    new_w = w.copy()
                    new_w[i] += delta
                    new_w[j] -= delta
                    if new_w[i] < 0 or new_w[j] < 0:
                        continue
                    new_w_sum = new_w.sum()
                    if new_w_sum <= 1e-9:
                        continue
                    new_w = new_w / new_w_sum
                    P = _compute_P_fixed_support(s_sorted, new_w, sorted_sums, sort_order)
                    if P > best_P + 1e-12:
                        w = new_w
                        best_P = P
                        improved = True
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.7
            if step_size < 1e-8 or no_improve_count > 50:
                break
        else:
            no_improve_count = 0
    
    return w, best_P

def _coordinate_descent_support(s_sorted, weights, timeout=60):
    """Custom coordinate descent for joint support and weight optimization."""
    n = len(s_sorted)
    s = s_sorted.copy()
    w = np.asarray(weights, dtype=np.float64).copy()
    w_sum = w.sum()
    w = w / w_sum
    
    best_P = _compute_P(s, w)
    start_time = time.time()
    
    step_size_s = 0.05
    step_size_w = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try adjusting each support point
        for i in range(n):
            for delta in [step_size_s, -step_size_s]:
                new_s = s.copy()
                new_s[i] += delta
                if new_s[i] < 0:
                    continue
                new_s = np.sort(new_s)
                P = _compute_P(new_s, w)
                if P > best_P + 1e-12:
                    s = new_s
                    best_P = P
                    improved = True
        
        # Try adjusting weights
        for i in range(n):
            for delta in [step_size_w, -step_size_w]:
                new_w = w.copy()
                new_w[i] += delta
                if new_w[i] < 0:
                    continue
                new_w_sum = new_w.sum()
                if new_w_sum <= 1e-9:
                    continue
                new_w = new_w / new_w_sum
                P = _compute_P(s, new_w)
                if P > best_P + 1e-12:
                    w = new_w
                    best_P = P
                    improved = True
        
        if not improved:
            no_improve_count += 1
            step_size_s *= 0.7
            step_size_w *= 0.7
            if step_size_s < 1e-8 or no_improve_count > 50:
                break
        else:
            no_improve_count = 0
    
    return s, w, best_P

def _simulated_annealing(s_sorted, weights, sorted_sums, sort_order, timeout=60):
    """Simulated annealing for weight optimization with fixed support."""
    n = len(s_sorted)
    w = np.asarray(weights, dtype=np.float64).copy()
    w_sum = w.sum()
    w = w / w_sum
    
    best_P = _compute_P_fixed_support(s_sorted, w, sorted_sums, sort_order)
    best_w = w.copy()
    current_P = best_P
    current_w = w.copy()
    
    start_time = time.time()
    T = 0.1
    cooling = 0.995
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < timeout and T > 1e-8:
        # Propose perturbation
        new_w = current_w.copy()
        idx = rng.randint(n)
        delta = rng.randn() * 0.05
        new_w[idx] += delta
        if new_w[idx] < 0:
            continue
        new_w_sum = new_w.sum()
        if new_w_sum <= 1e-9:
            continue
        new_w = new_w / new_w_sum
        
        new_P = _compute_P_fixed_support(s_sorted, new_w, sorted_sums, sort_order)
        
        if new_P > current_P or rng.random() < np.exp((new_P - current_P) / T):
            current_P = new_P
            current_w = new_w
            if new_P > best_P:
                best_P = new_P
                best_w = new_w.copy()
        
        T *= cooling
    
    return best_w, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 580  # Safety margin
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def _update_best(s, w, P):
        nonlocal best_support, best_weights, best_P
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 2-point optimal (analytic solution)
    p_opt = (15 - np.sqrt(33)) / 16
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1 - p_opt])
    initial_configs.append((s2, w2))
    
    # 2-point with different scales
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        s = np.array([0.0, scale])
        w = np.array([p_opt, 1 - p_opt])
        initial_configs.append((s, w))
    
    # 3-point configurations
    for a in [0.3, 0.5, 0.7, 1.0]:
        for b in [1.0, 2.0, 3.0, 5.0]:
            if a < b:
                s = np.array([0.0, a, b])
                for p0 in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    for p1 in [0.1, 0.2, 0.3, 0.4]:
                        if p0 + p1 < 1.0:
                            w = np.array([p0, p1, 1 - p0 - p1])
                            initial_configs.append((s, w))
    
    # 4-point configurations
    for n in [4, 5, 6, 8]:
        # Geometric spacing with various ratios
        for ratio in [1.5, 2.0, 3.0, 5.0]:
            s = np.array([0.0] + [ratio ** i for i in range(1, n)])
            s = s / s.max()
            # Various weight distributions
            for skew in [0.5, 1.0, 2.0, 3.0, 5.0]:
                w = np.exp(-skew * s)
                w = w / w.sum()
                initial_configs.append((s, w))
        
        # Uniform with various weight biases
        s = np.linspace(0, 1, n)
        for skew in [0.5, 1.0, 2.0, 3.0, 5.0]:
            w = np.exp(-skew * s)
            w = w / w.sum()
            initial_configs.append((s, w))
        
        # Bimodal
        half = n // 2
        s = np.concatenate([np.linspace(0, 0.2, half), np.linspace(0.5, 1.0, n - half)])
        for w0 in [0.3, 0.4, 0.5, 0.6]:
            w = np.concatenate([np.ones(half) * w0, np.ones(n - half) * (1 - w0)])
            w = w / w.sum()
            initial_configs.append((s, w))
    
    # Larger n configurations
    for n in [10, 12, 15]:
        # Log-spaced
        s = np.logspace(-3, 0, n)
        s = s / s.max()
        for skew in [1.0, 2.0, 3.0]:
            w = np.exp(-skew * s)
            w = w / w.sum()
            initial_configs.append((s, w))
        
        # Power-law
        s = np.array([i**0.5 for i in range(n)]) / np.sqrt(n - 1)
        w = np.ones(n) / n
        initial_configs.append((s, w))
        
        # Heavy at 0
        s = np.linspace(0, 1, n)
        w = np.exp(-3.0 * s)
        w = w / w.sum()
        initial_configs.append((s, w))
    
    # Evaluate all initial configs
    for support, weights in initial_configs:
        P = _compute_P(support, weights)
        _update_best(support, weights, P)
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Sort configs by initial P
    config_scores = []
    for i, (support, weights) in enumerate(initial_configs):
        P = _compute_P(support, weights)
        config_scores.append((P, i, support, weights))
    config_scores.sort(key=lambda x: -x[0])
    
    # Optimize top configs using coordinate descent and simulated annealing
    num_to_optimize = min(15, len(config_scores))
    for P_init, idx, support, weights in config_scores[:num_to_optimize]:
        if time.time() > deadline - 120:
            break
        
        n_pts = len(support)
        
        # Precompute support structure
        s_sorted, sorted_sums, sort_order = _precompute_support(support)
        
        # Coordinate descent on weights
        try:
            timeout_cd = min(40, max(10, (deadline - time.time()) * 0.3))
            opt_w, P_cd = _coordinate_descent(s_sorted, weights, sorted_sums, sort_order, timeout=timeout_cd)
            _update_best(support, opt_w, P_cd)
        except Exception:
            pass
        
        # Simulated annealing on weights
        try:
            timeout_sa = min(30, max(5, (deadline - time.time()) * 0.2))
            sa_w, P_sa = _simulated_annealing(s_sorted, weights, sorted_sums, sort_order, timeout=timeout_sa)
            _update_best(support, sa_w, P_sa)
        except Exception:
            pass
        
        # Joint coordinate descent for smaller supports
        if n_pts <= 10 and time.time() < deadline - 60:
            try:
                timeout_joint = min(30, max(5, (deadline - time.time()) * 0.3))
                opt_s, opt_w, P_joint = _coordinate_descent_support(s_sorted, weights, timeout=timeout_joint)
                _update_best(opt_s, opt_w, P_joint)
            except Exception:
                pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Final polish: multiple restarts with perturbation
    if best_support is not None and time.time() < deadline - 30:
        n = len(best_support)
        s_sorted, sorted_sums, sort_order = _precompute_support(best_support)
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(42 + trial * 137)
            scale_s = 0.03 * (1 + trial * 0.2)
            scale_w = 0.05 * (1 + trial * 0.1)
            
            perturbed_support = best_support + rng.randn(n) * scale_s
            perturbed_support = np.maximum(perturbed_support, 0.0)
            perturbed_support = np.sort(perturbed_support)
            
            perturbed_weights = best_weights * (1 + rng.randn(n) * scale_w)
            perturbed_weights = np.maximum(perturbed_weights, 1e-8)
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            # Coordinate descent from perturbed point
            try:
                timeout_cd = min(20, max(3, (deadline - time.time()) * 0.3))
                ps_sorted, ps_sums, ps_order = _precompute_support(perturbed_support)
                opt_w, P_cd = _coordinate_descent(ps_sorted, perturbed_weights, ps_sums, ps_order, timeout=timeout_cd)
                _update_best(perturbed_support, opt_w, P_cd)
            except Exception:
                pass
            
            # Joint coordinate descent
            if n <= 8 and time.time() < deadline - 15:
                try:
                    timeout_joint = min(15, max(3, (deadline - time.time()) * 0.3))
                    opt_s, opt_w, P_joint = _coordinate_descent_support(perturbed_support, perturbed_weights, timeout=timeout_joint)
                    _update_best(opt_s, opt_w, P_joint)
                except Exception:
                    pass
    
    # Ensure validity
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([p_opt, 1 - p_opt], dtype=np.float64)
    
    # Clean up: ensure no NaN/inf, non-negative
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=1e6, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=1.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights is positive
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
