import numpy as np
import time

def _compute_ratio(P):
    """Compute the ratio D/M for a given 10x10 weight matrix P."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    def entropy_from_values(values, weights):
        min_v = np.min(values)
        shifted = values - min_v
        counts = np.bincount(shifted, weights=weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_values(x_flat - y_flat, w)
    H0 = entropy_from_values(x_flat, w)
    H1 = entropy_from_values(x_flat + y_flat, w)
    H2 = entropy_from_values(x_flat + 2 * y_flat, w)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _compute_ratio_fast(w, diff_bins, h0_bins, h1_bins, h2_bins, n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2):
    """Fast ratio computation using precomputed bin indices."""
    def entropy_from_bins(bins, weights, n_bins):
        counts = np.zeros(n_bins)
        np.add.at(counts, bins, weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_bins(diff_bins, w, n_bins_diff)
    H0 = entropy_from_bins(h0_bins, w, n_bins_h0)
    H1 = entropy_from_bins(h1_bins, w, n_bins_h1)
    H2 = entropy_from_bins(h2_bins, w, n_bins_h2)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _soft_max_entropy_ratio(w, diff_bins, h0_bins, h1_bins, h2_bins, 
                            n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2, beta=10.0):
    """Compute ratio with smooth max via log-sum-exp for gradient optimization."""
    def entropy_from_bins(bins, weights, n_bins):
        counts = np.zeros(n_bins)
        np.add.at(counts, bins, weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_from_bins(diff_bins, w, n_bins_diff)
    H0 = entropy_from_bins(h0_bins, w, n_bins_h0)
    H1 = entropy_from_bins(h1_bins, w, n_bins_h1)
    H2 = entropy_from_bins(h2_bins, w, n_bins_h2)
    
    # Smooth max using log-sum-exp
    h_vals = np.array([H0, H1, H2])
    h_max = np.max(h_vals)
    log_sum_exp = h_max + np.log(np.sum(np.exp((h_vals - h_max) * beta)) / beta)
    M = log_sum_exp / beta  # This approximates max as beta -> inf
    
    if M <= 1e-9:
        return 0.0
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    # Precompute bin indices for fast entropy computation
    diff_vals = x_flat - y_flat  # range -9..9, shift by 9 -> 0..18
    diff_bins = diff_vals + 9
    n_bins_diff = 19
    
    h0_vals = x_flat  # range 0..9
    h0_bins = h0_vals
    n_bins_h0 = 10
    
    h1_vals = x_flat + y_flat  # range 0..18
    h1_bins = h1_vals
    n_bins_h1 = 19
    
    h2_vals = x_flat + 2 * y_flat  # range 0..27
    h2_bins = h2_vals
    n_bins_h2 = 28
    
    def compute_ratio(w):
        return _compute_ratio_fast(w, diff_bins, h0_bins, h1_bins, h2_bins,
                                   n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Candidate 1: uniform
    candidates.append(np.ones((n, n), dtype=np.float64) / 100.0)
    
    # Candidate 2: concentrate on main diagonal (x = y), where x-y = 0
    w2 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w2[i, i] = 1.0
    w2 /= w2.sum()
    candidates.append(w2)
    
    # Candidate 3: weight cells to make x-y as uniform as possible
    # For each x-y value d, there are cells (x,y) with x-y=d
    # Assign weight proportional to 1/sqrt(count) to spread more
    w3 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            count_d = min(n - abs(d), n)
            w3[i, j] = 1.0 / np.sqrt(count_d)
    w3 /= w3.sum()
    candidates.append(w3)
    
    # Candidate 4: focus on cells where x is in a narrow range but x-y varies
    w4 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(2, 8):
            w4[i, j] = 1.0
    w4 /= w4.sum()
    candidates.append(w4)
    
    # Candidate 5: anti-diagonal focus (x + y = constant)
    w5 = np.zeros((n, n), dtype=np.float64)
    target_sum = 9
    for i in range(n):
        j = target_sum - i
        if 0 <= j < n:
            w5[i, j] = 1.0
    w5 /= w5.sum()
    candidates.append(w5)
    
    # Candidate 6: combine strategies
    w6 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d = j - i
            s_xy = i + j
            score = np.exp(-0.1 * np.abs(d)) * np.exp(-0.05 * np.abs(s_xy - 9))
            w6[i, j] = score
    w6 /= w6.sum()
    candidates.append(w6)
    
    # Candidate 7: Make x+y concentrated while x-y spread
    # Put weight on cells where x+y = 4 or 5 (narrow range) but x-y varies
    w7 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            s = i + j
            if s in [4, 5]:
                d = j - i
                w7[i, j] = 1.0
    if w7.sum() > 0:
        w7 /= w7.sum()
        candidates.append(w7)
    
    # Candidate 8: Make x concentrated while x-y spread
    w8 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(4, 6):  # x in {4,5}
            w8[i, j] = 1.0
    if w8.sum() > 0:
        w8 /= w8.sum()
        candidates.append(w8)
    
    # Evaluate all candidates
    best_score = -1
    best_P = candidates[0]
    for c in candidates:
        sc = compute_ratio(c.flatten())
        if sc > best_score:
            best_score = sc
            best_P = c.copy()
    
    # Phase 1: Multi-cell local search
    np.random.seed(42)
    current_P = best_P.copy()
    current_score = best_score
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 150
    phase1_end = start_time + 180
    
    while time.time() < phase1_end:
        if no_improve_count > max_no_improve:
            # Restart from best with perturbation
            if current_score < best_score * 0.99:
                noise = np.random.normal(0, 0.15, (n, n))
                current_P = best_P + noise
                current_P = np.clip(current_P, 0, None)
                s = current_P.sum()
                if s > 1e-9:
                    current_P /= s
                else:
                    current_P = best_P.copy()
                current_score = compute_ratio(current_P.flatten())
            no_improve_count = 0
            step_size = 0.15
            max_no_improve = 100
            continue
        
        # Multi-cell move: perturb 2-4 cells
        n_cells = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.2])
        indices = np.random.randint(0, n * n, n_cells)
        deltas = np.random.uniform(-step_size, step_size, n_cells)
        
        new_P = current_P.copy()
        new_P_flat = new_P.flatten()
        new_P_flat[indices] += deltas
        new_P_flat = np.clip(new_P_flat, 0, None)
        s = new_P_flat.sum()
        if s <= 1e-9:
            continue
        new_P_flat /= s
        new_P = new_P_flat.reshape(n, n)
        
        new_score = compute_ratio(new_P_flat)
        
        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.98
        else:
            no_improve_count += 1
            step_size = max(0.01, step_size * 1.02)
    
    # Phase 2: Gradient-based optimization using log-space parameterization
    from scipy.optimize import minimize
    
    P_for_log = best_P.copy()
    P_for_log[P_for_log < 1e-12] = 1e-12
    log_P_init = np.log(P_for_log.flatten())
    
    def neg_ratio_from_log(log_p_flat):
        log_p = log_p_flat.reshape(n, n)
        log_p_max = np.max(log_p)
        p = np.exp(log_p - log_p_max)
        p_sum = p.sum()
        if p_sum <= 1e-15:
            return 1e10
        p = p / p_sum
        w = p.flatten()
        # Use smooth max for gradient
        ratio = _soft_max_entropy_ratio(w, diff_bins, h0_bins, h1_bins, h2_bins,
                                        n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2, beta=50.0)
        return -ratio
    
    phase2_end = start_time + timeout - safety_margin
    gradient_starts = [log_P_init]
    
    for _ in range(4):
        perturbed = log_P_init + np.random.normal(0, 0.5, n * n)
        gradient_starts.append(perturbed)
    
    best_gradient_score = best_score
    best_gradient_P = best_P.copy()
    
    for log_start in gradient_starts:
        if time.time() > phase2_end - 90:
            break
        
        try:
            result = minimize(
                neg_ratio_from_log,
                log_start,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            
            log_p_opt = result.x.reshape(n, n)
            log_p_max = np.max(log_p_opt)
            p_opt = np.exp(log_p_opt - log_p_max)
            p_opt = p_opt / p_opt.sum()
            
            score_opt = compute_ratio(p_opt.flatten())
            if score_opt > best_gradient_score:
                best_gradient_score = score_opt
                best_gradient_P = p_opt.copy()
        except Exception:
            continue
    
    if best_gradient_score > best_score:
        best_score = best_gradient_score
        best_P = best_gradient_P.copy()
    
    # Phase 3: Final high-precision polish
    current_P = best_P.copy()
    current_score = best_score
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 80
    
    while time.time() < phase2_end - 15:
        if no_improve_count > max_no_improve:
            break
        
        # Single cell fine-tuning
        i, j = np.random.randint(0, n, 2)
        delta = np.random.uniform(-step_size, step_size)
        
        new_P = current_P.copy()
        new_P[i, j] += delta
        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s
        
        new_score = compute_ratio(new_P.flatten())
        
        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.95
        else:
            no_improve_count += 1
            step_size = max(0.0005, step_size * 1.03)
    
    # Final validation and normalization
    result = np.clip(best_P, 0, None)
    s = result.sum()
    if s <= 1e-9:
        result = np.ones((n, n)) / 100.0
    else:
        result = result / s
    
    return result
