# EVOLVE-BLOCK-START
import numpy as np
import time

def _clip_and_normalize(P):
    """Clip negative values to 0 and normalize to sum 1."""
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.ones((10, 10)) / 100.0
    return P / total

def _shannon_entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count array."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 1e-15:
        return 0.0
    probs = counts / total
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    return float(-np.sum(probs[mask] * np.log2(probs[mask])))

def _compute_ratio(P):
    """Compute D/M ratio for a given 10x10 distribution P[y, x]."""
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y
    
    # D: H of x - y (slope -1)
    vals_d = X - Y  # values from -9 to 9
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (vals_d == v - 9)
        counts_d[v] = P[mask].sum()
    H_d = _shannon_entropy_from_counts(counts_d)
    
    # M: max of H for slopes 0, 1, 2
    max_h = 0.0
    for r in [0, 1, 2]:
        vals = X + r * Y
        # For r=0: values 0..9 (10 values)
        # For r=1: values 0..18 (19 values)
        # For r=2: values 0..27 (28 values)
        num_vals = n + r * (n - 1)
        counts = np.zeros(num_vals)
        for v in range(num_vals):
            mask = (vals == v)
            counts[v] = P[mask].sum()
        H_r = _shannon_entropy_from_counts(counts)
        if H_r > max_h:
            max_h = H_r
    
    if max_h <= 1e-9:
        return 0.0
    return H_d / max_h

def _compute_gradient(P, ratio):
    """
    Compute gradient of D/M with respect to P.
    Uses the chain rule and the fact that entropy gradient is -log2(p) - 1/ln(2) * ... 
    Actually, for entropy H = -sum p_i log2 p_i, dH/dp_i = -log2(p_i) - 1/ln(2)
    But we need to be careful: the counts are linear in P, so d(count)/dP_j = 1 if cell j is in that bin.
    Let's derive:
    H = -sum_bins (c_bin / S) log2(c_bin / S) where S = total sum = 1 (since normalized)
    But P is already normalized. Let's work with the unnormalized weights w = P * 100 (or just P since sum=1).
    Actually, let's use the fact that for a normalized distribution, the gradient of H with respect to p_j is -log2(p_j) - 1/ln(2).
    But here, the "probability" of a bin is c_bin = sum_{j in bin} P_j.
    So dH/dP_j = dH/dc_bin * dc_bin/dP_j = (-log2(c_bin) - 1/ln(2)) * (1/S) * ... wait.
    
    Let H = -sum_bins p_bin log2 p_bin, where p_bin = c_bin / S, S = sum P = 1.
    So H = -sum_bins c_bin log2(c_bin) + sum_bins c_bin log2(S) = -sum_bins c_bin log2(c_bin) since S=1.
    dH/dc_bin = -log2(c_bin) - 1/ln(2).
    dc_bin/dP_j = 1 if j in bin, 0 otherwise.
    So dH/dP_j = -log2(c_bin(j)) - 1/ln(2), where c_bin(j) is the count of the bin containing cell j.
    
    For D/M, let D = H_d, M = max(H_0, H_1, H_2).
    d(D/M)/dP_j = (dD/dP_j * M - D * dM/dP_j) / M^2.
    dM/dP_j = d(H_r*)/dP_j where r* is the slope achieving the max.
    If multiple slopes tie, we can use a soft-max or just pick one. For simplicity, let's use the argmax.
    """
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    log2_factor = 1.0 / np.log(2.0)
    
    # D gradient
    vals_d = X - Y
    bin_d = vals_d + 9  # 0..18
    counts_d = np.zeros(19)
    for v in range(19):
        mask = (bin_d == v)
        counts_d[v] = P[mask].sum()
    # dH_d/dP_j = -log2(counts_d[bin_d[j]]) - log2_factor
    grad_d = -np.log2(np.maximum(counts_d[bin_d], 1e-15)) - log2_factor
    
    # M gradient: find which r gives max H
    H_vals = []
    bin_ids = []
    counts_list = []
    for r in [0, 1, 2]:
        vals = X + r * Y
        num_vals = n + r * (n - 1)
        bin_id = vals  # 0..num_vals-1
        counts = np.zeros(num_vals)
        for v in range(num_vals):
            mask = (bin_id == v)
            counts[v] = P[mask].sum()
        H_r = _shannon_entropy_from_counts(counts)
        H_vals.append(H_r)
        bin_ids.append(bin_id)
        counts_list.append(counts)
    
    r_star = int(np.argmax(H_vals))
    M = H_vals[r_star]
    
    # dH_r*/dP_j
    counts_r = counts_list[r_star]
    bin_r = bin_ids[r_star]
    grad_m = -np.log2(np.maximum(counts_r[bin_r], 1e-15)) - log2_factor
    
    D = H_vals[0]  # Wait, H_d is separate. Let's recompute H_d properly.
    H_d = _shannon_entropy_from_counts(counts_d)
    
    if M <= 1e-9:
        return np.zeros_like(P)
    
    grad = (grad_d * M - H_d * grad_m) / (M * M)
    return grad

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # 590s safety margin
    
    # Initialize with uniform
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P.copy()
    best_ratio = _compute_ratio(P)
    
    # Gradient ascent with adaptive step size
    step_size = 0.01
    patience = 0
    max_patience = 500
    
    while time.time() < deadline:
        P = _clip_and_normalize(P)
        ratio = _compute_ratio(P)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P.copy()
            patience = 0
        else:
            patience += 1
        
        if patience >= max_patience:
            break
        
        grad = _compute_gradient(P, ratio)
        
        # Normalize gradient to avoid huge steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        P_new = P + step_size * grad
        P_new = _clip_and_normalize(P_new)
        
        ratio_new = _compute_ratio(P_new)
        
        if ratio_new > ratio:
            P = P_new
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.8
            if step_size < 1e-6:
                break
        
        # Random restart if stuck
        if patience > 200:
            P = np.random.dirichlet(np.ones(n * n))
            P = P.reshape(n, n)
            step_size = 0.01
            patience = 0
    
    return _clip_and_normalize(best_P)

# EVOLVE-BLOCK-END
