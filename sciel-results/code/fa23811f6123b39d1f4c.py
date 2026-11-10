# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_FRACTION = 0.90
_START_TIME = None


def _keich_positions(n=64):
    """Classical Keich construction for n a power of 2."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return positions


def _compute_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using quadrature at M heights.
    Uses the identity: union_length = max(L) - min(L) + w*(n-1) - sum_j max(gap_j, w)
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    L = x[:, None] + a[:, None] * y[None, :]  # (n, M)
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    
    # union = max - min + w*(n-1) - sum(max(gaps, w))
    union_lengths = (L_sorted[-1, :] - L_sorted[0, :]) + w * (n - 1) - np.sum(np.maximum(gaps, w[None, :]), axis=0)
    
    area = np.mean(union_lengths)
    return area


def _compute_area_and_subgrad(x, n=64, M=20000):
    """
    Compute area and subgradient w.r.t. x using the fast identity.
    Returns (area, subgrad) where subgrad has shape (n,).
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # (M,)
    
    L = x[:, None] + a[:, None] * y[None, :]  # (n, M)
    order = np.argsort(L, axis=0)  # (n, M)
    L_sorted = np.take_along_axis(L, order, axis=0)  # (n, M)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    
    # union = max - min + w*(n-1) - sum(max(gaps, w))
    union_lengths = (L_sorted[-1, :] - L_sorted[0, :]) + w * (n - 1) - np.sum(np.maximum(gaps, w[None, :]), axis=0)
    area = np.mean(union_lengths)
    
    # Subgradient:
    # d(union)/dL_i for each i
    # union = L_max - L_min + w*(n-1) - sum_j max(gap_j, w)
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # d(union)/dL_sorted[j] = -1 if j is argmin, +1 if j is argmax
    # For each gap_j: if gap_j > w, d/dL_sorted[j] -= 1, d/dL_sorted[j+1] += 1
    #                 if gap_j < w, no contribution (clamped)
    
    dL_sorted = np.zeros_like(L_sorted)  # (n, M)
    
    # min and max contributions
    min_idx = np.argmin(L_sorted, axis=0)  # (M,)
    max_idx = np.argmax(L_sorted, axis=0)  # (M,)
    dL_sorted[min_idx, np.arange(M)] -= 1.0
    dL_sorted[max_idx, np.arange(M)] += 1.0
    
    # gap contributions
    above_w = gaps > w[None, :]  # (n-1, M)
    dL_sorted[:-1, :] -= above_w  # d/dL_sorted[j] for gap j
    dL_sorted[1:, :] += above_w   # d/dL_sorted[j+1] for gap j
    
    # Map back to original indices
    dL_orig = np.zeros_like(L)  # (n, M)
    np.put_along_axis(dL_orig, order, dL_sorted, axis=0)
    
    # d(union)/dx_i = d(union)/dL_i since L_i = x_i + a_i*y
    subgrad = dL_orig.mean(axis=1)  # (n,)
    
    return area, subgrad


def _compute_soft_area_and_grad(x, n=64, M=20000, beta=50.0):
    """
    Compute a smooth surrogate for the area and its gradient.
    Uses soft-min/soft-max to handle the non-smooth max(gap, w) term.
    soft_max(a, b) = log(exp(beta*a) + exp(beta*b)) / beta
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # (M,)
    
    L = x[:, None] + a[:, None] * y[None, :]  # (n, M)
    order = np.argsort(L, axis=0)  # (n, M)
    L_sorted = np.take_along_axis(L, order, axis=0)  # (n, M)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    
    # Soft max of gaps and w: soft_max(gap, w) ≈ log(exp(beta*gap) + exp(beta*w)) / beta
    # But we need to be careful with numerical stability. Use the log-sum-exp trick.
    # soft_max(gap, w) = max(gap, w) + log(1 + exp(-beta*|gap - w|)) / beta
    diff = gaps - w[None, :]  # (n-1, M)
    abs_diff = np.abs(diff)
    # For large beta*abs_diff, log(1 + exp(-beta*abs_diff)) ≈ 0
    # For small beta*abs_diff, log(1 + exp(-beta*abs_diff)) ≈ log(2)
    log_term = np.log1p(np.exp(-beta * abs_diff)) / beta
    soft_max_gaps = np.maximum(gaps, w[None, :]) + log_term  # (n-1, M)
    
    # union_soft = max - min + w*(n-1) - sum(soft_max_gaps)
    union_soft = (L_sorted[-1, :] - L_sorted[0, :]) + w * (n - 1) - np.sum(soft_max_gaps, axis=0)
    area_soft = np.mean(union_soft)
    
    # Gradient w.r.t. L_sorted
    dL_sorted = np.zeros_like(L_sorted)  # (n, M)
    
    # min and max contributions (same as before)
    min_idx = np.argmin(L_sorted, axis=0)  # (M,)
    max_idx = np.argmax(L_sorted, axis=0)  # (M,)
    dL_sorted[min_idx, np.arange(M)] -= 1.0
    dL_sorted[max_idx, np.arange(M)] += 1.0
    
    # Gradient of soft_max(gap, w) w.r.t. gap:
    # d/dgap [max(gap, w) + log(1 + exp(-beta*|gap - w|)) / beta]
    # If gap > w: 1 + 0 = 1
    # If gap < w: 0 + (-exp(-beta*(w-gap)) / (1 + exp(-beta*(w-gap)))) * (-1) / beta * beta
    #            = exp(-beta*(w-gap)) / (1 + exp(-beta*(w-gap)))
    # This is the sigmoid: sigma(beta*(gap - w))
    # So d(soft_max)/d(gap) = sigma(beta*(gap - w))
    # And d(soft_max)/d(w) = 1 - sigma(beta*(gap - w))
    
    # But we only need d/d(gap) since w is fixed.
    # d(union_soft)/d(gap_j) = -sigma(beta*(gap_j - w))
    sigmoid_term = 1.0 / (1.0 + np.exp(-beta * diff))  # (n-1, M)
    dL_sorted[:-1, :] -= sigmoid_term  # d/dL_sorted[j] for gap j
    dL_sorted[1:, :] += sigmoid_term   # d/dL_sorted[j+1] for gap j
    
    # Map back to original indices
    dL_orig = np.zeros_like(L)  # (n, M)
    np.put_along_axis(dL_orig, order, dL_sorted, axis=0)
    
    # d(area_soft)/dx_i = mean over m of dL_orig[i, m]
    grad = dL_orig.mean(axis=1)  # (n,)
    
    return area_soft, grad


def _coordinate_descent(x, n=64, M=20000, deadline=None, max_cd_iters=500):
    """
    Coordinate descent to minimize area.
    """
    best_area = _compute_area(x, n, M)
    best_x = x.copy()
    w = 1.0 / n
    
    for cd_iter in range(max_cd_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try different values for x[i]
            # Current value and neighbors
            candidates = [x[i] - 0.01, x[i] - 0.005, x[i] - 0.001, x[i], 
                         x[i] + 0.001, x[i] + 0.005, x[i] + 0.01]
            
            best_cand_val = x[i]
            best_cand_area = best_area
            
            for cv in candidates:
                test_x = x.copy()
                test_x[i] = cv
                test_area = _compute_area(test_x, n, M)
                if test_area < best_cand_area:
                    best_cand_area = test_area
                    best_cand_val = cv
            
            if best_cand_val != x[i]:
                x[i] = best_cand_val
                best_area = best_cand_area
                best_x = x.copy()
                improved = True
        
        if not improved:
            break
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = 64
    M = 20000
    deadline = _START_TIME + 630.0 * _DEADLINE_FRACTION
    
    x = _keich_positions(n)
    best_x = x.copy()
    best_area = _compute_area(x, n, M)
    
    # Phase 1: Smooth gradient descent with adaptive beta
    step = 0.02
    min_step = 1e-8
    max_iter = 100000
    no_improve_count = 0
    patience = 300
    beta = 20.0  # Start with moderate smoothing
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Use smooth gradient for direction, but evaluate true area
        area_soft, grad = _compute_soft_area_and_grad(x, n, M, beta=beta)
        
        # Evaluate true area for tracking
        true_area = _compute_area(x, n, M)
        if true_area < best_area - 1e-12:
            best_area = true_area
            best_x = x.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Gradually increase beta to sharpen the approximation
        if it % 50 == 0 and beta < 200.0:
            beta = min(200.0, beta * 1.5)
        
        if no_improve_count > patience:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        direction = grad / grad_norm
        
        # Line search: try several step sizes
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            cand = x - step * scale * direction
            cand_area = _compute_area(cand, n, M)
            if cand_area < true_area:
                x = cand
                improved = True
                break
        
        if not improved:
            step *= 0.7
            if step < min_step:
                break
        else:
            if it % 100 == 0 and step < 0.02:
                step = min(0.02, step * 1.3)
    
    # Phase 2: Subgradient descent with true subgradient for fine-tuning
    if time.time() < deadline - 60:
        x_sg = x.copy()
        step = 0.005
        min_step = 1e-10
        max_iter_sg = 50000
        no_improve_count = 0
        patience_sg = 100
        
        for it in range(max_iter_sg):
            if time.time() > deadline - 30:
                break
            
            area, subgrad = _compute_area_and_subgrad(x_sg, n, M)
            if area < best_area - 1e-12:
                best_area = area
                best_x = x_sg.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > patience_sg:
                break
            
            sg_norm = np.linalg.norm(subgrad)
            if sg_norm < 1e-12:
                break
            
            direction = subgrad / sg_norm
            
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                cand = x_sg - step * scale * direction
                cand_area = _compute_area(cand, n, M)
                if cand_area < area:
                    x_sg = cand
                    improved = True
                    break
            
            if not improved:
                step *= 0.7
                if step < min_step:
                    break
            else:
                if it % 50 == 0 and step < 0.005:
                    step = min(0.005, step * 1.2)
    
    # Phase 3: Coordinate descent refinement
    if time.time() < deadline - 30:
        x_cd, area_cd = _coordinate_descent(x, n, M, deadline=deadline)
        if area_cd < best_area:
            best_x = x_cd.copy()
            best_area = area_cd
    
    # Final validation
    if np.any(~np.isfinite(best_x)) or best_area <= 0:
        return _keich_positions(n)
    
    return best_x


# EVOLVE-BLOCK-END
