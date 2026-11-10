# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area(positions, a, y, w):
    """Compute the union area of the 64 triangles given base positions."""
    # L_i(y) = x_i + a_i * y, shape (20000, 64)
    L = positions[None, :] + a[None, :] * y[:, None]
    L_sorted = np.sort(L, axis=1)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
    union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
    return np.mean(union_lengths)

def _keich_positions(n=64):
    """Generate the Keich construction positions."""
    k = int(np.log2(n))
    positions = np.zeros(n, dtype=np.float64)
    j = np.arange(1, k + 1, dtype=np.float64)
    for i in range(n):
        eps = np.zeros(k, dtype=np.float64)
        for bit in range(k):
            if i & (1 << bit):
                eps[bit] = 1.0
        positions[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    return positions

def _subgradient_descent(positions, a, y, w, max_iter, deadline, initial_lr=0.01):
    """Subgradient descent to minimize the area."""
    n = len(positions)
    M = len(y)
    
    best_pos = positions.copy()
    best_area = _compute_area(best_pos, a, y, w)
    lr = initial_lr
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute L = x + a*y
        L = positions[None, :] + a[None, :] * y[:, None]  # (M, n)
        L_sorted = np.sort(L, axis=1)
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # (M, n-1)
        
        # Subgradient computation
        gap_lt_w = (diffs < w[:, None])  # (M, n-1) boolean
        
        # grad_sorted: (M, n)
        grad_sorted = np.zeros_like(L_sorted)
        
        if n > 1:
            # +1 from gap j-1 for j >= 1
            grad_sorted[:, 1:] += gap_lt_w
            # -1 from gap j for j <= n-2
            grad_sorted[:, :-1] -= gap_lt_w
        
        # Map back to original indices
        perm = np.argsort(L, axis=1)  # (M, n)
        
        # grad_original: (M, n)
        grad_original = np.zeros_like(L)
        rows = np.arange(M)[:, None]
        grad_original[rows, perm] = grad_sorted
        
        # The area is mean over m of union_len[m]
        # d(area)/d(x_i) = (1/M) * sum_m grad_original[m, i]
        grad = np.mean(grad_original, axis=0)
        
        # Update
        positions = positions - lr * grad
        area = _compute_area(positions, a, y, w)
        
        if area < best_area:
            best_area = area
            best_pos = positions.copy()
        
        # Decay learning rate
        lr *= 0.995
    
    return best_pos, best_area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses Keich construction as starting point, then refines with subgradient descent
    and coordinate descent for fine-tuning.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    # Precompute fixed values
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    M = 20000
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    
    # Start with Keich construction
    pos = _keich_positions(n)
    best_pos = pos.copy()
    best_area = _compute_area(best_pos, a, y, w)
    
    # Phase 1: Coarse subgradient descent
    max_iter_phase1 = 2000
    pos, area = _subgradient_descent(pos, a, y, w, max_iter_phase1, deadline, initial_lr=0.05)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 10:
        return best_pos
    
    # Phase 2: Fine subgradient descent from best
    pos = best_pos.copy()
    max_iter_phase2 = 5000
    pos, area = _subgradient_descent(pos, a, y, w, max_iter_phase2, deadline, initial_lr=0.01)
    if area < best_area:
        best_area = area
        best_pos = pos.copy()
    
    if time.time() > deadline - 10:
        return best_pos
    
    # Phase 3: Coordinate descent fine-tuning
    step_sizes = [0.001, 0.0005, 0.0001, 0.00005]
    for step in step_sizes:
        if time.time() > deadline - 60:
            break
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 60:
                break
            improved = False
            for idx in range(n):
                if time.time() > deadline - 60:
                    break
                # Try positive and negative steps
                for sign in [1, -1]:
                    x_new = best_pos.copy()
                    x_new[idx] += sign * step
                    area_new = _compute_area(x_new, a, y, w)
                    if area_new < best_area:
                        best_pos = x_new
                        best_area = area_new
                        improved = True
            if not improved:
                break
    
    if time.time() > deadline - 10:
        return best_pos
    
    # Phase 4: Very fine coordinate descent
    step_sizes_fine = [0.00001, 0.000005]
    for step in step_sizes_fine:
        if time.time() > deadline - 30:
            break
        for _ in range(2):
            if time.time() > deadline - 30:
                break
            for idx in range(n):
                if time.time() > deadline - 30:
                    break
                for sign in [1, -1]:
                    x_new = best_pos.copy()
                    x_new[idx] += sign * step
                    area_new = _compute_area(x_new, a, y, w)
                    if area_new < best_area:
                        best_pos = x_new
                        best_area = area_new
    
    return best_pos

# EVOLVE-BLOCK-END
