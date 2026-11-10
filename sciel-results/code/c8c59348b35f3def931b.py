# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_area_fast(x, n=64, M=20000):
    """Compute only the area (no gradient) for evaluation."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    return area


def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    
    # Gradient computation
    # For each y, the union length is w + sum_j min(gap_j, w)
    # d(union)/d(x_i) = sum_j d(min(gap_j, w))/d(x_i)
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # d(gap_j)/d(x_i) = 1 if x_i is L_sorted[j+1], -1 if x_i is L_sorted[j], 0 otherwise
    # d(min(gap_j, w))/d(x_i) = d(gap_j)/d(x_i) if gap_j < w, else 0
    
    grad = np.zeros(n)
    for j in range(n - 1):
        mask = gaps[j, :] < w[None, :]
        if not np.any(mask):
            continue
        lower_idx = idx[j, mask]
        upper_idx = idx[j + 1, mask]
        count_mask = mask.sum()
        # For each triangle i, count how many times it appears as upper or lower
        # Use bincount for efficiency
        grad[upper_idx] += 1.0
        grad[lower_idx] -= 1.0
    
    # Divide by M since area is mean
    grad /= M
    return area, grad


def _keich_initial(n=64):
    """Generate the Keich construction initial point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _basin_hopping(x0, n=64, M=20000, deadline=None, max_hops=200, 
                   hop_sizes=[0.1, 0.05, 0.02, 0.01], polish_maxiter=100):
    """
    Basin hopping with structured perturbations and L-BFGS polish.
    """
    x = x0.copy()
    best_x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    
    rng = np.random.default_rng(67890)
    
    for hop in range(max_hops):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        # Generate structured perturbation
        pert_type = rng.integers(0, 4)
        x_perturbed = x.copy()
        
        if pert_type == 0:
            # Subset shift: move a random subset by a random amount
            subset_size = rng.integers(4, n // 2)
            indices = rng.choice(n, size=subset_size, replace=False)
            shift = rng.normal(0, 0.05, size=subset_size)
            x_perturbed[indices] += shift
        elif pert_type == 1:
            # Pairwise swap: swap positions of two random subsets
            half = n // 2
            idx1 = rng.choice(n, size=half, replace=False)
            remaining = np.setdiff1d(np.arange(n), idx1)
            idx2 = rng.choice(remaining, size=half, replace=False)
            x_perturbed[idx1], x_perturbed[idx2] = x_perturbed[idx2].copy(), x_perturbed[idx1].copy()
        elif pert_type == 2:
            # Gaussian noise
            x_perturbed += rng.normal(0, 0.03, size=n)
        else:
            # Scale and shift
            x_perturbed = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)
        
        # Polish with L-BFGS
        try:
            result = minimize(
                _compute_area_and_grad,
                x_perturbed,
                args=(n, M),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': polish_maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_area:
                new_area = result.fun
                if new_area < best_area:
                    best_area = new_area
                    best_x = result.x.copy()
                    x = result.x.copy()
        except Exception:
            pass
    
    return best_x, best_area


def _coordinate_descent_polish(x, n=64, M=20000, deadline=None, max_rounds=50):
    """
    Coordinate descent with adaptive step sizes for final polish.
    """
    x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    L_base = a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    def _area_with_xi(xi_val, xi_idx):
        L = x[:, None] + L_base
        L[xi_idx, :] = xi_val + L_base[xi_idx, :]
        idx = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, idx, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
        return np.mean(union_lengths)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    for round_idx in range(max_rounds):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline - 15:
                break
            
            current_xi = x[i]
            best_xi = current_xi
            best_local_area = best_area
            
            for step in step_sizes:
                for direction in [1, -1]:
                    new_xi = current_xi + direction * step
                    new_area = _area_with_xi(new_xi, i)
                    if new_area < best_local_area - 1e-15:
                        best_local_area = new_area
                        best_xi = new_xi
            
            if best_xi != current_xi:
                x[i] = best_xi
                best_area = best_local_area
                improved = True
        
        if not improved:
            break
    
    return x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses basin hopping with structured perturbations and L-BFGS polish,
    followed by coordinate descent for final refinement.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    n = 64
    M = 20000
    
    # Start from Keich construction
    x_keich = _keich_initial(n)
    
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    # Generate multiple initial points
    rng = np.random.default_rng(42)
    initial_points = [x_keich.copy()]
    
    # Add perturbed versions of Keich
    for scale in [0.05, 0.1, 0.2]:
        initial_points.append(x_keich + rng.normal(0, scale, size=n))
    
    # Add random points
    for _ in range(5):
        initial_points.append(rng.uniform(-0.5, 0.5, size=n))
    
    # Run basin hopping from each initial point
    for init_idx, x0 in enumerate(initial_points):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - elapsed
        max_hops = max(20, int(remaining / 10))
        
        try:
            bh_x, bh_area = _basin_hopping(
                x0, n, M, deadline=deadline,
                max_hops=max_hops, polish_maxiter=80
            )
            if bh_area < best_area:
                best_area = bh_area
                best_x = bh_x.copy()
        except Exception:
            continue
    
    # Final coordinate descent polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            refined_x, refined_area = _coordinate_descent_polish(
                best_x, n, M, deadline=deadline, max_rounds=100
            )
            if refined_area < best_area:
                best_area = refined_area
                best_x = refined_x.copy()
        except Exception:
            pass
    
    # One more basin hopping round if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 45:
        try:
            bh_x2, bh_area2 = _basin_hopping(
                best_x, n, M, deadline=deadline,
                max_hops=50, polish_maxiter=60
            )
            if bh_area2 < best_area:
                best_area = bh_area2
                best_x = bh_x2.copy()
            
            # Final polish
            if time.time() < deadline - 20:
                refined_x2, refined_area2 = _coordinate_descent_polish(
                    best_x, n, M, deadline=deadline, max_rounds=50
                )
                if refined_area2 < best_area:
                    best_area = refined_area2
                    best_x = refined_x2.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
