# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the area of the union of 64 Kakeya-needle triangles.
    
    The current best known approach uses:
    1. A soft-min surrogate to smooth the nonsmooth union metric
    2. Basin-hopping with structured perturbations (subset shifts + pairwise swaps)
    3. Final Nelder-Mead polish with tight tolerances
    
    This implementation starts from the Keich construction and refines it.
    """
    n = 64
    w = 1.0 / n
    M = 20000  # quadrature heights
    
    # Deadline guard
    deadline = time.time() + 580  # 20s safety margin
    
    # Precompute quadrature heights and weights
    m = np.arange(M)
    y = (m + 0.5) / M
    
    # a_i for each triangle
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    
    def compute_area(x):
        """
        Compute the area of the union of triangles given base positions x.
        x: array of shape (n,)
        Returns: area (scalar)
        """
        # L_i = x_i + a_i * y for each height y
        # We compute this as: L[i, m] = x[i] + a[i] * y[m]
        L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
        
        # Width at height y: w(y) = (1 - y) / n
        widths = (1.0 - y) / n  # shape (M,)
        
        # Sort L at each height
        L_sorted = np.sort(L, axis=0)  # shape (n, M)
        
        # Union length at each height:
        # w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
        # where w is the width at that height
        gaps = np.diff(L_sorted, axis=0)  # shape (n-1, M)
        min_gaps = np.minimum(gaps, widths[None, :])  # shape (n-1, M)
        
        # Union length = w + sum of min_gaps
        union_lengths = widths + np.sum(min_gaps, axis=0)  # shape (M,)
        
        # Area = mean of union lengths
        area = np.mean(union_lengths)
        return area
    
    def soft_min(gaps, widths, beta):
        """
        Soft-min surrogate: -1/beta * log(sum(exp(-beta * min(gaps, widths))))
        Actually, we want to smooth min(gaps, widths).
        Use: min(a,b) ≈ -1/beta * log(exp(-beta*a) + exp(-beta*b))
        """
        # min(gaps, widths) where gaps has shape (n-1, M), widths has shape (M,)
        # We want: min(gaps, widths) ≈ -1/beta * log(exp(-beta*gaps) + exp(-beta*widths))
        # But this is not quite right. Let's think...
        # min(a, b) = -max(-a, -b)
        # max(-a, -b) ≈ 1/beta * log(exp(-beta*(-a)) + exp(-beta*(-b))) = 1/beta * log(exp(beta*a) + exp(beta*b))
        # So min(a,b) ≈ -1/beta * log(exp(beta*a) + exp(beta*b))
        # But for numerical stability, we should use log-sum-exp trick
        
        # Actually, let's use a different approach:
        # min(gaps, widths) = gaps - max(0, gaps - widths)
        # max(0, z) ≈ 1/beta * log(1 + exp(beta*z))
        # So min(gaps, widths) ≈ gaps - 1/beta * log(1 + exp(beta*(gaps - widths)))
        
        z = gaps - widths[None, :]  # shape (n-1, M)
        # max(0, z) ≈ 1/beta * log(1 + exp(beta*z))
        # For numerical stability:
        # log(1 + exp(beta*z)) = beta*z + log(1 + exp(-beta*z)) when z > 0
        # = log(1 + exp(beta*z)) when z <= 0
        
        # Use the stable formula
        beta_z = beta * z
        max_0_z = np.where(
            beta_z > 0,
            beta_z + np.log1p(np.exp(-beta_z)),
            np.log1p(np.exp(beta_z))
        ) / beta
        
        min_gaps_soft = gaps - max_0_z
        return min_gaps_soft
    
    def soft_area(x, beta):
        """
        Compute the soft area using soft-min surrogate.
        """
        L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
        widths = (1.0 - y) / n  # shape (M,)
        
        L_sorted = np.sort(L, axis=0)  # shape (n, M)
        gaps = np.diff(L_sorted, axis=0)  # shape (n-1, M)
        
        min_gaps_soft = soft_min(gaps, widths, beta)
        union_lengths = widths + np.sum(min_gaps_soft, axis=0)
        area = np.mean(union_lengths)
        return area
    
    def gradient_soft_area(x, beta):
        """
        Compute the gradient of the soft area with respect to x.
        This requires careful chain rule through the sorting.
        
        For now, use finite differences for simplicity, but this is slow.
        Let's use a smarter approach: the gradient of the union length with respect to
        each x_i can be computed analytically if we track which triangles are in the union.
        
        Actually, let's use a simpler approach: since the problem is permutation-invariant,
        we can use the fact that the gradient of the sorted union length with respect to
        the unsorted inputs is just the gradient permuted back.
        
        For the soft version, let's use automatic differentiation via finite differences
        with a small step, or use the analytical gradient.
        
        For now, let's use a simple approach: compute the gradient numerically.
        """
        # Use central differences
        h = 1e-6
        grad = np.zeros_like(x)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (soft_area(x_plus, beta) - soft_area(x_minus, beta)) / (2 * h)
        return grad
    
    # Start from Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    best_x = x.copy()
    best_area = compute_area(x)
    
    # Phase 1: Soft-min refinement with adaptive beta
    beta_start = 20.0
    beta_end = 200.0
    n_beta_steps = 5
    
    for b_step in range(n_beta_steps):
        if time.time() > deadline - 60:
            break
        beta = beta_start * (beta_end / beta_start) ** (b_step / max(n_beta_steps - 1, 1))
        
        # Do a few gradient descent steps
        for gd_iter in range(100):
            if time.time() > deadline - 60:
                break
            
            grad = gradient_soft_area(x, beta)
            # Adaptive step size
            step_size = 0.01 / (1 + gd_iter * 0.01)
            x_new = x - step_size * grad
            
            # Check if this improves the true area
            area_new = compute_area(x_new)
            if area_new < best_area:
                x = x_new
                best_x = x.copy()
                best_area = area_new
            else:
                # Try smaller step
                step_size *= 0.5
                x_new = x - step_size * grad
                area_new = compute_area(x_new)
                if area_new < best_area:
                    x = x_new
                    best_x = x.copy()
                    best_area = area_new
                else:
                    break
    
    # Phase 2: Basin hopping with structured perturbations
    if time.time() < deadline - 120:
        x = best_x.copy()
        best_x = x.copy()
        best_area = compute_area(x)
        
        n_hops = 50
        for hop in range(n_hops):
            if time.time() > deadline - 60:
                break
            
            # Structured perturbation: subset shifts + pairwise swaps
            x_perturbed = x.copy()
            
            # Subset shifts
            n_subsets = 3
            for _ in range(n_subsets):
                subset_size = np.random.randint(1, n // 4 + 1)
                subset = np.random.choice(n, subset_size, replace=False)
                shift = np.random.normal(0, 0.05)
                x_perturbed[subset] += shift
            
            # Pairwise swaps (swap positions of two triangles)
            n_swaps = 2
            for _ in range(n_swaps):
                i, j_idx = np.random.choice(n, 2, replace=False)
                x_perturbed[[i, j_idx]] = x_perturbed[[j_idx, i]]
            
            # Local refinement of the perturbed point
            x_local = x_perturbed.copy()
            for local_iter in range(50):
                if time.time() > deadline - 30:
                    break
                
                # Use Nelder-Mead style or simple gradient step
                # For simplicity, use a small gradient step
                grad = gradient_soft_area(x_local, beta_end)
                step_size = 0.005
                x_candidate = x_local - step_size * grad
                area_candidate = compute_area(x_candidate)
                
                if area_candidate < compute_area(x_local):
                    x_local = x_candidate
                else:
                    break
            
            area_local = compute_area(x_local)
            if area_local < best_area:
                best_x = x_local.copy()
                best_area = area_local
                x = x_local.copy()
    
    # Phase 3: Final Nelder-Mead polish
    if time.time() < deadline - 30:
        from scipy.optimize import minimize
        
        def objective(x_flat):
            return compute_area(x_flat)
        
        # Use Nelder-Mead with tight tolerances
        result = minimize(
            objective,
            best_x,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if result.fun < best_area:
            best_x = result.x
            best_area = result.fun
    
    # Ensure finite values
    if not np.all(np.isfinite(best_x)):
        # Fallback to Keich construction
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        best_x = np.array(positions, dtype=np.float64)
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
