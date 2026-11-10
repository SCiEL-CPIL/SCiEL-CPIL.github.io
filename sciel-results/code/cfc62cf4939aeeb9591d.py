# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimizes the area of the union of 64 Kakeya needle triangles.
    Uses the Keich construction as an initial guess and refines it using
    L-BFGS-B with an analytic subgradient.
    """
    n = 64
    k = int(np.log2(n))
    
    # 1. Generate Keich Baseline
    # a_i = (i+1)/n
    a = np.array([(i + 1) / n for i in range(n)], dtype=np.float64)
    
    # Keich construction: x_i = sum_{j=1}^{k} ((1 - j) / k) * eps_j * 2^{-j}
    # where eps_j is the j-th bit of i (1-indexed from MSB? No, bin(i) is MSB first)
    # The seed code used:
    # eps = np.array(list(str(bin(i))[2:]), dtype=int)
    # eps = np.pad(eps, (k - len(eps), 0), "constant")
    # positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    # Note: j in seed code was 1..k. eps[0] corresponds to 2^(k-1) bit?
    # Let's stick to the seed's exact logic to ensure we start at the same point.
    
    j_idx = np.arange(1, k + 1) # 1 to k
    weights = (1 - j_idx) * (2.0 ** -j_idx) / k
    
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bin_str = bin(i)[2:]
        eps = np.array(list(bin_str), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum(eps * weights)
        
    # 2. Setup Quadrature Points
    # y_m = (m + 0.5) / 20000
    M = 20000
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    
    # Precompute a * y for each i? No, L_i = x_i + a_i * y.
    # We can compute L as a matrix of shape (M, n)
    # L[m, i] = x[i] + a[i] * y[m]
    
    def area_and_grad(x):
        # x is shape (n,)
        # L is shape (M, n)
        L = x[None, :] + (a * y[:, None])
        
        # Sort L along axis 1
        # We need the sorted values and the indices to compute the gradient correctly
        sorted_L = np.sort(L, axis=1)
        
        # Gaps
        gaps = sorted_L[:, 1:] - sorted_L[:, :-1]
        
        # Union length at each y
        # U[y] = w[y] + sum(min(gaps, w[y]))
        # min(gaps, w) = gaps where gaps < w, else w
        # Since w is a vector of shape (M,), we need to broadcast
        min_gaps = np.minimum(gaps, w[:, None])
        union_lengths = w + np.sum(min_gaps, axis=1)
        
        area = np.mean(union_lengths)
        
        # Gradient computation
        # d U[y] / d x[k] = sum_{j: gap_j < w[y]} (delta_{k, idx_{j+1}} a_{idx_{j+1}} - delta_{k, idx_j} a_{idx_j})
        # where idx_j is the index in original x of the j-th smallest L.
        
        # Get indices of sorted L
        # argsort returns indices that would sort L
        idx = np.argsort(L, axis=1) # shape (M, n)
        
        # We need to accumulate gradients for each k
        # grad[k] = sum_{m} (1/M) * sum_{j} (indicator(gaps[m,j] < w[m]) * (a[idx[m, j+1]] * delta_{k, idx[m, j+1]} - a[idx[m, j]] * delta_{k, idx[m, j]}))
        
        # This can be vectorized.
        # Let's construct a matrix G of shape (M, n) for the gradient contribution at each y.
        # For a fixed y (row m), the gradient is a vector of size n.
        # The term for gap j (between sorted pos j and j+1) contributes:
        # + a[idx[m, j+1]] to component idx[m, j+1]
        # - a[idx[m, j]] to component idx[m, j]
        # IF gaps[m, j] < w[m]
        
        # Initialize gradient matrix
        grad_M = np.zeros((M, n), dtype=np.float64)
        
        # We need to add to grad_M[m, idx[m, j+1]] the value a[idx[m, j+1]] if gap < w
        # And subtract a[idx[m, j]] from grad_M[m, idx[m, j]] if gap < w
        
        # Mask for gaps < w
        mask = gaps < w[:, None] # shape (M, n-1)
        
        # Get the indices and values
        # We need to scatter-add
        # This is tricky with pure NumPy without loops or advanced indexing tricks.
        # Loop over M might be too slow (20000 iterations).
        # Let's try to vectorize the scatter.
        
        # Create an array of indices for the "plus" part
        # For each m, for each j where mask[m,j] is True, add a[idx[m, j+1]] to grad_M[m, idx[m, j+1]]
        
        # Let's flatten the mask
        # This approach is complex. Let's consider if L-BFGS-B can work with finite differences.
        # Finite diff cost: 64 evaluations of area.
        # Area eval cost: Sort 20000x64 matrix. ~ 20000 * 64 * log(64) ops. ~ 10^6 ops.
        # 64 * 10^6 = 6 * 10^7 ops per gradient step.
        # L-BFGS-B takes ~ 50-100 steps. 10^9 ops. Might be too slow in Python/NumPy if not optimized.
        
        # Let's optimize the area evaluation first.
        # np.sort on (20000, 64) is fast.
        
        # Let's try to implement the gradient more efficiently or just use finite diff if time permits.
        # Given the 600s limit, and Python overhead, finite diff might be risky.
        # However, the Keich point is very good. Maybe only a few steps are needed.
        
        # Alternative: Coordinate Descent with analytic 1D minimization?
        # The function is convex in each x_i if others are fixed.
        # For fixed x_j (j!=i), U(y) is piecewise linear in x_i.
        # Breakpoints occur when x_i + a_i y = x_j + a_j y (intersection)
        # or when the gap becomes w.
        
        # Let's stick to the plan: Use L-BFGS-B with finite differences initially to see if it's fast enough.
        # If not, I will switch to a simpler refinement like Nelder-Mead or just return Keich if optimization is too slow.
        
        # To be safe and fast, let's implement a simple coordinate descent step manually if scipy is too slow.
        # But the directive suggests scipy.
        
        # Let's provide the gradient using a vectorized approach if possible.
        # Actually, `scipy.optimize.minimize` with `method='L-BFGS-B'` and `jac=None` uses finite differences.
        # Let's test the speed of `area_and_grad` (just the area part) in a loop.
        
        # For now, return area and None (let scipy do finite diff).
        # To make it faster, we can reduce M for optimization and use full M for final check?
        # The score is computed with M=20000. We must optimize for M=20000.
        
        # Let's try to implement the gradient properly.
        # Scatter add in NumPy:
        # np.add.at(grad_M, (row_indices, col_indices), values)
        
        row_idx = np.repeat(np.arange(M), n-1)
        # We need to filter by mask
        # mask is (M, n-1)
        # Let's flatten mask
        flat_mask = mask.flatten()
        
        # Get indices where mask is True
        true_indices = np.where(flat_mask)[0]
        
        # Map flat index to (m, j)
        m_idx, j_idx = np.unravel_index(true_indices, (M, n-1))
        
        # For each (m, j), we add to grad_M[m, idx[m, j+1]] the value a[idx[m, j+1]]
        # And subtract from grad_M[m, idx[m, j]] the value a[idx[m, j]]
        
        # Plus part
        plus_col = idx[m_idx, j_idx + 1]
        plus_val = a[plus_col]
        np.add.at(grad_M, (m_idx, plus_col), plus_val)
        
        # Minus part
        minus_col = idx[m_idx, j_idx]
        minus_val = a[minus_col]
        np.add.at(grad_M, (m_idx, minus_col), -minus_val)
        
        # Average over M
        grad = np.mean(grad_M, axis=0)
        
        return area, grad

    # Run optimization
    # Start from x0
    # Use L-BFGS-B
    # maxiter=200 should be enough for refinement
    # ftol=1e-9
    
    start_time = time.time()
    deadline = start_time + 500  # 500s limit to leave margin
    
    try:
        res = scipy.optimize.minimize(
            area_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, # We provide gradient
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x_opt = res.x
    except Exception:
        x_opt = x0

    # Check time
    if time.time() > deadline:
        return x0

    # Final verification: ensure finite and valid
    if not np.all(np.isfinite(x_opt)):
        return x0

    return x_opt

# EVOLVE-BLOCK-END
