import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initial guess: Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # Shape (M,)
    
    # Precompute A = a[:, None] * y[None, :] for L calculation
    # L = x[:, None] + A
    A = a[:, None] * y[None, :]
    
    def calc_area_and_grad(x_vals):
        # x_vals: shape (n,)
        L = x_vals[:, None] + A  # Shape (n, M)
        
        # Sort L along axis 0
        # I: indices of sorted elements, shape (n, M)
        # L_sorted: sorted values, shape (n, M)
        I = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, I, axis=0)
        
        # Gaps
        d = L_sorted[1:, :] - L_sorted[:-1, :]  # Shape (n-1, M)
        
        # Determine where gaps are less than w (overlap)
        # w is shape (M,), broadcast to (n-1, M)
        overlap_mask = d < w[None, :]
        
        # Gradient calculation
        # For each height m, the gradient of the union length w.r.t L_sorted[k, m] is:
        # -1 if k is the left interval of an overlapping pair
        # +1 if k is the right interval of an overlapping pair
        # 0 otherwise (or if it's the first/last and no overlap on that side, etc.)
        
        # Let's accumulate gradients for L_sorted
        # grad_L_sorted shape (n, M)
        grad_L_sorted = np.zeros_like(L)
        
        # For each gap k (0 to n-2), if overlap_mask[k, m] is True:
        # grad_L_sorted[k, m] -= 1  (left interval)
        # grad_L_sorted[k+1, m] += 1 (right interval)
        
        # We can do this with vectorized operations using np.add.at or similar, 
        # but since n is small (64), a loop over n-1 might be faster or comparable.
        # Actually, we can use np.where and sums.
        
        # Let's construct the gradient for the sorted positions directly.
        # For a specific column m:
        # If d[k, m] < w[m], then L_sorted[k, m] gets -1 and L_sorted[k+1, m] gets +1.
        
        # grad_L_sorted[k, :] = -sum(overlap_mask[k, :]) for k in 0..n-2 ? No, it's per column.
        # We need to scatter the gradients back to the original indices.
        
        # Let's compute grad_L_sorted (n, M)
        # Initialize to 0
        # For k in 0..n-2:
        #   mask = overlap_mask[k, :]
        #   grad_L_sorted[k, mask] -= 1
        #   grad_L_sorted[k+1, mask] += 1
        
        # This loop is 63 iterations, each operating on (M,) arrays. This is fast.
        for k in range(n - 1):
            mask = overlap_mask[k, :]
            grad_L_sorted[k, mask] -= 1.0
            grad_L_sorted[k + 1, mask] += 1.0
            
        # Now grad_L_sorted contains d(U)/d(L_sorted)
        # We need d(U)/d(x_i).
        # L_i = x_i + A_i. So d(L_i)/d(x_i) = 1.
        # But L_sorted is a permutation of L.
        # So d(U)/d(x_i) = sum over m of d(U)/d(L_i, m)
        # d(U)/d(L_i, m) is the value in grad_L_sorted at the position where L_i was in the sorted order.
        
        # We can use np.add.at to accumulate gradients back to original indices.
        # grad_x shape (n,)
        grad_x = np.zeros(n)
        
        # For each column m, for each k, the gradient grad_L_sorted[k, m] belongs to x_{I[k, m]}
        # So we can do:
        # np.add.at(grad_x, I, grad_L_sorted) -- but this adds the whole matrix?
        # No, we want sum over m for each i.
        
        # Let's use a loop over columns? No, M is 20000.
        # Let's use np.add.at with proper indexing.
        # We want to sum grad_L_sorted[k, m] into grad_x[I[k, m]] for all k, m.
        
        # Efficient way:
        # Create a flat array of indices and values
        # But np.add.at is slow for large arrays.
        
        # Alternative:
        # grad_L_sorted is (n, M). I is (n, M).
        # We can use a histogram-like approach or just scatter.
        
        # Since n is small, let's iterate over i and sum the contributions.
        # For a fixed i, find all (k, m) such that I[k, m] == i.
        # This is expensive.
        
        # Better: Use np.add.at
        # It might be slow, but let's try.
        # Actually, we can compute the gradient analytically for each x_i?
        
        # Let's stick to the scatter approach but optimize it.
        # We can transpose the problem?
        
        # Let's try a different way to compute grad_x.
        # grad_x[i] = sum_m grad_L_sorted[k, m] where I[k, m] == i.
        
        # We can use np.bincount on flattened indices?
        # Indices are I flattened. Values are grad_L_sorted flattened.
        # But we need to sum by original index i.
        
        # Let's use np.add.at. It is the standard way.
        np.add.at(grad_x, I, grad_L_sorted)
        
        # The area is the mean of union lengths.
        # Union length U(y) = w + sum(min(d, w))
        # We already have d and w.
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        
        # The gradient of the area is the mean of the gradients of the union lengths.
        # So we need to divide grad_x by M.
        grad_x /= M
        
        return area, grad_x

    # Optimization
    start_time = time.time()
    timeout = 600
    safety_margin = 10
    
    # Use L-BFGS-B
    # x0 is the initial guess
    # We need to define the objective and gradient functions for scipy
    def obj_func(x_vals):
        area, _ = calc_area_and_grad(x_vals)
        return area
    
    def grad_func(x_vals):
        _, grad = calc_area_and_grad(x_vals)
        return grad

    # To avoid recomputing area and grad separately, we can use a wrapper or just accept the cost.
    # L-BFGS-B will call obj and grad.
    
    # Let's try to run the optimization.
    # Bounds: x_i should be in a reasonable range.
    # The triangles are in [0, 1] roughly.
    # Let's use bounds (-1, 2) to be safe.
    bounds = [(-1.0, 2.0) for _ in range(n)]
    
    try:
        res = scipy.optimize.minimize(
            obj_func,
            x,
            method='L-BFGS-B',
            jac=grad_func,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < obj_func(x):
            x = res.x
            
    except Exception as e:
        # If optimization fails, keep original x
        pass

    # Check time
    if time.time() - start_time > timeout - safety_margin:
        return x

    # If we have time left, we can do a second pass or refine.
    # But L-BFGS-B should be good.
    # Let's do a quick check if we can improve further with a different initial guess or more iterations.
    
    # Actually, the gradient calculation using np.add.at might be a bottleneck.
    # Let's profile mentally:
    # L: (64, 20000)
    # argsort: O(M * n log n) = 20000 * 64 * 6 ~ 7.6M ops.
    # Loop 63 times:
    #   mask: (20000,)
    #   grad_L_sorted[k, mask] -= 1: 20000 ops.
    #   Total: 63 * 40000 ~ 2.5M ops.
    # np.add.at: (64, 20000) -> (64,). This is a scatter operation.
    # It should be fast enough.
    
    # Let's return the result.
    return x
