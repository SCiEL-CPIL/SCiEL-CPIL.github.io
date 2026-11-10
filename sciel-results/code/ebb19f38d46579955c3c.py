# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_union_stats(base, motion, c_vals):
    """
    Compute union length and gradients for a set of c values.
    base: (K,) array
    motion: (K,) array
    c_vals: (M,) array
    Returns:
    lengths: (M,) array of union lengths
    grads: (M, K) array of d(Union Length)/d(base[k])
    """
    K = len(base)
    M = len(c_vals)
    width = 1.0 / 8.0
    
    # We need to sort intervals by start point for each c.
    # start_k = base[k] + c * motion[k]
    # end_k = start_k + width
    
    # To avoid M separate sorts, we can't easily vectorize the sort 
    # because the order changes with c.
    # However, K=8 is small. M can be large.
    # We will loop over M, but optimize the inner loop.
    
    lengths = np.zeros(M)
    grads = np.zeros((M, K))
    
    for m in range(M):
        c = c_vals[m]
        starts = base + c * motion
        ends = starts + width
        
        # Sort indices by start position
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Calculate union length and determine which intervals contribute
        # An interval contributes 1 to the derivative if its end is greater than 
        # the max end of all previous intervals in the sorted order.
        # Actually, if interval i (in sorted order) extends the union, 
        # its end is > current_max_end.
        # The contribution to length is e_sorted[i] - current_max_end.
        # The derivative w.r.t base[k] is 1 if interval k is the one extending 
        # the union at this step, else 0.
        
        current_max_end = -np.inf
        L = 0.0
        grad_vec = np.zeros(K)
        
        for idx in range(K):
            orig_idx = order[idx]
            s = s_sorted[idx]
            e = e_sorted[idx]
            
            if e > current_max_end:
                # This interval extends the union
                contribution = e - current_max_end
                L += contribution
                # The derivative of L w.r.t base[orig_idx] is 1 
                # because d(e)/d(base[orig_idx]) = 1.
                # Note: d(s)/d(base[orig_idx]) = 1 as well, but s only 
                # matters if it is the start of a new segment. 
                # If s > current_max_end, the segment is [s, e]. 
                # d(contribution)/d(base) = d(e - s)/d(base) = 1 - 1 = 0?
                # Wait.
                
                # Let's re-evaluate the derivative.
                # Union length is sum of lengths of disjoint segments.
                # A segment is defined by [start_of_segment, end_of_segment].
                # start_of_segment is the min start of intervals in the component.
                # end_of_segment is the max end of intervals in the component.
                
                # If interval k is the one that extends the union (e_k > current_max_end),
                # it might be:
                # 1. Starting a new segment (if s_k > current_max_end).
                #    Segment: [s_k, e_k]. Length = e_k - s_k.
                #    dL/dbase_k = 1 - 1 = 0.
                # 2. Extending an existing segment (if s_k <= current_max_end < e_k).
                #    The segment end moves from current_max_end to e_k.
                #    Length increase = e_k - current_max_end.
                #    dL/dbase_k = d(e_k)/dbase_k = 1.
                
                if s > current_max_end:
                    # New segment starts here.
                    # The length added is e - s.
                    # d(e-s)/d(base_k) = 1 - 1 = 0.
                    # So grad remains 0 for this interval?
                    # Wait, if I move base_k, s_k and e_k move together.
                    # The length of the segment [s_k, e_k] is constant (width).
                    # So moving an isolated interval does not change the total length?
                    # Correct. If an interval is isolated, moving it doesn't change the union length.
                    # It only changes the position.
                    # So grad should be 0.
                    pass
                else:
                    # Extending existing segment.
                    # The start of the segment was fixed by a previous interval.
                    # The end is determined by this interval.
                    # dL/dbase_k = 1.
                    grad_vec[orig_idx] += 1.0
                
                current_max_end = e
        
        lengths[m] = L
        grads[m] = grad_vec
        
    return lengths, grads


def _union_len_sq_obj_grad(base, motion, M=1000):
    """
    Objective: Integral of (Union Length)^2 over c in [0, 1].
    Returns value and gradient w.r.t base.
    """
    K = len(base)
    c_vals = np.linspace(0.0, 1.0, M)
    dc = 1.0 / (M - 1)
    
    lengths, grads = _compute_union_stats(base, motion, c_vals)
    
    # Value = sum( L(c)^2 * dc )
    val = dc * np.sum(lengths * lengths)
    
    # Gradient = sum( 2 * L(c) * dL/dbase * dc )
    # grads is (M, K). lengths is (M,).
    # grad_k = dc * sum_m ( 2 * lengths[m] * grads[m, k] )
    g = 2.0 * dc * (lengths[:, np.newaxis] * grads).sum(axis=0)
    
    return val, g


def _solve_axis(motion, base0, iters=200):
    """
    Minimize the union volume contribution for one axis.
    Uses L-BFGS-B with a barrier for non-overlap at c=0.
    Constraint: x[k] + 1/8 <= x[k+1] for all k.
    """
    K = len(base0)
    width = 1.0 / 8.0
    lam = 1e8  # High barrier coefficient
    
    def fun(x):
        v, g = _union_len_sq_obj_grad(x, motion, M=1000)
        
        # Barrier for non-overlap: x[k] + width <= x[k+1]
        # Penalty: sum (max(0, x[k] + width - x[k+1])^2) * lam
        # Gradient: 2 * lam * (x[k] + width - x[k+1]) for violating pairs
        
        for k in range(K - 1):
            diff = x[k] + width - x[k+1]
            if diff > 0:
                v += lam * diff * diff
                g[k] += 2.0 * lam * diff
                g[k+1] -= 2.0 * lam * diff
                
        return v, g
    
    # Initialize with a valid grid
    base_init = np.arange(K) * width
    
    res = minimize(fun, base_init, jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "ftol": 1e-14, "gtol": 1e-10, "maxls": 50})
    
    return res.x


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # x-positions depend only on i, y only on j.
    motion = np.arange(N, dtype=np.float64) / N  # c*i/8
    
    # Solve for x and y independently
    base_x = _solve_axis(motion, np.zeros(N), iters=200)
    base_y = _solve_axis(motion, np.zeros(N), iters=200)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions = np.stack([base_x[i_flat], base_y[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
