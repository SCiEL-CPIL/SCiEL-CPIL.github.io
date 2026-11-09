# EVOLVE-BLOCK-START
import numpy as np
import time

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    # L_i(y) = x_i + a_i * y
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)
    
    # Consecutive differences
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length = w + sum over j of min(d_j, w)
    min_d_w = np.minimum(d, w[np.newaxis, :])
    
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Strategy: Combinatorial Swap-Descent
    # 1. Generate a set of candidate base positions (slots)
    # 2. Assign indices to slots
    # 3. Iteratively swap assignments to reduce area
    
    # The "Keich" construction suggests that the optimal x_i are related to binary representations.
    # Let's generate a pool of candidate x values.
    # We will try to find the best permutation of a fixed set of x values.
    
    # Generate diverse initial pools
    # Pool 1: Keich-like
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0_keich[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    # Pool 2: Uniform random
    np.random.seed(42)
    x0_rand = np.random.uniform(-1.0, 1.0, n)
    
    # Pool 3: Linear spacing
    x0_lin = np.linspace(-0.5, 0.5, n)
    
    best_x = None
    best_area = np.inf
    
    for pool_name, x0 in [("keich", x0_keich), ("rand", x0_rand), ("lin", x0_lin)]:
        if time.time() > deadline - 100:
            break
        
        # For each pool, we treat the values as fixed "slots" and we permute which index i
        # takes which slot.
        # However, the problem allows x_i to be ANY float. 
        # But often, optimal solutions to such piecewise linear problems lie on a "grid" or specific structure.
        # Let's try a hybrid: Start with x0, then do local search where we can swap x_i and x_j,
        # or slightly perturb them.
        
        # Actually, a pure permutation search on a fixed set of x values is a strong structural constraint.
        # Let's try:
        # 1. Sort x0 to get a set of unique values (or just use x0 as is).
        # 2. Randomly permute the assignment of these values to indices 0..63.
        # 3. Perform swap-descent: try swapping x_i and x_j for all pairs (i,j) and accept if area improves.
        
        # To make this more powerful, let's allow the x values themselves to be optimized locally after each swap?
        # No, that's back to gradient descent.
        
        # Let's stick to:
        # A. Fixed set of x values (from x0), random initial permutation.
        # B. Swap descent.
        # C. Repeat with perturbed x0.
        
        current_x = x0.copy()
        # Randomize initial permutation
        np.random.shuffle(current_x)
        
        current_area = _union_area(current_x, n, M)
        if current_area < best_area:
            best_area = current_area
            best_x = current_x.copy()
        
        # Swap Descent
        improved = True
        while improved and time.time() < deadline - 100:
            improved = False
            # To speed up, we can sample a subset of swaps or use a heuristic order.
            # With n=64, there are ~2000 pairs. 2000 area evaluations per pass.
            # Each area eval is ~0.1s? 2000*0.1 = 200s. Too slow for multiple passes.
            # We need to be smarter.
            
            # Let's evaluate area only for promising swaps.
            # Or, limit the number of swaps per epoch.
            
            # Let's do a few passes, but limit the work per pass.
            # Or, use a coarser M for the search, and refine at the end.
            
            pass # Placeholder for the actual search logic below

    # The above loop was a sketch. Let's implement the actual search properly.
    
    # Refined Strategy:
    # 1. Use a coarse quadrature (M=1000) for the search phase.
    # 2. Use a fine quadrature (M=20000) for final evaluation.
    # 3. Search method: Simulated Annealing or Hill Climbing on the permutation of a fixed set of x values.
    #    Plus, allow small continuous perturbations of the x values.
    
    M_search = 1000
    
    def _union_area_fast(x, n=64, M=M_search):
        m = np.arange(M)
        y = (m + 0.5) / M
        i_idx = np.arange(n)
        a = (i_idx + 1) / n
        w = (1.0 - y) / n
        
        L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        min_d_w = np.minimum(d, w[np.newaxis, :])
        union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
        return np.mean(union_lengths)
    
    best_x = None
    best_area = np.inf
    
    # Generate several diverse starting points
    starts = []
    
    # 1. Keich
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0_keich[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    starts.append(x0_keich.copy())
    
    # 2. Random
    np.random.seed(42)
    starts.append(np.random.uniform(-1.0, 1.0, n))
    
    # 3. Linear
    starts.append(np.linspace(-0.5, 0.5, n))
    
    # 4. "Bunched" - cluster some
    x_bunch = np.zeros(n)
    x_bunch[:32] = np.linspace(-1, 0, 32)
    x_bunch[32:] = np.linspace(0, 1, 32)
    starts.append(x_bunch)
    
    for x0 in starts:
        if time.time() > deadline - 100:
            break
            
        current_x = x0.copy()
        current_area = _union_area_fast(current_x, n)
        
        # Hill Climbing with swaps and small perturbations
        no_improve_count = 0
        max_no_improve = 50
        
        while no_improve_count < max_no_improve and time.time() < deadline - 100:
            # Try a random swap
            i, j = np.random.choice(n, 2, replace=False)
            temp_x = current_x.copy()
            temp_x[i], temp_x[j] = current_x[j], current_x[i]
            
            new_area = _union_area_fast(temp_x, n)
            if new_area < current_area:
                current_x = temp_x
                current_area = new_area
                no_improve_count = 0
                if current_area < best_area:
                    best_area = current_area
                    best_x = current_x.copy()
            else:
                # Try a small perturbation of a random element
                i = np.random.choice(n)
                temp_x = current_x.copy()
                temp_x[i] += np.random.normal(0, 0.01)
                new_area = _union_area_fast(temp_x, n)
                if new_area < current_area:
                    current_x = temp_x
                    current_area = new_area
                    no_improve_count = 0
                    if current_area < best_area:
                        best_area = current_area
                        best_x = current_x.copy()
                else:
                    no_improve_count += 1
                    
    # Final Polish with full precision
    if best_x is not None and time.time() < deadline - 20:
        # Run a few iterations of L-BFGS-B on the best found to polish
        from scipy.optimize import minimize
        
        def _union_area_grad(x, n=64, M=20000):
            m = np.arange(M)
            y = (m + 0.5) / M
            i_idx = np.arange(n)
            a = (i_idx + 1) / n
            w = (1.0 - y) / n
            
            L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
            sort_idx = np.argsort(L, axis=0)
            L_sorted = np.take_along_axis(L, sort_idx, axis=0)
            d = L_sorted[1:, :] - L_sorted[:-1, :]
            min_d_w = np.minimum(d, w[np.newaxis, :])
            union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
            area = np.mean(union_lengths)
            
            grad = np.zeros(n, dtype=np.float64)
            mask = d < w[np.newaxis, :]
            j_indices, m_indices = np.nonzero(mask)
            if len(j_indices) > 0:
                upper_idx = sort_idx[j_indices + 1, m_indices]
                lower_idx = sort_idx[j_indices, m_indices]
                weight = 1.0 / M
                grad += np.bincount(upper_idx, weights=np.full_like(upper_idx, weight, dtype=np.float64), minlength=n)
                grad -= np.bincount(lower_idx, weights=np.full_like(lower_idx, weight, dtype=np.float64), minlength=n)
            return area, grad

        try:
            res = minimize(
                _union_area_grad,
                best_x,
                args=(n, M),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 20, 'gtol': 1e-5}
            )
            if res.fun < best_area:
                best_x = res.x
                best_area = res.fun
        except Exception:
            pass

    if best_x is None:
        return starts[0].astype(np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
