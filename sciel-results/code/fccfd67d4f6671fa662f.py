# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_area(x):
    """
    Compute the area of the union of 64 triangles.
    x: shape (64,)
    Returns: scalar area
    """
    n = 64
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n
    
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    area = union_len.mean()
    
    return area

def _solve_x_for_order(order):
    """
    Given a fixed order of L_i at all y (which implies a specific permutation of x's influence),
    find the x that minimizes area.
    
    Actually, the 'order' changes with y. 
    However, a key insight for Kakeya-type problems is that the optimal configuration 
    often corresponds to a specific 'stacking' order.
    
    Let's refine the strategy: 
    The area is convex in x. 
    The subgradient is non-zero only when gaps < w.
    At the optimum, the gradient is zero (or contains 0).
    
    Instead of pure continuous optimization, let's use a hybrid:
    1. Discrete search for the 'topology' of the overlap.
    2. Continuous refinement.
    
    But a simpler, materially different approach:
    Use **Random Restarts + Nelder-Mead** but with a smarter initialization 
    based on the structure of the problem.
    
    Wait, the directive asks for a DIFFERENT design family.
    Current: Gradient Descent on continuous x.
    Inspiration 1: Nelder-Mead (Derivative-free).
    Inspiration 3: Gradient-based but exploiting linearity.
    
    Let's try **Coordinate Descent with Exact Line Search** for each coordinate.
    Since f(x) is convex and piecewise linear, the optimal x_i for fixed others 
    can be found by checking the subgradient.
    
    Actually, let's look at the structure of the subgradient.
    grad_i = sum_y [ 1_{x_i is upper in active gap} - 1_{x_i is lower in active gap} ] / 20000
    
    If we fix x_{-i}, the function f(x_i) is a convex piecewise linear function.
    The minimum is where the slope crosses 0.
    The slope changes only when x_i crosses a 'critical' value where the ordering of L_i 
    changes relative to other L_j, or where a gap becomes exactly w.
    
    We can implement a **Block Coordinate Descent** where for each i, we find the 
    optimal x_i by binary search on the subgradient or by evaluating the piecewise linear segments.
    Given the complexity of the 'critical' points, a simpler robust method is:
    
    **Simulated Annealing on the permutation of 'effective' positions.**
    
    Let's stick to a robust **Nelder-Mead** implementation but with a very specific 
    initialization strategy that explores the space of 'anti-correlated' positions.
    The current program uses Keich. Keich is good but maybe not optimal for n=64.
    
    Let's try **Multiple Restarts with Nelder-Mead**.
    The current program does 1 run of Gradient Descent.
    Nelder-Mead is often better for non-smooth (piecewise linear) functions.
    
    Strategy:
    1. Define a fast area calculator.
    2. Run Nelder-Mead from several diverse initial points.
    3. Keep the best result.
    
    Initial points:
    - Keich construction
    - Random uniform in [-0.5, 0.5]
    - Anti-Keich (reversed)
    - Random permutations of Keich? No, x are continuous.
    
    Let's implement Nelder-Mead manually or use scipy. Scipy is available.
    Using scipy.optimize.minimize with method='Nelder-Mead' is a different family 
    than the current Gradient Descent.
    
    To make it materially different and potentially better:
    We will use **scipy.optimize.differential_evolution** for global search? 
    It might be too slow for 64 dims in 600s.
    
    Let's use **scipy.optimize.minimize** with `Nelder-Mead` from multiple starts.
    The current program's gradient descent might get stuck in a local minimum 
    (though the function is convex, so there is only one global minimum, 
    gradient descent on a non-smooth function can be slow/noisy).
    Nelder-Mead handles non-smoothness better.
    
    Also, we can exploit the fact that the problem is convex.
    The area is convex.
    
    Let's try **Subgradient Method with Polyak Step Size**? 
    We need a lower bound. The minimum area is at least n * (1/64) * (1/2) / 64? 
    No, the triangles have area 1/128 each. Total area if disjoint = 64/128 = 0.5.
    With overlap, it's less.
    The theoretical minimum for a Kakeya set is related to 1.
    For n=64, the area should be significantly less than 0.5.
    
    Let's just use **Nelder-Mead** with `maxiter` set high enough and `xatol`/`fatol` tight.
    We will run it for multiple initial guesses.
    
    One more different strategy:
    **Grid Search on a reduced set of parameters?** No, 64 dims is too high.
    
    Let's go with **Multi-start Nelder-Mead**.
    """
    pass

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620  # 20s safety margin
    
    n = 64
    
    # Pre-compute constants for area calculation
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n
    
    def area_calc(x):
        L = x[:, None] + a[:, None] * y[None, :]
        L_sorted = np.sort(L, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        contrib = np.minimum(gaps, w[None, :])
        union_len = w + contrib.sum(axis=0)
        return union_len.mean()

    # Generate initial points
    inits = []
    
    # 1. Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    inits.append(np.array(keich, dtype=np.float64))
    
    # 2. Random uniform
    random.seed(42)
    for _ in range(5):
        inits.append(np.random.uniform(-0.5, 0.5, n))
        
    # 3. Scaled Keich
    inits.append(np.array(keich, dtype=np.float64) * 0.5)
    inits.append(np.array(keich, dtype=np.float64) * -0.5)
    
    best_x = inits[0]
    best_area = area_calc(best_x)
    
    # Use Nelder-Mead from each initial point
    # We will use scipy.optimize.minimize
    from scipy.optimize import minimize
    
    # Time budget per run: ~30s. We have ~600s. So ~20 runs?
    # But Nelder-Mead in 64D is slow. 
    # Let's limit iterations and use the best of the runs.
    
    # Actually, Nelder-Mead in 64D might be very slow.
    # Let's check the cost of one evaluation: 
    # L is 64x20000 = 1.28M floats. Sort is O(64 log 64 * 20000) ~ 64*6*20000 = 7.6M ops.
    # This is fast in numpy. ~1-5ms per eval.
    # Nelder-Mead needs ~1000-5000 evals to converge in 64D?
    # 5000 * 5ms = 25s. 
    # So we can do ~10-20 runs.
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 60:
            break
            
        # Options for Nelder-Mead
        options = {
            'maxiter': 2000,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        try:
            res = minimize(area_calc, x0, method='Nelder-Mead', options=options)
            if res.success and res.fun < best_area:
                best_area = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final check
    final_area = area_calc(best_x)
    if final_area < best_area:
        best_area = final_area
        
    if not np.all(np.isfinite(best_x)):
        return inits[0]
        
    return best_x

# EVOLVE-BLOCK-END
