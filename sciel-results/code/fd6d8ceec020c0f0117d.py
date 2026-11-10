# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

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

def _solve_for_order(order):
    """
    Given a permutation 'order' (list of indices 0..63) such that 
    x_{order[0]} <= x_{order[1]} <= ... <= x_{order[63]},
    find the x that minimizes the area.
    
    The area is integral of union length.
    Union length at height y is sum of lengths of connected components.
    With sorted L, union length = w + sum_{j=0}^{62} min(L_{j+1} - L_j, w).
    
    This is a convex piecewise linear function of x.
    We can formulate this as a Linear Program (LP) or solve via subgradient methods 
    restricted to the polytope defined by the order constraints.
    
    However, the objective is non-linear in the 'min' part.
    Wait, min(A, B) is concave. Minimizing a concave function is non-convex.
    BUT, we are summing min(gaps, w).
    
    Let's look at the structure again.
    Area = E_y [ w + sum min(L_{j+1} - L_j, w) ]
    
    Actually, let's use a different insight.
    If we fix the order, the variables are x_i.
    The constraints are x_{order[k]} <= x_{order[k+1]}.
    
    The objective function is convex?
    f(x) = sum_y sum_j min(L_{j+1}(y) - L_j(y), w(y))
    min(a, b) is concave in (a,b). So the objective is concave in x?
    If the objective is concave, the minimum is at the boundary of the feasible region.
    The feasible region is a polytope defined by x_{order[k]} <= x_{order[k+1]}.
    The vertices of this polytope (intersected with some bounding box) are where many constraints are tight.
    
    Actually, let's just use a fast gradient method restricted to the order.
    Or, simpler: use the fact that for a fixed order, the optimal x_i are often such that 
    the gaps are either 0 or w (saturation).
    
    Let's try a simpler heuristic for the fixed order:
    The "Keich" construction is optimal for a specific order (the bit-reversal order).
    We can generate a random order, then perform coordinate descent to minimize area 
    subject to the order constraints.
    """
    n = 64
    # Initialize x with small random values respecting the order
    # Or start from a known good solution and project?
    # Let's start with x_i = i * 0.01 (roughly increasing) and then permute?
    # Better: Start with x all zeros, then add small noise, then sort?
    
    # We want x[order[0]] <= x[order[1]] ...
    # Let's generate random x, then sort them according to the order?
    # No, we need to find the specific values.
    
    # Let's use scipy.optimize.minimize with SLSQP or trust-constr?
    # Too slow for 64 vars in a loop.
    
    # Alternative: Coordinate Descent.
    # For each i, optimize x_i given others, subject to order constraints.
    # The function is convex in each x_i?
    # L_i(y) = x_i + a_i y.
    # The term involving x_i is in min(L_i - L_{prev}, w) + min(L_{next} - L_i, w).
    # This is concave in x_i. So minimum is at boundaries.
    # Boundaries are x_i = x_{prev} or x_i = x_{next} - w(y)?? No, w(y) varies.
    
    # Actually, let's look at the "Keich" construction again.
    # It places triangles such that they interleave.
    # The order of L_i(y) changes with y.
    # So a single fixed order for x_i does NOT imply a fixed order for L_i(y) for all y.
    # The order of L_i(y) is determined by x_i + a_i y.
    # Since a_i are distinct, the order changes at y where x_i + a_i y = x_j + a_j y.
    # There are O(n^2) such y values.
    
    # So the "fixed order" strategy is flawed if we fix the order of x_i.
    # We should fix the order of L_i(y) for a specific y? No.
    
    # Let's go back to the inspiration: "discrete combinatorial search over the ordering".
    # Perhaps it means searching over the *permutation of the a_i*? No, a_i are fixed.
    # It must mean searching over the *initial positions* x_i such that the resulting 
    # "interleaving" pattern is good.
    
    # Let's try a different strategy: **Simulated Annealing on x**.
    # This is robust and can escape local minima.
    # We will use a fast area calculation.
    
    return None # Placeholder, will implement in main function

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initialize with Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    best_x = x.copy()
    best_area = _calc_area(best_x)
    
    # Simulated Annealing
    T = 0.05  # Initial temperature
    T_min = 1e-6
    alpha = 0.995 # Cooling rate
    
    # Precompute constants for fast area calc
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n
    
    def fast_area(x_curr):
        L = x_curr[:, None] + a[:, None] * y[None, :]
        L_sorted = np.sort(L, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        contrib = np.minimum(gaps, w[None, :])
        union_len = w + contrib.sum(axis=0)
        return union_len.mean()
        
    # We need to be careful with time.
    # Each area calc is ~5-10ms.
    # 600s / 0.01s = 60,000 iterations possible.
    # But we have overhead. Let's aim for 10,000-20,000 iterations.
    
    iter_count = 0
    while time.time() < deadline - 10:
        if T < T_min:
            break
            
        # Generate neighbor
        # Move one random coordinate by a random amount
        i = np.random.randint(n)
        delta = np.random.normal(0, T * 0.5) # Scale step with T?
        # Or fixed step size?
        # Let's use fixed step size that decreases slowly.
        step_size = 0.01 * (T / 0.05)
        
        x_new = x.copy()
        x_new[i] += delta
        
        area_new = fast_area(x_new)
        area_curr = fast_area(x) # Cache this?
        # To save time, calculate area_curr once per loop if x didn't change?
        # But x changes if accepted.
        
        # Optimization: Keep current area in variable
        if not hasattr(kakeya_needle_2d_positions64, '_last_area'):
             kakeya_needle_2d_positions64._last_area = best_area
             
        # Actually, let's just compute it. It's fast enough.
        # But wait, we computed area_new. We need area_curr.
        # If we rejected last step, area_curr is still valid.
        # If we accepted, area_curr is area_new from previous step.
        
        # Let's maintain current_area
        if not hasattr(kakeya_needle_2d_positions64, '_curr_area'):
             kakeya_needle_2d_positions64._curr_area = _calc_area(x)
             
        curr_area = kakeya_needle_2d_positions64._curr_area
        
        delta_area = area_new - curr_area
        
        if delta_area < 0 or np.random.rand() < np.exp(-delta_area / T):
            x = x_new
            kakeya_needle_2d_positions64._curr_area = area_new
            if area_new < best_area:
                best_area = area_new
                best_x = x.copy()
                
        T *= alpha
        iter_count += 1
        
        # Safety check
        if time.time() > deadline - 5:
            break
            
    # Final cleanup
    if not np.all(np.isfinite(best_x)):
        return np.array(positions, dtype=np.float64)
        
    return best_x

# EVOLVE-BLOCK-END
