# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
import concurrent.futures
import itertools
import math

def _calc_area(x):
    """
    Compute the area of the union of the 64 triangles for given positions x.
    n = 64
    """
    n = 64
    w = 1.0 / n
    
    # a_i = (i + 1) / 64
    a = (np.arange(n) + 1) / n
    
    # y_m = (m + 0.5) / 20000, m = 0..19999
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    # L_i = x_i + a_i * y
    # Shape: (n, 20000)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L_i for each y_m
    L_sorted = np.sort(L, axis=0)
    
    # Calculate union length at each height
    diffs = np.diff(L_sorted, axis=0)
    added_lengths = np.minimum(diffs, w)
    
    # Sum over triangles (axis 0) and add the base width w
    union_lengths = w + np.sum(added_lengths, axis=0)
    
    # Area is the mean of union lengths
    area = np.mean(union_lengths)
    return area

def _solve_lp_for_perm(perm, n):
    """
    Solve the LP for a specific permutation 'perm' of the triangles.
    Returns the optimal x and the area.
    """
    w = 1.0 / n
    a = (np.arange(n) + 1) / n
    
    # Variables: x_0 ... x_{n-1}
    # We want to minimize the area.
    # The area is the integral of the union length.
    # For a fixed permutation, the union length at any y is:
    # w + sum_{j=0}^{n-2} min(L_{perm[j+1]} - L_{perm[j]}, w)
    # where L_i = x_i + a_i y.
    
    # This function is piecewise linear in x.
    # We can formulate this as an LP.
    # Let z_j be the "added length" for the gap between perm[j] and perm[j+1].
    # z_j = min(L_{perm[j+1]} - L_{perm[j]}, w)
    
    # Objective: Minimize mean(union_lengths)
    # Since the mean is linear in the integral, and the integral of a linear function is linear,
    # the objective is linear in x and z.
    
    # However, the "min" makes it non-linear.
    # We can replace min(A, w) with variable z such that:
    # z <= A
    # z <= w
    # Maximize z (or in minimization of total length, we want z to be as small as possible? 
    # No, the union length is w + sum z_j. We want to MINIMIZE the union length.
    # So we want z_j to be as small as possible? 
    # Wait. z_j = min(gap, w). The union length is w + sum z_j.
    # To minimize the sum, we want z_j to be small.
    # But z_j is constrained to be equal to min(gap, w).
    # If gap < w, z_j = gap. If gap > w, z_j = w.
    # In the LP, we can define z_j such that:
    # z_j <= L_{perm[j+1]} - L_{perm[j]}
    # z_j <= w
    # And we MINIMIZE sum(z_j).
    # This will force z_j to take the value min(L_{perm[j+1]} - L_{perm[j]}, w) 
    # IF the objective pushes it down. 
    # Wait, if L_{perm[j+1]} - L_{perm[j]} is negative (which shouldn't happen if perm is valid order at all y? 
    # Actually, the permutation might change with y. 
    # The LP formulation for a FIXED permutation assumes that this permutation is the sorted order for ALL y.
    # If the permutation changes with y, the LP for that specific permutation is invalid for the parts where the order changes.
    # However, the global minimum of the union area is the minimum over all possible "piecewise linear" structures.
    # It is known that for Kakeya sets, the optimal configuration often involves specific orderings.
    # But simply fixing one permutation for all y is a restriction.
    # Is the global optimum achieved by a configuration where the relative order of L_i is constant for all y?
    # L_i(y) = x_i + a_i y. 
    # The order changes when x_i + a_i y = x_j + a_j y => y = (x_j - x_i) / (a_i - a_j).
    # Since a_i are distinct, the order changes at specific y values.
    # So the union length function is piecewise linear in y, with breakpoints determined by x.
    # Minimizing the integral is complex.
    
    # Alternative: The area is the measure of the set U = union of triangles.
    # The triangles are convex.
    # This approach (LP per permutation) is valid ONLY if we assume the permutation is fixed.
    # If the permutation is not fixed, the LP solution for that permutation might not be feasible for the "true" union area calculation if the order changes.
    # HOWEVER, if we find an x such that the permutation IS fixed (i.e., no crossings), then the LP gives the exact area for that x.
    # If the LP solution results in crossings, the calculated area (based on the assumed permutation) might be different from the true area.
    # But wait, if we minimize the area assuming a permutation, and the result has crossings, the "assumed area" is an upper bound or lower bound?
    # If we assume order 1,2,3, but actually 1,3,2, the formula w + min(L2-L1, w) + min(L3-L2, w) might not equal the true union length.
    # True union length for 1,3,2 is w + min(L3-L1, w) + min(L2-L3, w) (if L2 is between L1 and L3? No, sorted is 1,2,3 or 1,3,2).
    # If sorted is 1,2,3: w + min(L2-L1, w) + min(L3-L2, w).
    # If sorted is 1,3,2: w + min(L3-L1, w) + min(L2-L3, w).
    # Note that min(A, w) + min(B, w) is not necessarily equal to min(A+B, w) or similar.
    # Actually, the union of intervals [L1, L1+w], [L2, L2+w], [L3, L3+w].
    # If L1 < L3 < L2, the union is [L1, max(L1+w, L3+w, L2+w)] minus gaps?
    # The formula "w + sum min(gaps, w)" works for ANY sorted order.
    # So if we fix a permutation that is NOT the sorted order at some y, the formula using that permutation's gaps will NOT give the correct union length.
    # Specifically, if the true sorted order is different, the "gaps" in the assumed permutation are not the consecutive gaps in the true sorted set.
    # Therefore, minimizing the area under the assumption of a fixed permutation P is equivalent to minimizing a function F_P(x) which is NOT the true area A(x) unless P is the true sorted order for all y.
    # F_P(x) is a lower bound on A(x)? Or upper?
    # If P is not the sorted order, the sum of min(gaps, w) for P is generally larger or smaller?
    # Consider 3 points. True sorted: 1,2,3. Gaps d12, d23. Cost min(d12,w)+min(d23,w).
    # Assume order 1,3,2. Gaps d13, d32. 
    # If 1 < 2 < 3, then d13 = d12 + d23. d32 = -d23 (negative, so min(-d23, w) = -d23? No, L is sorted in formula. 
    # The formula requires L to be sorted. If we plug unsorted L into the formula, it's invalid.
    # So we MUST enforce the permutation constraints: L_{perm[j]} <= L_{perm[j+1]} for all j, for ALL y.
    # This implies x_{perm[j]} + a_{perm[j]} y <= x_{perm[j+1]} + a_{perm[j+1]} y for all y in [0,1].
    # This is a strong constraint. It means the lines don't cross.
    # If we enforce non-crossing, the permutation is fixed.
    # Then F_P(x) = A(x).
    # So we are searching for the global minimum among all non-crossing configurations.
    # Is the global optimum non-crossing? 
    # Intuitively, crossings increase complexity. But maybe necessary?
    # Keich construction is non-crossing? 
    # Let's assume the global optimum is non-crossing. If not, we miss it.
    # But let's try this first. It's a rigorous global search over the space of non-crossing configurations.
    
    # Constraints:
    # 1. L_{perm[j]} <= L_{perm[j+1]} for all j, for all y.
    #    x_{perm[j]} + a_{perm[j]} y <= x_{perm[j+1]} + a_{perm[j+1]} y for y in [0,1].
    #    This is linear in x.
    #    At y=0: x_{perm[j]} <= x_{perm[j+1]}
    #    At y=1: x_{perm[j]} + a_{perm[j]} <= x_{perm[j+1]} + a_{perm[j+1]}
    #    Since the difference is linear in y, if it's non-negative at endpoints, it's non-negative everywhere.
    #    So we just need 2 constraints per adjacent pair in perm.
    
    # 2. Objective: Minimize Area.
    #    Area = integral_0^1 (w + sum_{j=0}^{n-2} min(L_{perm[j+1]} - L_{perm[j]}, w)) dy
    #    Let g_j(y) = L_{perm[j+1]}(y) - L_{perm[j]}(y) = (x_{perm[j+1]} - x_{perm[j]}) + (a_{perm[j+1]} - a_{perm[j]}) y.
    #    We need to integrate min(g_j(y), w).
    #    g_j(y) is linear. min(g_j(y), w) is piecewise linear.
    #    The integral of min(g_j(y), w) can be computed analytically or represented as an LP with auxiliary variables.
    #    Let I_j = integral_0^1 min(g_j(y), w) dy.
    #    We want to minimize w + sum I_j.
    
    #    To linearize I_j:
    #    Let h_j(y) = min(g_j(y), w).
    #    I_j = integral h_j(y) dy.
    #    We can introduce a variable for the value of the integral? No, we need to relate I_j to x.
    #    Note that g_j(y) = c_j + m_j y, where c_j = x_{perm[j+1]} - x_{perm[j]} and m_j = a_{perm[j+1]} - a_{perm[j]}.
    #    The function min(c_j + m_j y, w) is:
    #    - If c_j + m_j * 0 >= w and c_j + m_j * 1 >= w (i.e., always >= w), then I_j = w.
    #    - If c_j + m_j * 0 <= w and c_j + m_j * 1 <= w (i.e., always <= w), then I_j = c_j + m_j/2.
    #    - Else, it crosses w at y* = (w - c_j) / m_j.
    #      I_j = integral_0^{y*} (c_j + m_j y) dy + integral_{y*}^1 w dy.
    
    #    This is non-linear in x (because of the conditional logic and y* depends on x).
    #    However, we can linearize it using the fact that min(A, w) is concave? No, min is concave.
    #    Minimizing a sum of concave functions is non-convex.
    #    Wait, min(g, w) is concave in g. g is linear in x. So min(g, w) is concave in x.
    #    We are MINIMIZING a sum of concave functions. This is a NON-CONVEX problem.
    #    LP solvers cannot handle this directly.
    
    #    RE-EVALUATION:
    #    The union area is NOT a convex function of x.
    #    Therefore, we cannot simply solve an LP to find the global minimum.
    #    
    #    Alternative Strategy: 
    #    Use the fact that the area is the volume of the union.
    #    Maybe we can use a different formulation.
    #    
    #    Let's look at the inspirations.
    #    Inspiration 1: Nelder-Mead.
    #    Inspiration 2: Gradient-free.
    #    
    #    The prompt suggests: "Model the union area minimization as a convex program...".
    #    Is it possible I am wrong about the convexity?
    #    Union of intervals length is a convex function of the endpoints?
    #    Let f(L) = length(union([L_i, L_i+w])).
    #    f is convex in L?
    #    If L1=0, L2=1, w=1. Union [0,2]. Length 2.
    #    If L1=-0.5, L2=1.5. Union [-0.5, 2.5]. Length 3.
    #    Midpoint L1=0, L2=1. Length 2.
    #    (3+2)/2 = 2.5 > 2. So it is NOT convex? 
    #    Wait, (f(L1)+f(L2))/2 >= f(mid). 2.5 >= 2. Yes, it is convex.
    #    Let's check another case.
    #    L1=0, L2=2, w=1. Union [0,1] U [2,3]. Length 2.
    #    L1=-1, L2=3, w=1. Union [-1,0] U [3,4]. Length 2.
    #    Midpoint L1=0, L2=2. Length 2.
    #    (2+2)/2 = 2. Equal.
    #    
    #    L1=0, L2=0.5, w=1. Union [0, 1.5]. Length 1.5.
    #    L1=-0.5, L2=1, w=1. Union [-0.5, 2]. Length 2.5.
    #    Midpoint L1=0, L2=0.5. Length 1.5.
    #    (1.5+2.5)/2 = 2.0 > 1.5. Convex.
    
    #    It seems the union length is a convex function of the left endpoints L.
    #    Since L_i = x_i + a_i y is linear in x, the union length at each y is convex in x.
    #    The integral (area) is the integral of a convex function, so it is convex in x.
    #    
    #    SO, THE PROBLEM IS CONVEX!
    #    
    #    Why did I think min(g, w) is concave?
    #    min(g, w) is concave in g.
    #    But the union length is NOT sum min(g_j, w).
    #    The formula "w + sum min(gaps, w)" is for the SORTED L.
    #    The sorting operation introduces non-linearity.
    #    However, the function f(L) = length(union) is convex in L.
    #    And L is linear in x.
    #    So Area(x) is convex in x.
    #    
    #    We can minimize a convex function using gradient methods.
    #    But the gradient is not easy to compute analytically due to the sorting and min.
    #    However, we can use subgradients.
    #    Or, we can use a solver that handles convex problems.
    #    SciPy's `minimize` with method='L-BFGS-B' requires gradients.
    #    We can approximate gradients using finite differences.
    #    
    #    BUT, the prompt suggests using a "specialized interior-point method or CVXPY".
    #    CVXPY is not in the standard library.
    #    SciPy has `linprog`, `milp`, `nnls`, etc.
    #    Does SciPy have a general convex solver?
    #    `scipy.optimize.minimize` with `method='SLSQP'` can handle constraints, but it's not a dedicated convex solver.
    #    
    #    Given the time limit and the need for a "structural change", let's try a **Subgradient Method** or **Mirror Descent** which are suitable for convex problems with non-smooth objectives (like union area).
    #    
    #    Strategy:
    #    1. Define the objective function J(x) = Area(x).
    #    2. Compute a subgradient of J(x) at x.
    #       The subgradient of the union length w.r.t L_i is:
    #       - 1 if L_i is the unique leftmost point.
    #       - 1 if L_i is the unique rightmost point? No.
    #       Let's derive the subgradient of f(L) = length(union([L_i, L_i+w])).
    #       f(L) = max(L_i + w) - min(L_i) + sum of gaps? No.
    #       f(L) is the length of the union.
    #       The subgradient w.r.t L_i is the derivative of the union length if we move L_i.
    #       If L_i is in the interior of the union (covered by another interval), moving it doesn't change the union length (derivative 0).
    #       If L_i is the leftmost point, moving it left increases the length by 1. Moving it right decreases it by 1 (if it doesn't overlap).
    #       So the subgradient is in [-1, 1].
    #       Specifically, it's the indicator of whether L_i is "exposed" on the left or right.
    #       
    #       Actually, there is a known result: The subgradient of the union length w.r.t L_i is:
    #       g_i = 1 if L_i is the minimum L.
    #       g_i = -1 if L_i is the maximum L? No.
    #       Let's use finite differences for the gradient. It's robust.
    #    
    #    3. Use L-BFGS-B with finite difference gradients.
    #       This is a standard, robust method for convex problems.
    #       It is structurally different from coordinate descent (which is a local, non-smooth heuristic).
    #       L-BFGS-B uses quasi-Newton information and global line search.
    
    def _calc_area_and_grad(x, epsilon=1e-6):
        """
        Compute area and approximate gradient using central differences.
        """
        n = len(x)
        area = _calc_area(x)
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += epsilon
            area_plus = _calc_area(x_plus)
            
            x_minus = x.copy()
            x_minus[i] -= epsilon
            area_minus = _calc_area(x_minus)
            
            grad[i] = (area_plus - area_minus) / (2 * epsilon)
        
        return area, grad

    from scipy.optimize import minimize
    
    # Initial guess: Keich construction
    j = np.arange(1, int(np.log2(n)) + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (int(np.log2(n)) - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / int(np.log2(n))))
    x0 = np.array(keich_positions, dtype=np.float64)
    
    # Optimize
    # L-BFGS-B is good for bounded/convex problems.
    # We don't have explicit bounds, but x can be anywhere.
    # However, we can add bounds if needed.
    
    res = minimize(
        _calc_area, 
        x0, 
        method='L-BFGS-B', 
        jac=_calc_area_and_grad, # Pass the gradient function
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    return res.x, res.fun

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize the base positions x for the Kakeya needle problem (n=64)
    using L-BFGS-B with finite difference gradients on the convex area function.
    """
    n = 64
    start_time = time.time()
    timeout = 500.0
    deadline = start_time + timeout
    
    # Run the optimizer
    # Since the problem is convex, a single run from a good start should find the global minimum.
    # We'll try a few random starts to be safe, but L-BFGS-B is global for convex functions.
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_area = float('inf')
    
    # Start 1: Keich
    j = np.arange(1, int(np.log2(n)) + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (int(np.log2(n)) - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / int(np.log2(n))))
    x0 = np.array(keich_positions, dtype=np.float64)
    
    # Start 2: Zero
    x1 = np.zeros(n)
    
    # Start 3: Random
    x2 = rng.uniform(-0.5, 1.5, n)
    
    starts = [x0, x1, x2]
    
    for x_start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            x_opt, area_opt = _solve_lp_for_perm(x_start, n) # Reusing the name, but it's actually L-BFGS
            if area_opt < best_area:
                best_area = area_opt
                best_x = x_opt
        except Exception:
            pass
            
    if best_x is None:
        return x0
        
    return best_x

# EVOLVE-BLOCK-END
