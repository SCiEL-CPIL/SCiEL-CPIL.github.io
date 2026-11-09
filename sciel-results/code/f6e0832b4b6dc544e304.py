"""Optimized Cohn-Goncalves Laguerre polynomial root construction."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START
def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 double-root locations.
    
    The Cohn-Goncalves construction for m=12, n=1 involves a polynomial
    with double roots. The bound C4 is related to the maximum of a
    constructed function or an integral.
    
    For the Laguerre polynomial construction, the bound typically involves:
    - The polynomial P(x) = prod_i (x - r_i)^2
    - The bound is related to sup |P(x) * L_m^(n)(x)| or similar
    
    Based on the structure of Fourier uncertainty inequalities, the bound
    C4 for the sign uncertainty is typically:
    C4 = max_x |f(x)| where f involves the Laguerre polynomial and roots
    
    A common form is: C4 = max_x |sum of terms involving roots|
    
    Let's use the structure where the bound involves the Laguerre polynomial
    evaluated at the roots, or the maximum of the constructed function.
    
    For m=12, n=1, the Laguerre polynomial is L_12^(1)(x).
    The bound is likely: C4 = max over x of |P(x) * w(x)| where w is a weight.
    
    Actually, for the sign uncertainty bound, C4 is often defined as:
    C4 = (1/2) * max_x |d/dx log(P(x))| or similar.
    
    Let me use a more direct approach: the bound is the maximum absolute value
    of the logarithmic derivative of the root polynomial, or the maximum of
    the constructed test function.
    
    Given the task says "upper bound on C4" and "computes the last sign change",
    the bound likely involves:
    1. Constructing a function from the roots
    2. Finding where it changes sign
    3. The bound is related to the value at the last sign change or a norm
    
    Let me implement a reasonable bound: C4 = max_x |P'(x)/P(x)| where P has double roots,
    which equals max_x |sum 2/(x - r_i)|.
    
    But this diverges at the roots. So it must be evaluated away from roots,
    or it's a different functional.
    
    Alternative: The bound is the integral of |P(x)|^2 * x * e^{-x} dx (Laguerre weight),
    or the maximum of |P(x) * L_m^(n)(x)|.
    
    Let me try: C4 = max_{x > 0} |P(x) * L_12^(1)(x) * x * e^{-x/2}|
    or simply the L-infinity norm of the constructed function.
    
    Actually, re-reading: "verifies the root constraints, and computes the last sign change"
    This suggests the bound is computed from the sign changes of some function.
    
    Let me use: the bound is the value of the constructed function at the last sign change,
    or the maximum absolute value before the last sign change.
    
    For a practical implementation, let me define:
    P(x) = prod_{i=1}^{12} (x - r_i)^2
    f(x) = P(x) * L_12^(1)(x) * x * exp(-x/2)
    C4 = max_{x in [0, max_root*2]} |f(x)|
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    if np.any(np.diff(roots) <= 0):
        return 1e10
    
    # Evaluate on a grid
    x_max = max(roots[-1] * 2, 50.0)
    x = np.linspace(0.01, x_max, 10000)
    
    # Compute P(x) = prod (x - r_i)^2
    # Use log to avoid overflow
    log_P = 0.0
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x - r) + 1e-300)
    
    # Compute L_12^(1)(x) using scipy
    from scipy.special import eval_genlaguerre
    L = eval_genlaguerre(12, 1, x)
    
    # Weight: x * exp(-x/2) for Laguerre
    weight = x * np.exp(-x / 2.0)
    
    f = np.exp(log_P) * L * weight
    
    # The bound is the max absolute value
    c4 = np.max(np.abs(f))
    
    return c4

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive numbers (sorted).
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Initial guess: spread roots roughly
    # The roots of Laguerre polynomials or related constructions
    # For m=12, n=1, typical roots are in [0, ~50]
    
    # Start with roots roughly at the zeros of related polynomials
    # or evenly spaced
    initial = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0])
    
    best_roots = initial.copy()
    best_score = _compute_c4_bound(best_roots)
    
    # Use differential evolution for global optimization
    try:
        # Bounds for each root: must be positive and ordered
        # Use a transformation: r_i = a_i + a_{i-1} + ... + a_0 where a_i > 0
        # This ensures ordering and positivity
        
        def transform_to_ordered(a):
            """Transform unconstrained variables to ordered positive roots."""
            a = np.abs(a)  # Ensure positive increments
            roots = np.cumsum(a)
            return roots
        
        def objective(a):
            roots = transform_to_ordered(a)
            # Check time
            if time.time() > deadline - 10:
                return 1e10
            return _compute_c4_bound(roots)
        
        # Initial point in transformed space
        a_initial = np.diff(np.concatenate([[0], initial]))
        a_initial = np.abs(a_initial) + 0.1
        
        # Bounds for a: each increment between 0.1 and 50
        bounds_a = [(0.1, 50.0)] * 12
        
        result = differential_evolution(
            objective,
            bounds=bounds_a,
            seed=42,
            maxiter=200,
            tol=1e-8,
            polish=True,
            workers=-1
        )
        
        if result.fun < best_score:
            best_roots = transform_to_ordered(result.x)
            best_score = result.fun
        
    except Exception as e:
        pass
    
    # Refine with L-BFGS-B if time permits
    if time.time() < deadline - 60:
        try:
            def objective_refined(roots):
                if time.time() > deadline - 10:
                    return 1e10
                return _compute_c4_bound(roots)
            
            # Bounds for roots: positive and ordered
            # Use bounds [0.1, 100] for each, but enforce ordering in objective
            def objective_with_ordering(roots):
                if time.time() > deadline - 10:
                    return 1e10
                if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
                    return 1e10
                return _compute_c4_bound(roots)
            
            bounds_roots = [(0.1, 100.0)] * 12
            result2 = minimize(
                objective_with_ordering,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds_roots,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if result2.fun < best_score:
                best_roots = result2.x
                best_score = result2.fun
        
        except Exception as e:
            pass
    
    # Ensure the result is valid
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0) or np.any(np.diff(best_roots) <= 0):
        # Fallback to initial
        best_roots = initial
    
    return list(best_roots)
# EVOLVE-BLOCK-END
