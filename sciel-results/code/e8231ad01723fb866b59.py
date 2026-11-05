"""
Rewritten program using Differential Evolution for global search 
followed by L-BFGS-B for local refinement.
Includes structured initializations: Laguerre, Generalized Laguerre, 
and paired roots of L_24.
"""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_laguerre, roots_genlaguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# Precompute factorials up to 80
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff) * e^{-x} dx = sum_k coeffs[k]*k!."""
    n = min(len(coeffs), len(_FACTS))
    if n == 0:
        return 0.0
    return float(np.dot(coeffs[:n], _FACTS[:n]))

def _compute_c4(roots):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = prod (x-x_i)^2."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf

    # Build B(x) = prod (x - x_i), then P = B^2, P2 = P^2 = B^4
    B = polyfromroots(roots)  # ascending order, degree 12
    P = polymul(B, B)         # degree 24
    P2 = polymul(P, P)        # degree 48

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf
    c4 = I2 / (I1 * I1)
    return float(c4)

def _objective(roots):
    return _compute_c4(roots)

def _objective_grad(roots):
    """Compute C4 and its gradient for L-BFGS-B."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros_like(roots)

    B = polyfromroots(roots)
    P = polymul(B, B)
    P2 = polymul(P, P)

    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        return np.inf, np.zeros_like(roots)
    c4 = I2 / (I1 * I1)

    # Gradient calculation
    # dC4/dx_i = dI2/(I1^2) - 2*I2*dI1/(I1^3)
    # dI1/dx_i = -2 * int B*R_i * e^{-x} dx
    # dI2/dx_i = -4 * int B^3*R_i * e^{-x} dx
    # where R_i = prod_{j!=i}(x-x_j)
    
    grad = np.zeros_like(roots)
    B3 = polymul(polymul(B, B), B)
    
    for i in range(n):
        R_i = np.array([1.0])
        for j in range(n):
            if j == i:
                continue
            R_i = polymul(R_i, np.array([1.0, -roots[j]]))
        
        BR_i = polymul(B, R_i)
        dI1_i = -2.0 * _poly_moment(BR_i)
        
        B3R_i = polymul(B3, R_i)
        dI2_i = -4.0 * _poly_moment(B3R_i)
        
        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)
    
    return float(c4), grad

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 30.0
    
    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    # 1. Generate Initial Candidates
    candidates = []
    
    # Standard Laguerre L_12
    try:
        x_lag, _ = roots_laguerre(12)
        candidates.append(np.sort(x_lag))
    except Exception:
        pass

    # Generalized Laguerre L_12^(alpha) for a few alphas
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        try:
            x_gen, _ = roots_genlaguerre(12, alpha)
            candidates.append(np.sort(x_gen))
        except Exception:
            pass
            
    # Paired roots of L_24
    # Hypothesis: The optimal double-root configuration is related to the roots of L_24
    try:
        x_24, _ = roots_laguerre(24)
        x_24 = np.sort(x_24)
        # Pair adjacent roots: (y1, y2), (y3, y4)...
        # The root of the square factor would be the geometric mean or arithmetic mean?
        # If B(x) = prod (x - z_k), and we want z_k to represent y_{2k-1}, y_{2k},
        # perhaps z_k = sqrt(y_{2k-1} * y_{2k}) or (y_{2k-1}+y_{2k})/2
        # Let's try arithmetic mean
        z_arith = np.array([(x_24[2*i] + x_24[2*i+1])/2 for i in range(12)])
        candidates.append(np.sort(z_arith))
        
        # Let's try geometric mean
        z_geom = np.array([np.sqrt(x_24[2*i] * x_24[2*i+1]) for i in range(12)])
        candidates.append(np.sort(z_geom))
        
        # Let's try pairing from ends (y1, y24), (y2, y23)...
        z_ends_arith = np.array([(x_24[i] + x_24[23-i])/2 for i in range(12)])
        candidates.append(np.sort(z_ends_arith))
    except Exception:
        pass

    # 2. Evaluate Initial Candidates
    best_roots = None
    best_c4 = np.inf
    
    for cand in candidates:
        if len(cand) != 12: continue
        if not np.all(np.isfinite(cand)) or not np.all(cand > 0): continue
        c4_val = _compute_c4(cand)
        if c4_val < best_c4:
            best_c4 = c4_val
            best_roots = cand.copy()

    if best_roots is None:
        # Fallback to standard Laguerre if something went wrong
        x_lag, _ = roots_laguerre(12)
        best_roots = np.sort(x_lag)
        best_c4 = _compute_c4(best_roots)

    # 3. Local Refinement of Best Initial Candidate
    if time_ok():
        try:
            res = minimize(
                _objective_grad,
                best_roots,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, 200)] * 12,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
        except Exception:
            pass

    # 4. Global Search using Differential Evolution
    if time_ok():
        # Define bounds for DE
        # Roots of L_12 are approx in [1, 30]. Roots of L_24 in [1, 50].
        # Let's use [0.01, 100]
        bounds = [(0.01, 100.0)] * 12
        
        # We will run DE with a limited number of iterations
        # maxiter is the number of generations. 
        # popsize=10 -> 120 individuals.
        # 100 generations -> 12000 evaluations.
        # Each eval takes ~1ms? 12s total.
        # We have ~500s left. We can afford many more.
        
        # However, DE can be slow to converge. 
        # Let's try a moderate number of iterations.
        
        try:
            # Use a seed for reproducibility if needed, but random is fine
            de_res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                atol=1e-12,
                polish=True,  # This runs L-BFGS-B on the best result
                init='sobol', # Better coverage than 'latinhypercube' or 'random'
                updating='deferred' # Faster?
            )
            
            if de_res.success and np.isfinite(de_res.fun):
                candidate = np.sort(de_res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_c4:
                    best_c4 = c4
                    best_roots = candidate.copy()
            else:
                # Even if not "success", check the fun value
                if np.isfinite(de_res.fun):
                    candidate = np.sort(de_res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
                        
        except Exception as e:
            print(f"DE failed: {e}")

    # 5. Final Polish
    if time_ok():
        for _ in range(5):
            if not time_ok(): break
            try:
                res = minimize(
                    _objective_grad,
                    best_roots,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(1e-10, 200)] * 12,
                    options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 30}
                )
                if np.all(res.x > 0) and np.isfinite(res.fun):
                    candidate = np.sort(res.x)
                    c4 = _compute_c4(candidate)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
                    else:
                        break
            except Exception:
                break

    # Fallback
    if not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)

    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
