"""Optimized uncertainty bound via linear parameterization + analytic gradient."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import factorial, sqrt, pi

_DEADLINE = 190.0

# Precomputed Hermite coefficients and integrals
def _hermite_coeffs(n):
    if n == 0: return {0: 1.0}
    if n == 1: return {1: 2.0}
    h_prev = {0: 1.0}
    h_curr = {1: 2.0}
    for m in range(2, n + 1):
        h_new = {}
        for p, c in h_curr.items():
            new_p = p + 1
            h_new[new_p] = h_new.get(new_p, 0.0) + 2.0 * c
        for p, c in h_prev.items():
            h_new[p] = h_new.get(p, 0.0) - 2.0 * (m - 1) * c
        h_new = {p: c for p, c in h_new.items() if abs(c) > 1e-15}
        h_prev, h_curr = h_curr, h_new
    return h_curr

def _int_x2m(m):
    if m == 0: return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val

# Degrees of freedom for p(x)
DEGREES = [0, 4, 8, 12]
N_DEG = 4

# Precompute Hermite polynomials as coefficient vectors
H_POLYS = []
for d in DEGREES:
    hc = _hermite_coeffs(d)
    vec = np.zeros(d + 1)
    for p, c in hc.items():
        vec[p] = c
    H_POLYS.append(vec)

# Precompute M0, M2, L matrices
M0_mat = np.zeros((N_DEG, N_DEG))
for i, d in enumerate(DEGREES):
    M0_mat[i, i] = (2 ** d) * factorial(d)

M2_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        deg_i = DEGREES[i]
        deg_j = DEGREES[j]
        ci = H_POLYS[i]
        cj = H_POLYS[j]
        prod = np.convolve(ci, cj)
        prod_x2 = np.zeros(len(prod) + 2)
        prod_x2[2:] = prod
        val = 0.0
        for k in range(0, len(prod_x2), 2):
            m = k // 2
            val += prod_x2[k] * _int_x2m(m)
        M2_mat[i, j] = val

Q_POLYS = []
for i, d in enumerate(DEGREES):
    q_vec = np.zeros(max(d + 1, 1))
    if d == 0:
        q_vec = np.zeros(2)
        q_vec[1] = -1.0
    else:
        h_minus = _hermite_coeffs(d - 1)
        h_plus = _hermite_coeffs(d + 1)
        max_deg_q = d + 1
        q_vec = np.zeros(max_deg_q + 1)
        for p, c in h_minus.items():
            q_vec[p] += d * c
        for p, c in h_plus.items():
            q_vec[p] -= 0.5 * c
    Q_POLYS.append(q_vec)

L_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        ci = Q_POLYS[i]
        cj = Q_POLYS[j]
        prod = np.convolve(ci, cj)
        val = 0.0
        for k in range(0, len(prod), 2):
            m = k // 2
            val += prod[k] * _int_x2m(m)
        L_mat[i, j] = val

# Construct the transformation matrix B such that a_4 = B @ a_3
# a_3 = [a0, a4, a8]
# a_12 = -(a0 + 12*a4 + 2520*a8) / 27720
# a_4_vec = [a0, a4, a8, a12]
# B = [[1, 0, 0],
#      [0, 1, 0],
#      [0, 0, 1],
#      [-1/27720, -12/27720, -2520/27720]]

B = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0/27720.0, -12.0/27720.0, -2520.0/27720.0]
])

# Effective 3x3 matrices
M0_eff = B.T @ M0_mat @ B
M2_eff = B.T @ M2_mat @ B
L_eff = B.T @ L_mat @ B

def _objective_linear(a3):
    """
    a3 = [a0, a4, a8]
    """
    M0 = a3 @ M0_eff @ a3
    M2 = a3 @ M2_eff @ a3
    L = a3 @ L_eff @ a3
    
    if M0 <= 1e-12 or L <= 1e-12 or M2 <= 1e-12:
        return np.inf, np.zeros(3)
    
    bound = np.sqrt(M2 * L) / M0
    
    # Gradient w.r.t a3
    dM0 = 2 * M0_eff @ a3
    dM2 = 2 * M2_eff @ a3
    dL = 2 * L_eff @ a3
    
    term1 = (L * dM2 + M2 * dL) / (2.0 * M0 * np.sqrt(M2 * L))
    term2 = bound * dM0 / (M0 * M0)
    
    grad_a3 = term1 - term2
    
    return bound, grad_a3

def construct():
    start_time = time.time()
    
    # Multiple initial guesses to avoid local minima
    # Heuristic guesses based on known good solutions or random scaling
    inits = [
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-0.5, -0.5, 0.1]),
        np.array([-2.0, 0.1, -0.01]),
        np.array([0.0, 0.0, 0.0]) # Will be inf, but L-BFGS-B might handle or we skip
    ]
    
    best_x = None
    best_bound = np.inf
    
    for x0 in inits:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        try:
            # Use L-BFGS-B with analytic gradient
            result = minimize(
                _objective_linear,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
        
        # If we found a very good solution, refine it
        if best_bound < 1.5: # Arbitrary threshold for "good enough" to warrant refinement
             try:
                result = minimize(
                    _objective_linear,
                    best_x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
             except Exception:
                 pass

    if best_x is None:
        # Fallback to a known reasonable value if all optimizations failed
        return np.array([-1.13, 0.04, 0.0003])
        
    return best_x

# EVOLVE-BLOCK-END
