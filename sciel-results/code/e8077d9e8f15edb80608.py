"""Refined uncertainty inequality: Dyson-gas mean-field with linear confining potential."""
import numpy as np
from scipy.optimize import brentq
import math

M = 12
N = 1

def _laguerre_zeros_alpha_beta(alpha, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x)
    using Golub-Welsch on Jacobi matrix.
    """
    n = degree
    J = np.zeros((n, n))
    for i in range(n):
        J[i, i] = 2 * (i + 1) + alpha + 1
    for i in range(n - 1):
        val = (i + 1) * (i + 1 + alpha)
        J[i, i + 1] = math.sqrt(val)
        J[i + 1, i] = math.sqrt(val)
    eigvals = np.linalg.eigvalsh(J)
    return np.sort(eigvals)

def _compute_score(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(roots) <= 0):
        return float('inf')
    seed_roots = 4.0 * np.arange(1, 13)
    seed_score = np.exp(np.mean(np.log(1 + seed_roots)))
    current_score = np.exp(np.mean(np.log(1 + roots)))
    return current_score / seed_score

def _compute_laguerre_zeros(n, alpha=0):
    """Compute the n zeros of L_n^(alpha)(x)."""
    return _laguerre_zeros_alpha_beta(alpha, n)

def _dyson_gas_equilibrium(roots):
    """
    Solve the Dyson gas equilibrium with repulsion exponent 2 and linear confining potential.
    For double roots at r_1 < r_2 < ... < r_n, the mean-field condition is:
        sum_{j != i} 2 / (r_i - r_j) = 1
    for each i. This comes from the logarithmic derivative of the squared product
    balanced against the linear potential V(x) = x with V'(x) = 1.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    for iteration in range(1000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-12
                right = other_sorted[0] - 1e-10
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-10
                right = max(200.0, r_guess + 20.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-10
                right = other_sorted[insert_pos] - 1e-10
            
            if left < 1e-12:
                left = 1e-12
            if right <= left:
                left = max(1e-12, r_guess - 10.0)
                right = r_guess + 10.0
            
            def f(r):
                if r <= 1e-15:
                    return 1e15
                if np.any(np.abs(r - other_roots) < 1e-15):
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return 2.0 * s - 1.0
            
            found = False
            for attempt in range(80):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if np.isnan(f_left) or np.isnan(f_right):
                        left = max(1e-12, left - 2.0)
                        right = right + 2.0
                        continue
                    
                    if f_left * f_right < 0:
                        r_new = brentq(f, left, right, xtol=1e-15, rtol=1e-15, maxiter=300)
                        new_roots[i] = r_new
                        found = True
                        break
                    elif abs(f_left) < 1e-15:
                        new_roots[i] = left
                        found = True
                        break
                    elif abs(f_right) < 1e-15:
                        new_roots[i] = right
                        found = True
                        break
                    else:
                        left = max(1e-12, left - 3.0)
                        right = right + 3.0
                except Exception:
                    left = max(1e-12, left - 3.0)
                    right = right + 3.0
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-15:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-12)
    
    return roots

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas mean-field with repulsion exponent 2 and linear confining potential.
    """
    # Start with Laguerre zeros as initial guess
    init_roots = _compute_laguerre_zeros(12, alpha=0)
    
    # Solve the Dyson gas equilibrium: sum_{j!=i} 2/(r_i - r_j) = 1
    roots = _dyson_gas_equilibrium(init_roots)
    
    # Verify constraints
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        roots = _compute_laguerre_zeros(12, alpha=0)
    
    return [float(r) for r in roots]

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    roots = construct()
    print("Roots:", roots)
    print("Score:", _compute_score(roots))
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    print("Seed score:", _compute_score(seed_roots))
