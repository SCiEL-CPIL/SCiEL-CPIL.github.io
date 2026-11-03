"""Refined uncertainty inequality: Dyson-gas mean-field fixed point for Laguerre roots."""
import numpy as np
from scipy.optimize import minimize, brentq
import math

# Constants
M = 12
N = 1  # dimension

def _laguerre_zeros_alpha_beta(alpha, beta, degree):
    """
    Compute zeros of generalized Laguerre polynomial L_degree^(alpha)(x)
    using Golub-Welsch on Jacobi matrix.
    For Laguerre: weight x^alpha e^{-x}, alpha > -1.
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
    The C4 upper bound is the geometric mean of (1 + r_i) over all roots.
    """
    roots = np.array(roots, dtype=float)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(roots) <= 0):
        return float('inf')
    
    # The upper bound on C4 is the geometric mean of (1 + r_i)
    current_score = np.exp(np.mean(np.log(1 + roots)))
    
    return current_score

def _compute_laguerre_zeros(n, alpha=0):
    """Compute the n zeros of L_n^(alpha)(x)."""
    return _laguerre_zeros_alpha_beta(alpha, 0, n)

def _dyson_gas_equilibrium(roots, double_root_factor=2.0):
    """
    Solve the Dyson gas equilibrium for double roots with Laguerre confinement.
    
    The mean-field potential for a Dyson gas with repulsion exponent beta=2
    and linear confining potential V(x) = x (from Laguerre weight e^{-x})
    plus the boundary term from x^alpha (alpha=0) gives the equilibrium:
    
    For each root r_i:
      double_root_factor * sum_{j != i} 1/(r_i - r_j) = 1/(2*r_i) + 1/2
    
    We solve this system using Gauss-Seidel iteration with bisection
    for each coordinate, using the mean-field approximation where the
    other roots are held fixed.
    """
    roots = np.array(roots, dtype=float).copy()
    n = len(roots)
    
    for iteration in range(2000):
        new_roots = roots.copy()
        max_change = 0.0
        
        for i in range(n):
            other_roots = np.delete(roots, i)
            other_sorted = np.sort(other_roots)
            r_guess = roots[i]
            
            # Determine bracket: r_i must lie between adjacent other roots
            insert_pos = np.searchsorted(other_sorted, r_guess)
            
            if insert_pos == 0:
                left = 1e-10
                right = other_sorted[0] - 1e-8
            elif insert_pos == len(other_sorted):
                left = other_sorted[-1] + 1e-8
                right = max(200.0, r_guess * 2.0 + 20.0)
            else:
                left = other_sorted[insert_pos - 1] + 1e-8
                right = other_sorted[insert_pos] - 1e-8
            
            if left < 1e-10:
                left = 1e-10
            if right <= left:
                # Fallback: use a wider bracket around the guess
                left = max(1e-10, r_guess * 0.5)
                right = r_guess * 1.5 + 5.0
            
            # Define the equilibrium equation
            def f(r):
                if r <= 1e-14:
                    return 1e15
                s = np.sum(1.0 / (r - other_roots))
                return double_root_factor * s - 1.0/(2.0*r) - 0.5
            
            # Use bisection with sign change detection
            found = False
            for attempt in range(200):
                try:
                    f_left = f(left)
                    f_right = f(right)
                    
                    if not (np.isfinite(f_left) and np.isfinite(f_right)):
                        left = max(1e-10, left - 0.5)
                        right = right + 0.5
                        continue
                    
                    if f_left * f_right < 0:
                        # Sign change: use bisection
                        for bisection_iter in range(200):
                            mid = 0.5 * (left + right)
                            f_mid = f(mid)
                            if not np.isfinite(f_mid):
                                left = mid + 1e-10
                                continue
                            if abs(f_mid) < 1e-15:
                                left = mid
                                right = mid
                                break
                            if f_left * f_mid < 0:
                                right = mid
                                f_right = f_mid
                            else:
                                left = mid
                                f_left = f_mid
                            if right - left < 1e-14:
                                break
                        r_new = 0.5 * (left + right)
                        new_roots[i] = r_new
                        found = True
                        break
                    elif abs(f_left) < 1e-14:
                        new_roots[i] = left
                        found = True
                        break
                    elif abs(f_right) < 1e-14:
                        new_roots[i] = right
                        found = True
                        break
                    else:
                        # No sign change: expand bracket
                        if f_left > 0 and f_right > 0:
                            # Both positive: need to go right
                            right = right + 1.0
                        elif f_left < 0 and f_right < 0:
                            # Both negative: need to go left
                            left = max(1e-10, left - 1.0)
                        else:
                            left = max(1e-10, left - 0.5)
                            right = right + 0.5
                except Exception:
                    left = max(1e-10, left - 0.5)
                    right = right + 0.5
            
            if not found:
                new_roots[i] = r_guess
            
            max_change = max(max_change, abs(new_roots[i] - roots[i]))
        
        roots = new_roots
        
        if max_change < 1e-15:
            break
    
    roots = np.sort(roots)
    roots = np.maximum(roots, 1e-10)
    
    return roots

def _mean_field_potential(roots):
    """
    Compute the mean-field potential energy for the Dyson gas.
    E = sum_i V(r_i) + (beta/2) * sum_{i<j} -log|r_i - r_j|
    For beta=2, V(x) = x (Laguerre linear confinement)
    """
    roots = np.array(roots, dtype=float)
    n = len(roots)
    
    # Confinement energy
    e_conf = np.sum(roots)
    
    # Repulsion energy (beta=2)
    e_rep = 0.0
    for i in range(n):
        for j in range(i+1, n):
            e_rep += -math.log(abs(roots[i] - roots[j]))
    
    return e_conf + e_rep

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses Dyson gas mean-field fixed point with double-root modification.
    
    The equilibrium is derived from the asymptotic limit of Laguerre polynomial zeros
    where the density satisfies a log-concave potential with:
    - Linear confining potential V(x) = x (from e^{-x} in Laguerre weight)
    - Logarithmic repulsion with exponent 2 (double roots => beta=2)
    - Boundary term from the x^alpha factor (alpha=0 for standard Laguerre)
    
    The fixed point equation for each root r_i is:
      2 * sum_{j != i} 1/(r_i - r_j) = 1/(2*r_i) + 1/2
    """
    # Start with Laguerre zeros as initial guess (good asymptotic approximation)
    init_roots = _compute_laguerre_zeros(12, alpha=0)
    
    # Solve the modified Dyson gas equilibrium for double roots
    # The double root factor of 2 in the repulsion term accounts for
    # the squared factor in the polynomial P(x) = prod (x - r_i)^2
    roots = _dyson_gas_equilibrium(init_roots, double_root_factor=2.0)
    
    # Verify constraints
    if np.any(roots <= 0) or np.any(np.diff(roots) <= 0):
        # Fallback to Laguerre zeros
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
    print("Seed raw score:", _compute_score(seed_roots))
