import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import eval_chebyt
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Precompute Chebyshev basis
    # We map the discrete indices to [-1, 1] to use standard Chebyshev polynomials.
    # This enforces a smooth structure on the step heights h.
    x = np.linspace(-1, 1, N)
    K_basis = 15
    basis_cheb = np.zeros((N, K_basis + 1))
    for k in range(K_basis + 1):
        basis_cheb[:, k] = eval_chebyt(k, x)
    
    # Precompute sums of basis functions to quickly check for zero sum
    basis_sums = np.sum(basis_cheb, axis=0)

    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.any(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    def objective_coeffs(c):
        """
        Objective function for optimization over Chebyshev coefficients.
        Uses a smooth surrogate for the max operation to allow gradient-based methods
        to work effectively in the polish phase, but for DE we can use the exact max
        as it is robust. However, to maintain consistency and allow potential
        hybridization, we stick to the exact max for DE as it is the true objective.
        """
        if not np.all(np.isfinite(c)):
            return float('inf')
        
        h = c @ basis_cheb.T
        
        # Quick check for sum
        s = c @ basis_sums
        if abs(s) < 1e-9:
            return float('inf')
            
        if not np.all(np.isfinite(h)):
            return float('inf')

        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    # 2. Global Search: Differential Evolution on the coefficient space
    # The dimension is K_basis + 1 = 16.
    # We need to ensure the sum of h is not zero. 
    # h = c @ basis.T. Sum(h) = c @ sum(basis, axis=0).
    # basis_sums[0] is N (sum of ones). basis_sums[k>0] is roughly 0 for Chebyshev on uniform grid.
    # Specifically, sum of T_k(x) over uniform x in [-1,1] is 0 for odd k, and non-zero for even k > 0?
    # Actually, integral of T_k is 0 for odd k. Discrete sum is close to 0.
    # To be safe, we don't restrict c0 too tightly, but we can bias it.
    
    # Bounds for coefficients
    # c0 controls the mean. If mean is 0, sum is 0.
    # We want to avoid sum=0.
    # Let's set bounds for c0 to [0.1, 5.0] to ensure positive mean roughly, 
    # and [-5.0, 5.0] for others.
    # Wait, the problem allows negative heights. But sum(h) must not be zero.
    # If we force c0 > 0, sum(h) will likely be positive and non-zero.
    
    bounds = [(0.5, 5.0)] + [(-5.0, 5.0)] * K_basis
    
    # DE Parameters
    # popsize = 20 * dim = 320
    # maxiter = 100
    # This should be fast enough for dim 16.
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    try:
        # Use differential_evolution
        # seed=42 for reproducibility
        res = differential_evolution(
            objective_coeffs, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-5,
            polish=False # We will do our own polish
        )
        
        if res.fun < best_val:
            c_best = res.x
            h_best = c_best @ basis_cheb.T
            if np.all(np.isfinite(h_best)) and abs(np.sum(h_best)) > 1e-9:
                best_val = res.fun
                best_h = h_best
    except Exception:
        pass

    # 3. Local Polish: Multi-start L-BFGS-B with Smooth Surrogate
    # The exact objective has a kink at the max. 
    # We can use a soft-max approximation for the gradient.
    # max(z) approx log-sum-exp(z / tau) * tau
    # However, the objective is abs(2*N*max(z)/S^2).
    # Let's define a smooth objective for the polish phase.
    
    def smooth_objective(c):
        if not np.all(np.isfinite(c)):
            return float('inf')
        h = c @ basis_cheb.T
        s = c @ basis_sums
        if abs(s) < 1e-9:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        
        z = np.convolve(h, h)
        
        # Smooth max using log-sum-exp
        # To avoid overflow, subtract max
        z_max = np.max(z)
        if z_max > 100: # Scale if necessary, though z values are usually moderate
            # Actually, z can be large if h is large.
            # Let's just use a fixed temperature if we want to be safe, 
            # or rely on the fact that h is normalized by s^2 in the final value, 
            # but here we are optimizing c directly.
            # The value is scale invariant. We can fix the scale of h to simplify?
            # No, c determines scale.
            # Let's just use log-sum-exp carefully.
            z_shifted = z - z_max
            log_sum_exp = z_max + np.log(np.sum(np.exp(z_shifted)))
            max_z_smooth = log_sum_exp
        else:
            z_shifted = z - z_max
            log_sum_exp = z_max + np.log(np.sum(np.exp(z_shifted)))
            max_z_smooth = log_sum_exp
            
        # Temperature tau
        tau = 1.0
        # Actually, log-sum-exp is a smooth upper bound on max.
        # We want to minimize the value. Minimizing the upper bound is a good strategy.
        
        val = abs(2 * N * max_z_smooth / (s * s))
        return val

    # Generate multiple starting points for local polish
    # 1. The result from DE
    # 2. Random Chebyshev coefficients
    # 3. Perturbed best_h
    
    starting_points = []
    
    # Add DE result
    if 'c_best' in locals():
        starting_points.append(c_best.copy())
    
    # Add random starts
    rng = np.random.default_rng(42)
    num_starts = 10
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
        c_rand = np.zeros(K_basis + 1)
        c_rand[0] = rng.uniform(0.5, 2.0)
        c_rand[1:] = rng.uniform(-2.0, 2.0, size=K_basis)
        starting_points.append(c_rand)
        
        # Also try a "bump" like function
        # Gaussian in Chebyshev space? Hard.
        # Just random is fine for multi-start.

    # Polish each starting point
    for c_start in starting_points:
        if time.time() > deadline - 5:
            break
            
        try:
            res_local = minimize(
                smooth_objective, 
                c_start, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_local.fun < best_val:
                # Verify with exact objective
                h_check = res_local.x @ basis_cheb.T
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check
        except Exception:
            pass

    # 4. Final Polish on the best found h using direct gradient descent on h?
    # No, h is 400-dim. Chebyshev is better.
    # Let's do one more DE run with tighter bounds if time permits?
    # Or just return best_h.

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
