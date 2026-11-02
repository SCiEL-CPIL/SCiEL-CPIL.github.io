"""Rewritten program using Basin Hopping for global search on the uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, basinhopping

def _get_hermite_norm_sq(n):
    """Returns 2^n * n! (excluding sqrt(pi) which cancels in the ratio)."""
    return (2.0 ** n) * math.factorial(n)

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product Delta_x * Delta_xi for the test function.
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    Constraint: f(0) = 0 => P(0) = 0.
    H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128 (Physicists' Hermite).
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (i)^n * H_n(xi)*exp(-xi^2/2).
    For even n in {0, 4, 8, 12}, (i)^n = 1.
    Thus the function is self-Fourier (up to phase 1).
    Delta_x^2 = Delta_xi^2 = <x^2> = integral x^2 f^2 dx / ||f||_2^2.
    Product = <x^2>.
    """
    # Determine a12 from P(0) = 0
    # a0*1 + a4*12 + a8*1680 + a12*2032128 = 0
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # 1. Calculate ||f||_2^2
    norms_sq = np.array([_get_hermite_norm_sq(d) for d in degrees])
    norm_total = np.sum(coeffs**2 * norms_sq)
    
    if norm_total < 1e-30:
        return 1e10
        
    # 2. Calculate <x^2> = integral x^2 f^2 dx / ||f||_2^2
    # Orthogonality of Hermite polynomials: <H_n H_m> = 0 if n != m.
    # x^2 H_n(x) = 1/4 H_{n+2}(x) + n H_n(x) + n(n-1) H_{n-2}(x)
    # We need <x^2 f, f>. Since f is a sum of H_0, H_4, H_8, H_12,
    # x^2 f will contain terms H_{n+2}, H_n, H_{n-2}.
    # Only terms with the same degree as components of f contribute to the inner product.
    
    # Contribution from a_n H_n:
    # The H_n part of x^2 (a_n H_n) is a_n * n * H_n.
    # This overlaps with a_n H_n in f.
    # Integral is a_n * (a_n * n) * ||H_n||^2.
    
    # Are there cross terms?
    # x^2 (a_0 H_0) = 1/4 H_2. No H_0, H_4... in f. No contribution.
    # x^2 (a_4 H_4) = 1/4 H_6 + 4 H_4 + 12 H_2. Overlaps with a_4 H_4.
    # x^2 (a_8 H_8) = 1/4 H_10 + 8 H_8 + 56 H_6. Overlaps with a_8 H_8.
    # x^2 (a_12 H_12) = 1/4 H_14 + 12 H_12 + 132 H_10. Overlaps with a_12 H_12.
    
    # So, <x^2 f, f> = 4 a_4^2 ||H_4||^2 + 8 a_8^2 ||H_8||^2 + 12 a_12^2 ||H_12||^2
    
    norm_H4_sq = _get_hermite_norm_sq(4)
    norm_H8_sq = _get_hermite_norm_sq(8)
    norm_H12_sq = _get_hermite_norm_sq(12)
    
    integral_x2_f2 = (4.0 * a4**2 * norm_H4_sq + 
                      8.0 * a8**2 * norm_H8_sq + 
                      12.0 * a12**2 * norm_H12_sq)
    
    variance = integral_x2_f2 / norm_total
    
    return variance

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # 30s safety margin for 210s limit
    
    # Seed values from "Prior state of the art"
    # These are known good starting points
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _compute_bound(*x0)
    best_x = x0.copy()
    
    # 1. Initial local refinement with L-BFGS-B to ensure we start from a local minimum
    try:
        res = minimize(
            _compute_bound,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # 2. Global search using Basin Hopping
    # Basin Hopping is robust for low-dimensional non-convex problems.
    # We define a step function that perturbs the current point.
    def step(x):
        # Random step in Gaussian distribution
        return x + np.random.normal(0, 0.5, size=3)
    
    # We will run Basin Hopping. It internally performs local minimization after each step.
    # We use the L-BFGS-B minimizer for the local steps within BH.
    
    try:
        # Set up Basin Hopping
        # niter is the number of large steps. 
        # min_interval is the minimum number of local minimization steps between large steps (default 5).
        # We use a tight local optimizer inside BH to ensure each "basin" is explored well.
        
        bh = basinhopping(
            _compute_bound, 
            x0, 
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'ftol': 1e-12}},
            stepfun=step,
            niter=50, # 50 large steps
            T=1.0,     # Temperature for Metropolis acceptance
            accept='opt' # Only accept if the local minimum is better (greedy) or use 'opt' for standard BH. 
                         # 'opt' accepts if new min is better. 'min' accepts with probability exp(-delta/T).
                         # For finding global min, 'opt' is often sufficient and faster.
                         # Let's use 'opt' to be safe and fast, but 'min' might escape better.
                         # Actually, standard BH uses 'min' (Metropolis). Let's stick to standard behavior.
                         # scipy basinhopping default accept is 'opt'? No, default is 'min' (Metropolis).
                         # Let's explicitly set 'opt' to force downhill moves only? 
                         # No, Metropolis allows uphill moves to escape. Let's use default 'min'.
        )
        
        # Run it
        # We need to monitor time. Basin Hopping doesn't have a built-in timeout.
        # We can wrap the objective function to check time, but BH calls it many times.
        # Better to just run a limited number of iterations and check time after.
        
        # To be safe with time, we can run BH in a loop or just run a moderate number of iterations.
        # 50 iterations of BH with L-BFGS (100 iter each) is roughly 50 * (100 * cost_func) + overhead.
        # cost_func is very fast (analytic). So this should be well under 10 seconds.
        
        res_bh = bh.run()
        
        if res_bh.fval < best_val:
            best_val = res_bh.fval
            best_x = res_bh.x
            
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x
    
    # 3. Final high-precision polish
    # Use Nelder-Mead or L-BFGS-B with very tight tolerances on the best point found.
    try:
        res_final = minimize(
            _compute_bound,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15}
        )
        if res_final.fun < best_val:
            best_val = res_final.fun
            best_x = res_final.x
    except Exception:
        pass

    # Optional: Multi-start local refinement if time permits
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        # Try a few random starts to ensure we didn't miss a basin
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            trial_x = rng.normal(0, 1.0, size=3)
            try:
                res_trial = minimize(
                    _compute_bound,
                    trial_x,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                if res_trial.fun < best_val:
                    best_val = res_trial.fun
                    best_x = res_trial.x
            except Exception:
                pass

    return best_x
# EVOLVE-BLOCK-END
