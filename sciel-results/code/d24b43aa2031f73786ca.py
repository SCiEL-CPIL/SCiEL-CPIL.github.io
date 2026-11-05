"""
Strategy: Structured Cubic Chirp Design
Instead of optimizing 400 free variables, we parameterize h as a cubic chirp:
h[i] = A * cos(2*pi*(a*i^3 + b*i^2 + c*i + d) / N + phi) + DC
This structure is known to have good autocorrelation properties (low sidelobes).
We use differential evolution to find the optimal coefficients (a, b, c, d, phi, A, DC).
Then we perform a local refinement on the full h vector to correct for discretization errors.
"""
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        # Clip to avoid overflow in exp
        x_shifted = (x - x_max) / tau
        x_shifted = np.clip(x_shifted, -100, 100)
        return x_max + tau * np.log(np.sum(np.exp(x_shifted)))

    def objective_signed(params, tau=0.05):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def generate_chirp(params):
        """
        Generate h from chirp parameters.
        params: [a, b, c, d, phi, A, DC]
        h[i] = A * cos(2*pi*(a*i^3 + b*i^2 + c*i + d) / N + phi) + DC
        """
        i = np.arange(N, dtype=float)
        # Normalize i to [0, 1) for phase calculation
        t = i / N
        phase = 2 * np.pi * (params[0] * t**3 + params[1] * t**2 + params[2] * t + params[3]) + params[4]
        h = params[5] * np.cos(phase) + params[6]
        return h

    def chirp_objective(params):
        """Objective for chirp parameters."""
        h = generate_chirp(params)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        # Use a fixed moderate tau for DE to keep it smooth enough for global search
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: Global search using Differential Evolution on Chirp Parameters
    if time.time() < deadline - 40.0:
        try:
            # Bounds for chirp coefficients
            # a, b, c, d: phase coefficients. Larger values mean faster frequency sweep.
            # We limit them to avoid excessive aliasing/high frequency content that might be hard to optimize locally later.
            bounds = [
                (-5.0, 5.0),   # a (cubic)
                (-5.0, 5.0),   # b (quadratic)
                (-5.0, 5.0),   # c (linear)
                (-5.0, 5.0),   # d (constant phase)
                (0.0, 2.0 * np.pi), # phi (initial phase)
                (-2.0, 2.0),   # A (amplitude)
                (-2.0, 2.0)    # DC (offset)
            ]
            
            # Ensure DC is large enough to keep sum non-zero? 
            # The objective penalizes small sum. 
            # A good initial guess is around DC=1, A=0.5.
            
            de_result = differential_evolution(
                chirp_objective,
                bounds,
                maxiter=60,
                popsize=20,
                seed=42,
                tol=1e-5,
                polish=True,
                init='sobol'
            )
            
            if de_result.fun < best_val:
                h_chirp = generate_chirp(de_result.x)
                val_chirp = compute_value(h_chirp)
                if val_chirp < best_val and np.all(np.isfinite(h_chirp)) and abs(np.sum(h_chirp)) > 1e-6:
                    best_h = h_chirp.copy()
                    best_val = val_chirp
                    print(f"Chirp DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Chirp DE error: {e}", file=sys.stderr)

    # Phase 2: Local Refinement in full h-space
    # The chirp structure provides a very good starting point.
    # Now we allow all 400 dimensions to adjust to minimize the exact peak (using smooth max).
    if time.time() < deadline - 20.0:
        try:
            # Use a tighter tau for the local optimization to approximate the max better
            def objective_tight(params, tau=0.01):
                h = params
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10 + (1e-4 - abs(s)) * 1e6
                conv = np.convolve(h, h)
                sm = smooth_max(conv, tau=tau)
                return 2.0 * N * sm / (s * s)

            result = minimize(
                objective_tight,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Local Refine: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Local Refine error: {e}", file=sys.stderr)

    # Phase 3: Final Polish with even tighter smooth max
    if time.time() < deadline - 5.0 and best_val < 2.0:
        try:
            def objective_very_tight(params, tau=0.005):
                h = params
                s = np.sum(h)
                if abs(s) < 1e-4:
                    return 1e10 + (1e-4 - abs(s)) * 1e6
                conv = np.convolve(h, h)
                sm = smooth_max(conv, tau=tau)
                return 2.0 * N * sm / (s * s)

            result = minimize(
                objective_very_tight,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
            )
            
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Final Polish: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Final Polish error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
