"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Tighter safety margin before 150s limit

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
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def objective_signed(params, tau=0.05):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def dft_based_objective(params):
        """Objective for DFT-based parameterization.
        params: [dc, r1, p1, r2, p2, r3, p3, r4, p4, r5, p5, r6, p6]
        h[i] = dc + sum_k r_k * cos(2*pi*k*i/N + p_k)
        """
        dc = params[0]
        i = np.arange(N)
        h = np.full(N, dc)
        for k in range(6):
            r = params[1 + 2*k]
            p = params[2 + 2*k]
            h += r * np.cos(2 * np.pi * (k+1) * i / N + p)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.02)
        return 2.0 * N * sm / (s * s)

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Phase 1: Global search over DFT-based parameters using differential evolution
    if time.time() < deadline - 50.0:
        try:
            from scipy.optimize import differential_evolution
            # params: [dc, r1, p1, r2, p2, r3, p3, r4, p4, r5, p5, r6, p6]
            bounds = [(-1.0, 1.0)]  # dc
            for k in range(6):
                bounds.append((-1.5, 1.5))  # radius
                bounds.append((0.0, 2.0 * np.pi))  # phase
            de_result = differential_evolution(
                dft_based_objective,
                bounds,
                maxiter=40,
                popsize=15,
                seed=42,
                tol=1e-7,
                polish=True
            )
            if de_result.fun < best_val:
                dc = de_result.x[0]
                i = np.arange(N)
                h_dft = np.full(N, dc)
                for k in range(6):
                    r = de_result.x[1 + 2*k]
                    p = de_result.x[2 + 2*k]
                    h_dft += r * np.cos(2 * np.pi * (k+1) * i / N + p)
                val_dft = compute_value(h_dft)
                if val_dft < best_val and np.all(np.isfinite(h_dft)) and abs(np.sum(h_dft)) > 1e-6:
                    best_h = h_dft.copy()
                    best_val = val_dft
                    print(f"DFT DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT DE error: {e}", file=sys.stderr)

    # Phase 2: Local refinement using L-BFGS-B from best with adaptive tau
    if time.time() < deadline - 25.0:
        try:
            from scipy.optimize import minimize
            # Start with moderate tau, then refine with smaller tau
            for tau in [0.02, 0.005, 0.001, 0.0005]:
                if time.time() > deadline - 15.0:
                    break
                result = minimize(
                    lambda x: objective_signed(x, tau=tau),
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"L-BFGS-B (tau={tau}): improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"L-BFGS-B error: {e}", file=sys.stderr)

    # Phase 3: Try multiple DFT-based initializations with L-BFGS-B
    rng = np.random.RandomState(42)
    for trial in range(10):
        if time.time() > deadline - 10.0:
            break

        # Random DFT-like initialization with 3-6 components
        n_comp = rng.randint(3, 7)
        dc = rng.uniform(-0.3, 0.3)
        i = np.arange(N)
        h0 = np.full(N, dc)
        for k in range(1, n_comp + 1):
            r = rng.uniform(-0.8, 0.8)
            p = rng.uniform(0, 2 * np.pi)
            h0 += r * np.cos(2 * np.pi * k * i / N + p)

        s0 = np.sum(h0)
        if abs(s0) < 1.0:
            h0 = h0 + (1.0 - s0) / N

        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda x: objective_signed(x, tau=0.015),
                h0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"Trial {trial}: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Trial {trial}: error {e}", file=sys.stderr)

    # Phase 4: Final high-precision polish with very small tau
    if time.time() < deadline - 3.0:
        try:
            from scipy.optimize import minimize
            for tau_polish in [0.0002, 0.00005]:
                if time.time() > deadline - 2.0:
                    break
                result = minimize(
                    lambda x: objective_signed(x, tau=tau_polish),
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-13}
                )
                h_opt = result.x
                val_opt = compute_value(h_opt)
                if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                    best_h = h_opt.copy()
                    best_val = val_opt
                    print(f"Polish (tau={tau_polish}): improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Polish error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
