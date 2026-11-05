"""Combined program: DFT-spectral parameterization with smooth-max objective,
differential evolution for global search, and L-BFGS-B for local refinement.

The key structural idea is to represent h as a low-frequency DFT combination
(h[i] = dc + sum_k r_k * cos(2*pi*k*i/N + p_k)) which naturally produces
sequences with controlled autocorrelation structure. The smooth-max
approximation makes the objective differentiable, enabling gradient-based
refinement after a global search phase.

Speedup: use FFT-based autoconvolution (O(N log N) vs O(N^2)) to enable
more optimization iterations within the time budget.
"""
import numpy as np
import time
import sys

N = 400

# Precompute FFT size for autoconvolution (need at least 2*N-1)
_FFT_SIZE = 1  # will be set in construct
_FFT_SIZE = 1024  # next power of 2 >= 2*400-1 = 799

# EVOLVE-BLOCK-START

def _autoconvolve_fft(h):
    """Compute full autoconvolution h*h via FFT. O(N log N)."""
    size = 1024
    H = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(H * H, n=size)
    return conv[:2 * len(h) - 1]

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s kill

    n = np.arange(N, dtype=np.float64)

    def compute_value(h):
        """Compute the exact objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = _autoconvolve_fft(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.05):
        """Smooth approximation of max via log-sum-exp."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))

    def dft_heights(params):
        """Build h from DFT parameters.
        params: [dc, r1, p1, r2, p2, ..., rK, pK]
        h[i] = dc + sum_k r_k * cos(2*pi*k*i/N + p_k)
        """
        dc = params[0]
        h = np.full(N, dc, dtype=np.float64)
        K = (len(params) - 1) // 2
        for k in range(1, K + 1):
            r = params[1 + 2 * (k - 1)]
            p = params[2 + 2 * (k - 1)]
            h += r * np.cos(2.0 * np.pi * k * n / N + p)
        return h

    def dft_objective(params, tau=0.05):
        """Differentiable objective for DFT parameterization."""
        h = dft_heights(params)
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = _autoconvolve_fft(h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    def direct_objective(h, tau=0.05):
        """Differentiable objective in direct h-space."""
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = _autoconvolve_fft(h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # Initialize with constant ones (value = 2.0, score = 1.0)
    best_h = np.ones(N, dtype=np.float64)
    best_val = 2.0

    # Phase 1: Differential evolution in DFT space (global search)
    if time.time() < DEADLINE - 40.0:
        try:
            from scipy.optimize import differential_evolution

            K = 5  # number of DFT harmonics
            bounds = [(-2.0, 2.0)]  # dc
            for k in range(1, K + 1):
                bounds.append((-1.5, 1.5))  # amplitude
                bounds.append((0.0, 2.0 * np.pi))  # phase

            def _de_obj(params):
                if time.time() > DEADLINE - 10.0:
                    return 1e10
                return dft_objective(params, tau=0.05)

            de_result = differential_evolution(
                _de_obj,
                bounds,
                maxiter=80,
                popsize=20,
                seed=42,
                tol=1e-6,
                polish=False,
                init='sobol',
                mutation=(0.3, 0.8),
                recombination=0.9,
            )

            if de_result.fun < best_val:
                h_cand = dft_heights(de_result.x)
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"DE DFT: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DE error: {e}", file=sys.stderr)

    # Phase 2: L-BFGS-B refinement in DFT parameter space
    if time.time() < DEADLINE - 25.0:
        try:
            from scipy.optimize import minimize

            K = 5
            # Start from best DFT params if we have them, else from a reasonable init
            if 'de_result' in dir() and de_result.fun < best_val + 0.1:
                x0 = de_result.x.copy()
            else:
                x0 = np.zeros(1 + 2 * K)
                x0[0] = 1.0  # dc = 1
                # Small perturbations
                rng = np.random.RandomState(7)
                for k in range(1, K + 1):
                    x0[1 + 2*(k-1)] = rng.uniform(-0.3, 0.3)
                    x0[2 + 2*(k-1)] = rng.uniform(0, 2*np.pi)

            def _dft_obj_local(params):
                if time.time() > DEADLINE - 5.0:
                    return 1e10
                return dft_objective(params, tau=0.02)

            result = minimize(
                _dft_obj_local,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-11}
            )

            if result.fun < best_val:
                h_cand = dft_heights(result.x)
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"DFT L-BFGS: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT L-BFGS error: {e}", file=sys.stderr)

    # Phase 3: Multiple restarts in DFT space with different initializations
    if time.time() < DEADLINE - 15.0:
        try:
            from scipy.optimize import minimize

            K = 5
            rng = np.random.RandomState(123)

            for restart in range(8):
                if time.time() > DEADLINE - 8.0:
                    break

                x0 = np.zeros(1 + 2 * K)
                x0[0] = rng.uniform(0.5, 2.0)
                for k in range(1, K + 1):
                    x0[1 + 2*(k-1)] = rng.uniform(-1.0, 1.0)
                    x0[2 + 2*(k-1)] = rng.uniform(0, 2*np.pi)

                def _obj_r(params):
                    if time.time() > DEADLINE - 5.0:
                        return 1e10
                    return dft_objective(params, tau=0.03)

                result = minimize(
                    _obj_r,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11}
                )

                if result.fun < best_val:
                    h_cand = dft_heights(result.x)
                    val_cand = compute_value(h_cand)
                    if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                        best_h = h_cand.copy()
                        best_val = val_cand
                        print(f"DFT restart {restart}: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"DFT restarts error: {e}", file=sys.stderr)

    # Phase 4: Direct h-space refinement with L-BFGS-B
    if time.time() < DEADLINE - 10.0:
        try:
            from scipy.optimize import minimize

            def _direct_obj(h):
                if time.time() > DEADLINE - 5.0:
                    return 1e10
                return direct_objective(h, tau=0.01)

            result = minimize(
                _direct_obj,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-11}
            )

            if result.fun < best_val:
                h_cand = result.x
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"Direct L-BFGS: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Direct L-BFGS error: {e}", file=sys.stderr)

    # Phase 5: Tight polish with smaller tau
    if time.time() < DEADLINE - 3.0 and best_val < 2.0:
        try:
            from scipy.optimize import minimize

            def _tight_obj(h):
                if time.time() > DEADLINE - 2.0:
                    return 1e10
                return direct_objective(h, tau=0.003)

            result = minimize(
                _tight_obj,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12}
            )

            if result.fun < best_val:
                h_cand = result.x
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"Tight polish: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Tight polish error: {e}", file=sys.stderr)

    # Phase 6: Final ultra-tight polish
    if time.time() < DEADLINE - 1.5 and best_val < 1.5:
        try:
            from scipy.optimize import minimize

            def _ultra_obj(h):
                if time.time() > DEADLINE - 1.0:
                    return 1e10
                return direct_objective(h, tau=0.001)

            result = minimize(
                _ultra_obj,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-13}
            )

            if result.fun < best_val:
                h_cand = result.x
                val_cand = compute_value(h_cand)
                if val_cand < best_val and np.all(np.isfinite(h_cand)) and abs(np.sum(h_cand)) > 1e-6:
                    best_h = h_cand.copy()
                    best_val = val_cand
                    print(f"Ultra polish: {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"Ultra polish error: {e}", file=sys.stderr)

    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N, dtype=np.float64)

    return best_h

# EVOLVE-BLOCK-END
