"""
Strategy: Constant-amplitude square wave with low-order polynomial windowing.

We represent the step heights as:
    h[n] = sign[n] * A * W[n]
where sign[n] is a fixed square wave pattern (alternating blocks or single sign),
A is a global amplitude, and W[n] is a low-order polynomial window function 
(e.g., polynomial in (n - N/2)/N) designed to taper the edges.

The key insight is that the autoconvolution of a rectangular pulse has large 
sidelobes (Sinc-like). By multiplying by a smooth polynomial window (like a 
raised cosine or Chebyshev-based taper), we can significantly reduce the 
off-peak autoconvolution values while keeping the sum large (by choosing 
sign patterns that don't cancel out too much).

Since the objective is:
    value = abs(2 * N * max(conv(h,h)) / sum(h)**2)

We can fix the sign pattern and amplitude to 1, and optimize the polynomial 
coefficients of the window W[n] to minimize this ratio. We also allow the 
sign pattern to be one of a few candidates (all positive, alternating, 
block-based) and let the optimizer choose the best one via discrete search 
or by incorporating a sign-flip parameter.

To keep it continuous and differentiable for optimization, we will:
1. Fix the sign pattern to all positive (since negative values can be 
   absorbed into the window if we allow negative window values, but 
   typically we want positive heights for max sum). Actually, let's allow 
   the window to be non-negative and fix sign to +1. If the optimizer 
   finds that a different sign pattern is better, we can add that later.
   For now, assume h[n] = A * W[n] with W[n] >= 0.

2. W[n] = sum_{k=0}^{K} c_k * ((n - N/2) / (N/2))^k
   We enforce W[n] >= 0 by using W[n] = max(0, poly(n)) or by constraining 
   coefficients. A better approach: use a basis of non-negative polynomials 
   or simply optimize c_k and then project to non-negative.

   Actually, a simpler and more robust approach: 
   Use a polynomial window of degree K, and optimize the coefficients 
   to minimize the objective. We will use scipy.optimize.minimize with 
   L-BFGS-B or Nelder-Mead.

   To ensure W[n] is positive, we can parameterize W[n] = exp(sum c_k * phi_k(n)) 
   or simply W[n] = 1 + sum c_k * T_k(n) where T_k are Chebyshev polynomials 
   and we constrain the result to be positive.

   Let's use a simpler parameterization: 
   W[n] = 1 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4, where x = (2n - N + 1) / N
   and we enforce W[n] > 0 by adding a large positive constant if needed, 
   or by using W[n] = 1 + 0.5 * (1 - cos(pi * x)) * (1 + sum c_k * T_k(x))

   Actually, let's just use a polynomial and clip to a small positive value 
   during evaluation, and let the optimizer figure out the rest.

3. We will also allow a global offset to shift the window.

Parameters to optimize:
- c0, c1, c2, c3, c4: polynomial coefficients
- offset: DC offset added to the window
- scale: overall scaling (though this cancels in the ratio, so we can fix it)

Since scale cancels, we can fix A=1.

Let's define:
    x = (n - (N-1)/2) / ((N-1)/2)  # in [-1, 1]
    W[n] = 1 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + offset
    h[n] = W[n]

We need h[n] to be finite and sum(h) != 0.

We will use scipy.optimize.differential_evolution for global search over 
the 5-6 parameters, then refine with Nelder-Mead.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_DEADLINE = 140.0  # wall-clock limit with safety margin

def _h_from_poly(params):
    """
    Generate step heights from polynomial window parameters.
    params = [c1, c2, c3, c4, offset]
    """
    c1, c2, c3, c4, offset = params
    n = np.arange(N, dtype=float)
    x = (n - (N - 1) / 2.0) / ((N - 1) / 2.0)  # in [-1, 1]
    
    # Polynomial window
    W = 1.0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + offset
    
    # Ensure positivity to avoid sign flips that might reduce sum
    # We'll clip to a small positive value
    W = np.maximum(W, 0.01)
    
    return W

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective(params):
    """Objective function for optimization."""
    h = _h_from_poly(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    return val

def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Best solution so far
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0

    # Define bounds for differential evolution
    # c1, c2, c3, c4: polynomial coefficients
    # offset: DC offset
    bounds = [
        (-1.0, 1.0),   # c1
        (-1.0, 1.0),   # c2
        (-1.0, 1.0),   # c3
        (-1.0, 1.0),   # c4
        (-1.0, 1.0),   # offset
    ]

    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                polish=False,
                seed=42,
                init='sobol',
            )
            if de_result.success or _objective(de_result.x) < best_val:
                h_de = _h_from_poly(de_result.x)
                val_de = _evaluate(h_de)
                if np.isfinite(val_de) and val_de < best_val:
                    best_val = val_de
                    best_h = h_de
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            x0 = de_result.x.copy()
            nm_result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
            )
            if np.all(np.isfinite(nm_result.x)):
                h_nm = _h_from_poly(nm_result.x)
                val_nm = _evaluate(h_nm)
                if np.isfinite(val_nm) and val_nm < best_val:
                    best_val = val_nm
                    best_h = h_nm
        except Exception:
            pass

    # Phase 3: Try different initial points for multi-start
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            x0 = rng.uniform(-0.5, 0.5, size=5)
            try:
                nm = minimize(
                    _objective, x0, method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8},
                )
                if np.all(np.isfinite(nm.x)):
                    h_c = _h_from_poly(nm.x)
                    v_c = _evaluate(h_c)
                    if np.isfinite(v_c) and v_c < best_val:
                        best_val = v_c
                        best_h = h_c
            except Exception:
                pass

    # Phase 4: Final greedy local search on the best h
    if time.time() < deadline - 3:
        best_val = _evaluate(best_h)
        rng = np.random.default_rng(999)
        for _ in range(50000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = best_h.copy()
            h_new[idx] += rng.normal(0, 0.02)
            v_new = _evaluate(h_new)
            if np.isfinite(v_new) and v_new < best_val - 1e-12:
                best_val = v_new
                best_h = h_new

    # Final validation
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)

    return best_h
