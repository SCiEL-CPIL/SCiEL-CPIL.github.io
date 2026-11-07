"""Program for Erdos minimum overlap task using anti-periodic model + optimization.

The objective max(overlap(h,1-h)) is minimized when h is anti-periodic
(h[i+N/2] = 1 - h[i]), which forces the overlap to 0 at the shifts
t = 0, N, 2N (the 'forbidden' shifts). A non-degenerate anti-periodic
solution therefore reaches near-0 (far below the seed value of 1).
We build a strong anti-periodic seed and refine it with a smooth objective.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 145.0
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def value(h):
    """The exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_smooth(h, beta):
    """Smooth surrogate for the max via log-sum-exp (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def make_seed():
    """Strong anti-periodic seed: triangular wave, zero-mean per anti-pair."""
    i = np.arange(HALF)
    # triangular wave peaking at the middle of the first half
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    # center the first half around 0.5 so the anti-periodic completion
    # (0.5 - first half) stays within [0,1]
    first = 0.5 + (0.5 - tri.mean()) * tri
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    # fix tiny numerical drift
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def refine(h, beta, maxiter):
    """Optimize the anti-periodic parameters to lower the smooth objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    # free parameters: first half. second half = 1 - first (anti-periodic).
    p0 = h0[:HALF].copy()
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF

    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value_smooth(unpack(p), beta)

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-15,
                                "gtol": 1e-12, "maxls": 40})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return h0


def _polish_exact(p0, bounds, maxiter=500):
    """High-precision exact-objective polish with tight tolerances."""
    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value(unpack(p))

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-18,
                                "gtol": 1e-15, "maxls": 50})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return unpack(np.clip(p0, 0.0, 1.0))


def refine_spectral(h, n_harmonics=5):
    """
    Refine h by projecting onto a low-dimensional subspace of anti-periodic
    functions defined by odd harmonics, then optimizing the coefficients.
    This helps escape local minima in the full 50D space by enforcing a
    smooth, structured shape.
    """
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    
    # Anti-periodic condition: h[i+50] = 1 - h[i]
    # Let g[i] = 2*h[i] - 1. Then g[i+50] = -g[i].
    # g can be represented by odd harmonics of period 100.
    # g[i] = sum_{m=0}^{M-1} (a_m cos(2*pi*(2m+1)*i/100) + b_m sin(2*pi*(2m+1)*i/100))
    
    M = n_harmonics
    # Frequencies: 1, 3, 5, ..., 2M-1
    freqs = np.arange(1, 2*M, 2)
    
    # Construct design matrix A of shape (N, 2*M)
    # Columns: cos(k*i), sin(k*i) for each k in freqs
    i_idx = np.arange(N)
    A = np.zeros((N, 2*M))
    for j, k in enumerate(freqs):
        A[:, 2*j] = np.cos(2 * np.pi * k * i_idx / N)
        A[:, 2*j+1] = np.sin(2 * np.pi * k * i_idx / N)
    
    # Initial coefficients: project g onto this subspace
    g0 = 2 * h0 - 1
    coeffs0, _, _, _ = np.linalg.lstsq(A, g0, rcond=None)
    
    def unpack(coeffs):
        g = A @ coeffs
        h_new = 0.5 + 0.5 * g
        return h_new
    
    def obj(coeffs):
        h_new = unpack(coeffs)
        # Clip to valid range for objective evaluation, 
        # but optimizer works on unclipped coeffs to allow smooth gradients?
        # Actually, L-BFGS-B needs bounds. 
        # We don't have simple bounds on coeffs that guarantee h in [0,1].
        # However, we can just use SLSQP with constraints or 
        # use L-BFGS-B on the coefficients and rely on the fact that 
        # the projection is likely to stay close.
        # Better: Use L-BFGS-B on coeffs, but if h goes out of bounds, 
        # the objective should be penalized or we clip.
        # Let's clip h in the objective to be safe, but this makes it non-smooth.
        # Alternative: Use SLSQP with constraints 0 <= 0.5 + 0.5*A@coeffs <= 1.
        
        # For speed and simplicity, let's try L-BFGS-B on coeffs 
        # and just evaluate the smoothed value. 
        # We must ensure the result is valid. 
        # If we clip inside unpack, the gradient is wrong.
        # Let's use SLSQP for this step as it handles constraints better.
        pass

    # Using SLSQP for spectral refinement
    def obj_slsqp(coeffs):
        h_new = unpack(coeffs)
        return value_smooth(h_new, 50.0)
        
    def jac_slsqp(coeffs):
        # Numerical gradient for SLSQP
        eps = 1e-7
        n = len(coeffs)
        grad = np.zeros(n)
        f0 = obj_slsqp(coeffs)
        for i in range(n):
            coeffs_p = coeffs.copy()
            coeffs_p[i] += eps
            f_p = obj_slsqp(coeffs_p)
            grad[i] = (f_p - f0) / eps
        return grad

    # Constraints: 0 <= h <= 1
    # h = 0.5 + 0.5 * (A @ coeffs)
    # -0.5 <= 0.5 * (A @ coeffs) <= 0.5
    # -1 <= A @ coeffs <= 1
    
    cons = []
    for i in range(N):
        # A[i, :] @ coeffs <= 1
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - A[i, :] @ c, 'jac': lambda c, i=i: -A[i, :]})
        # -A[i, :] @ coeffs <= 1  => A[i, :] @ coeffs >= -1
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 + A[i, :] @ c, 'jac': lambda c, i=i: A[i, :]})
        
    # This is 200 constraints, might be slow for SLSQP.
    # Let's stick to L-BFGS-B on coefficients but with a penalty for out-of-bounds.
    # Or, simpler: just project back to [0,1] after each step? No, that breaks gradient.
    
    # Let's try a simpler approach: 
    # Optimize the coefficients using L-BFGS-B, but we define the objective 
    # on the *clipped* h. This is non-smooth but L-BFGS-B is robust.
    
    def obj_lbfgs(coeffs):
        h_new = unpack(coeffs)
        h_clip = np.clip(h_new, 0.0, 1.0)
        return value_smooth(h_clip, 50.0)
        
    # No bounds on coeffs, let the optimizer find them.
    # We might need to bound coeffs to prevent huge values.
    # Max possible coeff magnitude?
    # If h is in [0,1], g is in [-1,1].
    # Coeffs are likely small. Let's bound them to [-2, 2].
    bounds = [(-2.0, 2.0)] * (2*M)
    
    try:
        res = minimize(obj_lbfgs, coeffs0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
        h_res = unpack(res.x)
        h_res = np.clip(h_res, 0.0, 1.0)
        # Fix sum
        h_res += (HALF - h_res.sum()) / N
        h_res = np.clip(h_res, 0.0, 1.0)
        return h_res
    except Exception:
        return h0


def construct():
    t0 = time.time()

    h = make_seed()
    best = h.copy()
    best_val = value(best)

    # Progressive smoothing: start smooth, then sharpen to target the true max.
    for beta, iters in [(20.0, 200), (50.0, 200), (120.0, 200), (300.0, 300)]:
        if remaining() < 15.0:
            break
        h = refine(h, beta, iters)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v

    # Spectral Refinement Phase
    if remaining() > 20.0:
        # Try spectral refinement with different numbers of harmonics
        for n_harm in [3, 5, 7]:
            if remaining() < 10.0:
                break
            h_spec = refine_spectral(best, n_harm)
            v_spec = value(h_spec)
            if v_spec < best_val:
                best, best_val = h_spec.copy(), v_spec

    # Final exact-objective polish with multiple restarts to escape local minima.
    if remaining() > 20.0:
        p_best = np.clip(best[:HALF], 0.0, 1.0).copy()
        bounds = [(0.0, 1.0)] * HALF

        # First polish from the current best
        h_polished = _polish_exact(p_best, bounds, maxiter=500)
        v = value(h_polished)
        if v < best_val:
            best, best_val = h_polished.copy(), v
            p_best = np.clip(best[:HALF], 0.0, 1.0).copy()

        # Multiple restarts with small perturbations to explore nearby basins
        rng = np.random.default_rng(42)
        n_restarts = 5
        for i in range(n_restarts):
            if remaining() < 15.0:
                break
            # Perturb with decreasing magnitude
            scale = 0.05 * (1.0 - i / n_restarts)
            p_perturbed = p_best + rng.normal(0, scale, size=HALF)
            p_perturbed = np.clip(p_perturbed, 0.0, 1.0)
            
            h_try = _polish_exact(p_perturbed, bounds, maxiter=300)
            v_try = value(h_try)
            if v_try < best_val:
                best, best_val = h_try.copy(), v_try
                p_best = np.clip(best[:HALF], 0.0, 1.0).copy()

        # One more high-precision polish from the final best
        if remaining() > 10.0:
            h_final = _polish_exact(p_best, bounds, maxiter=1000)
            v_final = value(h_final)
            if v_final < best_val:
                best, best_val = h_final.copy(), v_final

    return best
