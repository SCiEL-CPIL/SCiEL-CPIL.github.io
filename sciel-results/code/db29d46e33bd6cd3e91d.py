"""
Strategy: Continuous-parameter optimization of smooth window shapes via
gradient-free Nelder-Mead, plus a fast local coordinate refinement.

Rationale:
1. The seed (50 equal heights) is flat, but the observed best solutions
   (score ~1.21-1.30) are smooth bump-shaped profiles that reduce the
   autocorrelation peak while maintaining a reasonable quadratic form.
2. A flat-topped rectangle has an autocorrelation that is triangular,
   giving a specific ratio. Smooth windows (Gaussian, raised-cosine)
   have more concentrated autocorrelation, changing the l2 / (norm_1 * norm_inf)
   ratio.
3. Instead of a large population GA (which has overhead), we run multiple
   independent Nelder-Mead searches from diverse initial shapes, each
   operating on a low-dimensional parameterization (center, width, skew,
   amplitude, plus a small number of harmonic perturbations).
4. After the global search, we do a fast coordinate-descent refinement
   on the raw 50-dim vector to squeeze out extra value.

The key insight from the task structure:
- g = convolve(h, h) is the autocorrelation.
- l2 is a quadratic form of g (with the y padding).
- norm_1 is the mean of |g| (since g >= 0 for h >= 0, it's just mean(g)).
- norm_inf is max(g).
- For h >= 0, g >= 0 always (autocorrelation of non-negative signal).
- So norm_1 = sum(g) / (2N) = (sum(h)^2) / (2N) ... wait, let me verify.
  Actually sum(g) = sum_k g[k] = sum_k sum_i h[i]h[k-i] = (sum h)^2.
  So norm_1 = (sum(h))^2 / (2N).
- norm_inf = max(g) = g[0] (since g[0] = sum h[i]^2, and for autocorrelation
  of non-negative h, g[0] is the maximum because g[k] = sum_i h[i]h[i+k]
  and by Cauchy-Schwarz or just the fact that shifting reduces overlap...
  actually for non-negative h, g[0] >= g[k] for all k, so norm_inf = g[0]).

So value = l2 / (norm_1 * norm_inf) where:
  norm_1 = (sum(h))^2 / (2*N)
  norm_inf = g[0] = sum(h[i]^2)

And l2 can be expressed in terms of g. Let me re-derive l2:
  y = [0, g[0], g[1], ..., g[m-1], 0] where m = 2N-1
  w = 1/(m+1) = 1/(2N)
  l2 = (w/3) * sum_{i=0}^{m} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)

Let me compute this explicitly:
  y[0] = 0, y[1] = g[0], y[2] = g[1], ..., y[m] = g[m-1], y[m+1] = 0
  where m = 2N-1, so y has m+2 = 2N+1 elements.

  sum_{i=0}^{m} (y[i]^2 + y[i]y[i+1] + y[i+1]^2):
  i=0: 0 + 0 + g[0]^2 = g[0]^2
  i=1: g[0]^2 + g[0]*g[1] + g[1]^2
  i=2: g[1]^2 + g[1]*g[2] + g[2]^2
  ...
  i=m-1: g[m-2]^2 + g[m-2]*g[m-1] + g[m-1]^2
  i=m: g[m-1]^2 + 0 + 0 = g[m-1]^2

  Total = 2*g[0]^2 + 2*g[m-1]^2 + sum_{k=0}^{m-2} (g[k]^2 + g[k]g[k+1] + g[k+1]^2)
        = 2*g[0]^2 + 2*g[m-1]^2 + sum_{k=0}^{m-2} g[k]^2 + sum_{k=0}^{m-2} g[k]g[k+1] + sum_{k=0}^{m-2} g[k+1]^2
        = 2*g[0]^2 + 2*g[m-1]^2 + sum_{k=0}^{m-2} g[k]^2 + sum_{k=1}^{m-1} g[k]^2 + sum_{k=0}^{m-2} g[k]g[k+1]
        = 2*g[0]^2 + 2*g[m-1]^2 + g[0]^2 + sum_{k=1}^{m-1} g[k]^2 + sum_{k=1}^{m-1} g[k]^2 + g[m-1]^2 + sum_{k=0}^{m-2} g[k]g[k+1]
        = 3*g[0]^2 + 3*g[m-1]^2 + 2*sum_{k=1}^{m-1} g[k]^2 + sum_{k=0}^{m-2} g[k]g[k+1]

Hmm, this is getting complex. Let me just compute it numerically in the evaluation.

For the optimization, I'll parametrize h as a smooth function and use
scipy.optimize.minimize with Nelder-Mead on the parameters.
"""

import numpy as np
import time
import random

N = 50


def compute_value(h):
    """Compute the objective value for a given height vector h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h, mode='full')
    m = len(g)  # 2N - 1 = 99
    w = 1.0 / (m + 1)  # 1/(2N)
    
    # Build y
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    
    y_i = y[:m + 1]
    y_ip1 = y[1:m + 2]
    terms = y_i ** 2 + y_i * y_ip1 + y_ip1 ** 2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf < 1e-15:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _make_h_from_params(params):
    """
    Generate a 50-dim height vector from continuous parameters.
    
    params: [center, width, amplitude, skew, n_harmonics, harm_amplitude, ...]
    We use a base shape (Gaussian or raised cosine) plus optional harmonic perturbations.
    """
    center = params[0]  # 0 to 1
    width = params[1]   # 0.1 to 2.0
    amplitude = params[2]  # 0.1 to 3.0
    skew = params[3]    # -1 to 1
    shape_type = params[4]  # 0=gaussian, 1=raised cosine, 2=triangle, 3=flat-top smooth
    n_harmonics = int(params[5]) if len(params) > 5 else 0
    
    x = np.linspace(0, 1, N)
    xc = (x - center) / (width * (1.0 + skew * (x - center)))
    
    if shape_type < 0.5:
        # Gaussian
        base = np.exp(-xc ** 2)
    elif shape_type < 1.5:
        # Raised cosine (Hann-like)
        base = 0.5 * (1 + np.cos(np.pi * xc))
        base = np.where(np.abs(xc) < 1, base, 0.0)
    elif shape_type < 2.5:
        # Triangle
        base = np.maximum(0, 1 - np.abs(xc))
    else:
        # Flat-top with smooth edges
        t = np.abs(xc)
        base = np.ones_like(t)
        mask = t > 0.8
        base[mask] = 0.5 * (1 + np.cos(np.pi * (t[mask] - 0.8) / 0.2))
        base[t > 1.0] = 0.0
    
    h = amplitude * base
    
    # Add harmonic perturbations if requested
    if n_harmonics > 0 and len(params) > 6 + n_harmonics:
        for k in range(n_harmonics):
            harm_amp = params[6 + k]
            freq = (k + 1) * 2 * np.pi
            h += harm_amp * np.sin(freq * x)
    
    h = np.clip(h, 0.0, None)
    return h


def _nelder_mead_search(params0, deadline, max_iter=2000):
    """Run Nelder-Mead from an initial parameter vector."""
    from scipy.optimize import minimize
    
    def objective(params):
        h = _make_h_from_params(params)
        return -compute_value(h)
    
    best_val = -objective(params0)
    best_params = params0.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x.copy()
        except Exception:
            break
    
    return best_params, best_val


def _local_refine(h, deadline):
    """Fast coordinate descent refinement on the raw 50-dim vector."""
    h = np.array(h, dtype=np.float64).copy()
    best_val = compute_value(h)
    best_h = h.copy()
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            for step in step_sizes:
                for sign in [1, -1]:
                    new_h = best_h.copy()
                    new_h[i] = max(0.0, best_h[i] + sign * step)
                    val = compute_value(new_h)
                    if val > best_val + 1e-12:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Try smaller steps
            step_sizes = [s * 0.5 for s in step_sizes]
            if step_sizes[0] < 1e-6:
                break
    
    return best_h, best_val


def construct():
    deadline = time.time() + 115.0
    rng = np.random.RandomState(42)
    
    # Diverse initial parameter sets for global search
    initial_params_list = []
    
    # Gaussian bumps at various centers and widths
    for center in [0.5, 0.3, 0.7]:
        for width in [0.3, 0.5, 0.8]:
            for amp in [1.0, 2.0]:
                initial_params_list.append([center, width, amp, 0.0, 0.0, 0])
    
    # Raised cosine shapes
    for center in [0.5]:
        for width in [0.4, 0.6, 0.8, 1.0]:
            for amp in [1.0, 1.5, 2.0]:
                initial_params_list.append([center, width, amp, 0.0, 1.0, 0])
    
    # Triangle shapes
    for width in [0.5, 0.8, 1.0]:
        initial_params_list.append([0.5, width, 1.0, 0.0, 2.0, 0])
    
    # Flat-top smooth
    for width in [0.6, 0.8, 1.0]:
        initial_params_list.append([0.5, width, 1.0, 0.0, 3.0, 0])
    
    # With some harmonics
    for center in [0.5]:
        for width in [0.5, 0.7]:
            initial_params_list.append([center, width, 1.5, 0.2, 0.0, 2, 0.05, -0.03])
            initial_params_list.append([center, width, 1.5, -0.2, 1.0, 2, 0.04, 0.06])
    
    # Add some random perturbations to break symmetry
    for i in range(0, len(initial_params_list), 3):
        initial_params_list[i][0] += rng.uniform(-0.05, 0.05)
        initial_params_list[i][1] *= (1 + rng.uniform(-0.1, 0.1))
    
    # Evaluate all initial points to find the best starting point
    best_initial_val = -np.inf
    best_initial_params = initial_params_list[0]
    for params in initial_params_list:
        h = _make_h_from_params(params)
        val = compute_value(h)
        if val > best_initial_val:
            best_initial_val = val
            best_initial_params = params.copy()
    
    # Run Nelder-Mead from top initial points
    # Sort by initial value, take top 10
    scored_inits = []
    for params in initial_params_list:
        h = _make_h_from_params(params)
        val = compute_value(h)
        scored_inits.append((val, params))
    scored_inits.sort(key=lambda x: -x[0])
    
    best_val = -np.inf
    best_h = None
    
    # Run from top 12 initial points, with time budget per run
    n_searches = min(12, len(scored_inits))
    time_per_search = 8.0  # seconds per Nelder-Mead run
    
    for i in range(n_searches):
        if time.time() > deadline - 30:  # Reserve time for local refinement
            break
        val, params = scored_inits[i]
        sub_deadline = min(deadline - 25, time.time() + time_per_search)
        
        try:
            _, search_val = _nelder_mead_search(params, sub_deadline, max_iter=500)
            if search_val > best_val:
                best_val = search_val
                best_h = _make_h_from_params(params)
        except Exception:
            continue
    
    # Also try a few more diverse random starts if time permits
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            center = rng.uniform(0.2, 0.8)
            width = rng.uniform(0.3, 1.2)
            amp = rng.uniform(0.5, 2.5)
            skew = rng.uniform(-0.5, 0.5)
            stype = rng.randint(0, 4)
            params = [center, width, amp, skew, float(stype), 0]
            
            h = _make_h_from_params(params)
            val = compute_value(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
            
            # Quick local search from here
            sub_deadline = min(deadline - 10, time.time() + 5.0)
            h_ref, val_ref = _local_refine(h, sub_deadline)
            if val_ref > best_val:
                best_val = val_ref
                best_h = h_ref
    
    # Final local refinement on the best candidate
    if best_h is not None and time.time() < deadline - 3:
        best_h, best_val = _local_refine(best_h, deadline - 1.0)
    
    # Ensure output is valid
    best_h = np.clip(np.asarray(best_h, dtype=np.float64), 0.0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return list(best_h)


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
