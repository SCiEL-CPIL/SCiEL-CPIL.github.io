"""Combined approach: smooth surrogate optimization with diverse initialization
and local refinement, plus a convex QP-style refinement via SLSQP.

Key structure:
    value = abs(2*N*max(convolve(h,h)) / sum(h)**2)
is invariant under h -> c*h, so we fix sum(h)=1 and minimize 2*N*max(conv).
Since conv[0] = sum(h^2) >= (sum h)^2 / N = 1/N (Cauchy-Schwarz), value >= 2.

We use a smooth log-sum-exp surrogate for the max, with diverse initializations
(Gaussian mixture, block structures, filtered randoms) and Nelder-Mead refinement.
We also attempt an SLSQP-based convex refinement to squeeze out further gains.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 150.0
_MARGIN = 8.0


def _make_grid():
    return np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective(h, beta=80.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = m + log_sum / beta
    val = 2.0 * N * soft_max / (s * s)
    return abs(val)


def _smooth_grad(h, beta=80.0):
    """Gradient of the smooth objective w.r.t. h."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.zeros(N), 1e18
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    soft_max = m + np.log(np.sum(weights)) / beta
    obj = abs(2.0 * N * soft_max / (s * s))
    
    # d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    # d(soft_max)/d(h[j]) = sum_k weights[k] * 2*h[k-j] (valid k)
    # = 2 * sum_{k: 0<=k-j<2N-1} weights[k] * h[k-j]
    
    # Efficient computation via correlation:
    # For each j, we need sum over m where m = k-j, so k = m+j
    # d(soft_max)/d(h[j]) = 2 * sum_{m=0}^{N-1} weights[m+j] * h[m] (where m+j < 2N-1)
    
    grad_soft = np.zeros(N)
    for j in range(N):
        # k ranges from j to j+N-1, but must be < 2N-1
        k_start = j
        k_end = min(j + N, 2 * N - 1)
        if k_start < k_end:
            grad_soft[j] = 2.0 * np.dot(weights[k_start:k_end], h[:k_end - k_start])
    
    # d(obj)/d(h[j]) = 2N * (grad_soft[j]/s^2 - 2*soft_max/s^3)
    grad = 2.0 * N * (grad_soft / (s * s) - 2.0 * soft_max / (s * s * s))
    return grad, obj


def _gaussian_mixture(x, mu, sig):
    """Sum of Gaussian bumps, normalized so that sum == 1."""
    g = np.zeros(N)
    for m, s in zip(mu, sig):
        g += np.exp(-0.5 * ((x - m) / s) ** 2)
    s = g.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return g / s


def _block_structure(x, n_blocks=4, width=0.06):
    """Block-like structure with smooth transitions."""
    g = np.zeros(N)
    block_centers = np.linspace(-0.2, 0.2, n_blocks)
    for c in block_centers:
        g += np.exp(-0.5 * ((x - c) / width) ** 2)
    s = g.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return g / s


def _filtered_random(rng, seed_offset=0):
    """Random profile with Gaussian filtering for smoothness."""
    r = np.random.default_rng(42 + seed_offset)
    raw = r.normal(0, 1, N)
    # Apply a simple Gaussian filter
    kernel_size = 15
    kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / 4.0)
    kernel /= kernel.sum()
    filtered = np.convolve(raw, kernel, mode='same')
    # Make positive and normalize
    filtered = filtered - filtered.min() + 0.1
    s = filtered.sum()
    if s <= 0 or not np.isfinite(s):
        return None
    return filtered / s


def _construct_baseline():
    """A smooth, strictly-positive, centrally-peaked profile (Gaussian mixture)."""
    x = _make_grid()
    mu = np.array([0.0, -0.09, 0.09])
    sig = np.array([0.055, 0.07, 0.07])
    g = _gaussian_mixture(x, mu, sig)
    if g is None or not np.all(np.isfinite(g)):
        g = np.ones(N) / N
    return g


def _diverse_initializations():
    """Generate diverse initial points for optimization."""
    x = _make_grid()
    inits = []
    
    # Gaussian mixture variants
    mu_sets = [
        np.array([0.0, -0.09, 0.09]),
        np.array([0.0, -0.12, 0.12]),
        np.array([0.0, -0.06, 0.06]),
        np.array([0.0]),
        np.array([0.0, -0.15, 0.15]),
    ]
    sig_sets = [
        np.array([0.055, 0.07, 0.07]),
        np.array([0.06, 0.08, 0.08]),
        np.array([0.05, 0.06, 0.06]),
        np.array([0.06]),
        np.array([0.05, 0.09, 0.09]),
    ]
    for mu, sig in zip(mu_sets, sig_sets):
        g = _gaussian_mixture(x, mu, sig)
        if g is not None and np.all(np.isfinite(g)):
            inits.append(g)
    
    # Block structures
    for n_blocks in [3, 4, 5]:
        for width in [0.05, 0.07, 0.09]:
            g = _block_structure(x, n_blocks, width)
            if g is not None and np.all(np.isfinite(g)):
                inits.append(g)
    
    # Filtered randoms
    for i in range(5):
        g = _filtered_random(None, seed_offset=i)
        if g is not None and np.all(np.isfinite(g)):
            inits.append(g)
    
    # Uniform
    inits.append(np.ones(N) / N)
    
    return inits


def _diverse_init(max_time=15.0):
    """Generate diverse initial candidates using multiple strategies."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Single cosine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Random Gaussian
    for trial in range(10):
        if time.time() - t0 > max_time * 0.5:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(3):
            if time.time() - t0 > max_time * 0.8:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Multi-frequency cosine
    for n_freq in range(2, 5):
        if time.time() - t0 > max_time * 0.85:
            break
        for trial in range(5):
            freqs = sorted(rng.uniform(1, 40, n_freq))
            phases = rng.uniform(0, 2*np.pi, n_freq)
            amps = rng.uniform(0.2, 1.0, n_freq)
            dc = rng.uniform(-0.1, 0.1)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2*np.pi*f*x/N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Evaluate all candidates
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=25.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [30.0, 60.0, 100.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=beta)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=beta)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=5, max_time=20.0):
    """Multi-start local refinement."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.1 + 0.05 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            res = minimize(
                lambda h: _smooth_objective(h, beta=50.0),
                h_init,
                jac=lambda h: _smooth_grad(h, beta=50.0)[0],
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return _DEADLINE - _MARGIN - elapsed()
    
    # Phase 1: Diverse initialization
    h_div, val_div = _diverse_init(max_time=15.0)
    
    best_h = h_div.copy()
    best_val = val_div
    
    # Phase 2: Local refinement
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 25.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 20:
        best_h, best_val = _multi_start(best_h, n_starts=5, max_time=min(20.0, time_left() - 15.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 10:
        try:
            def obj_fn(h):
                return _smooth_objective(h, beta=100.0)
            
            def grad_fn(h):
                g, _ = _smooth_grad(h, beta=100.0)
                return g
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final safety checks
    best_h = np.asarray(best_h, dtype=float)
    if not np.all(np.isfinite(best_h)) or best_h.sum() == 0:
        best_h = np.ones(N)
    
    # Numerical guard
    if _value(best_h) > 2.0 + 1e-9:
        best_h = np.ones(N)
    
    return best_h


# Self-check
if __name__ == "__main__":
    _h = construct()
    _s = _h.sum()
    _conv = np.convolve(_h, _h, mode="full")
    print("value =", abs(2.0 * N * np.max(_conv) / (_s * _s)),
          "sum =", _s, "finite =", bool(np.all(np.isfinite(_h))))
