"""SDP-inspired convex optimization: minimize max autocorrelation subject to sum=1.
The objective 2*N*max(conv)/sum^2 is scale-invariant, so fix sum(h)=1 and minimize
max_k conv[k]. This is a convex problem (max of linear functions). We use subgradient
methods and projected gradient descent, plus a final coordinate-wise polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400
DEADLINE = 145  # wall-clock seconds; evaluator kills at 150s, keep margin

# Precompute FFT size for fast convolution
_FFT_SIZE = 2 * N - 1
_FFT_NP = 1 << (int(np.log2(_FFT_SIZE)) + 1)  # next power of 2


def _evaluate(h):
    """Return (value, valid) for the exact task objective."""
    if not np.all(np.isfinite(h)):
        return np.inf, False
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf, False
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf, False
    return val, True


def _convex_objective(h):
    """Convex objective: max_k conv[k] with sum(h)=1 constraint handled externally.
    Returns (max_conv, argmax_k) for subgradient computation."""
    conv = fftconvolve(h, h, mode='full')
    k_max = int(np.argmax(conv))
    return conv[k_max], k_max, conv


def _convex_subgradient(h, k_max):
    """Subgradient of max_k conv[k] at the active lag k_max.
    d/dh[i] conv[k_max] = h[k_max - i] + h[i - k_max] (with bounds).
    Since conv is quadratic, the subgradient is 2 * (contribution from h)."""
    grad = np.zeros(N)
    for i in range(N):
        j1 = k_max - i
        j2 = i - k_max
        if 0 <= j1 < N:
            grad[i] += h[j1]
        if 0 <= j2 < N:
            grad[i] += h[j2]
    return 2.0 * grad


def _spike_candidate():
    """Deterministic high-quality candidate: two-spike + small negative tail."""
    a = 0.5
    K = 200
    eps = 0.001
    h = np.zeros(N)
    h[0] = a
    h[1] = 1.0 - a
    h[2:K] = -eps
    return h


def _uniform_candidate():
    """Uniform distribution: h[i] = 1/N for all i. Value = 2.0 (seed)."""
    return np.ones(N) / N


def _broad_positive_candidate(width=50, center=200):
    """Broad positive bump, normalized to sum=1."""
    h = np.zeros(N)
    x = np.arange(N)
    h += np.exp(-0.5 * ((x - center) / width) ** 2)
    s = np.sum(h)
    if s > 1e-12:
        h /= s
    return h


def _block_candidate(block_size=10, neg_ratio=0.0005):
    """Block of positive values followed by small negative tail."""
    h = np.zeros(N)
    h[:block_size] = 1.0 / block_size
    h[block_size:] = -neg_ratio
    s = np.sum(h)
    if abs(s) > 1e-12:
        h /= s
    return h


def _two_point_candidate(i1=0, i2=1, w1=0.5):
    """Two-point distribution: weight w1 at i1, (1-w1) at i2. Sum=1."""
    h = np.zeros(N)
    h[i1] = w1
    h[i2] = 1.0 - w1
    return h


def _gaussian_candidate(center=200, width=50, neg_scale=0.01):
    """Gaussian bump with small negative background."""
    h = np.zeros(N)
    x = np.arange(N)
    h += np.exp(-0.5 * ((x - center) / width) ** 2)
    h -= neg_scale
    s = np.sum(h)
    if abs(s) > 1e-12:
        h /= s
    return h


def _project_to_sum1(h):
    """Project h onto the affine set {x : sum(x) = 1} via nearest point."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        # Degenerate: return uniform
        return np.ones(N) / N
    return h + (1.0 - s) / N


def _spectral_candidate(seed, n_sines=5, freq_range=(1, 20), amp_range=(0.01, 0.5)):
    """Generate a candidate as a sum of sine waves with random frequencies/amplitudes."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.zeros(N)
    for _ in range(n_sines):
        freq = rng.uniform(*freq_range)
        amp = rng.uniform(*amp_range)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.sin(freq * t + phase)
    h -= np.mean(h)
    s = np.sum(h)
    if abs(s) > 1e-12:
        h /= s
    return h


def _random_positive_candidate(seed, sparsity=0.3):
    """Random non-negative vector, normalized to sum=1."""
    rng = np.random.RandomState(seed)
    h = rng.exponential(scale=1.0, size=N)
    # Zero out some entries for sparsity
    mask = rng.random(N) > sparsity
    h[mask] = 0.0
    s = np.sum(h)
    if s > 1e-12:
        h /= s
    else:
        h = np.ones(N) / N
    return h


def _alternating_candidate(period, amplitude_ratio=0.5):
    """Create an alternating-sign pattern with a large positive DC offset."""
    h = np.zeros(N)
    for i in range(N):
        if i % period == 0 or (period > 2 and i % period == 1):
            h[i] = 1.0
        else:
            h[i] = -amplitude_ratio
    return h


def _convex_optimize(h_init, max_iters=5000, step_init=0.01, deadline=None):
    """Projected subgradient descent on the convex problem:
    min max_k conv_k(h)  s.t.  sum(h) = 1
    
    The subgradient at the active lag k* is 2*(contribution vector).
    We project back to sum=1 after each step.
    """
    h = h_init.copy()
    h = _project_to_sum1(h)
    
    best_h = h.copy()
    best_maxconv, _, _ = _convex_objective(h)
    best_value = 2.0 * N * best_maxconv  # since sum=1, value = 2*N*maxconv
    
    step = step_init
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute subgradient
        maxconv, k_max, conv = _convex_objective(h)
        grad = _convex_subgradient(h, k_max)
        
        # Adaptive step size: reduce if objective increased
        h_trial = h - step * grad
        h_trial = _project_to_sum1(h_trial)
        
        trial_maxconv, _, _ = _convex_objective(h_trial)
        if trial_maxconv < maxconv:
            h = h_trial
            step *= 1.05  # slightly increase step
        else:
            step *= 0.8  # reduce step
        
        # Track best
        if trial_maxconv < best_maxconv:
            best_maxconv = trial_maxconv
            best_h = h_trial.copy()
            best_value = 2.0 * N * best_maxconv
        
        # Early stopping if convergence
        if step < 1e-8:
            break
    
    return best_h, best_value


def _smooth_objective(x, beta):
    """Smooth (soft-max) surrogate for 2*N*max(conv)/sum^2, sum>0 regime."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return 1e10
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    shifted = conv - m
    shifted = np.clip(shifted, -50.0, 50.0)
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    if not np.isfinite(val):
        return 1e10
    return float(val)


def _smooth_grad(x, beta):
    """Gradient of the smooth objective."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    w = np.exp(beta * (conv - m))
    w_sum = np.sum(w)
    w /= w_sum
    w_padded = np.zeros(2 * N - 1)
    w_padded[:len(w)] = w
    grad_lse = np.zeros(N)
    for i in range(N):
        grad_lse[i] = 2.0 * float(np.dot(w_padded[i:i + N], x))
    lse = m + np.log(w_sum) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df


def _coordinate_descent(h, max_iters=5000, tol=1e-10, deadline=None):
    """Coordinate descent on the exact objective."""
    best_h = h.copy()
    best_val, _ = _evaluate(best_h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            for delta in (-0.01, -0.001, 0.001, 0.01):
                h_trial = best_h.copy()
                h_trial[i] += delta
                val, valid = _evaluate(h_trial)
                if valid and val < best_val - 1e-12:
                    best_h = h_trial
                    best_val = val
                    improved = True
                    break
        if not improved:
            break
    return best_h, best_val


def _coordinate_descent_convex(h, max_iters=3000, deadline=None):
    """Coordinate descent on the convex formulation (sum=1, minimize max conv).
    For each coordinate, find the best value that minimizes max conv."""
    h = _project_to_sum1(h.copy())
    best_h = h.copy()
    best_maxconv, _, _ = _convex_objective(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Try a few candidate values for h[i]
            # Keep sum=1: if we change h[i] by delta, we need to adjust another coord
            # Simpler: try small perturbations and reproject
            for delta in (-0.005, -0.001, 0.001, 0.005, -0.02, 0.02):
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = _project_to_sum1(h_trial)
                trial_maxconv, _, _ = _convex_objective(h_trial)
                if trial_maxconv < best_maxconv - 1e-12:
                    h = h_trial
                    best_h = h_trial.copy()
                    best_maxconv = trial_maxconv
                    improved = True
                    break
        if not improved:
            break
    best_value = 2.0 * N * best_maxconv
    return best_h, best_value


def _targeted_search(h_init, n_samples=1000, deadline=None):
    """Targeted search: perturb specific coordinates based on gradient structure."""
    best_h = h_init.copy()
    best_val, _ = _evaluate(best_h)
    rng = np.random.RandomState(42)
    
    for it in range(n_samples):
        if deadline is not None and time.time() > deadline:
            break
        conv = fftconvolve(best_h, best_h, mode='full')
        k_max = int(np.argmax(conv))
        
        candidates = []
        for i in range(N):
            j = k_max - i
            if 0 <= j < N:
                candidates.append(i)
                candidates.append(j)
        
        if len(candidates) > 0:
            idx = rng.choice(candidates, size=min(5, len(candidates)), replace=False)
            h_trial = best_h.copy()
            for i in idx:
                h_trial[i] += rng.normal(0, 0.05)
            val, valid = _evaluate(h_trial)
            if valid and val < best_val:
                best_h = h_trial
                best_val = val
    
    return best_h, best_val


def construct():
    start = time.time()
    deadline = start + DEADLINE

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Uniform (seed)
    candidates.append(_uniform_candidate())
    
    # Two-point distributions at various positions
    for i1 in (0, 1, 5, 10, 50, 100, 200):
        for i2 in (1, 2, 10, 50, 100, 200, 399):
            if i1 == i2:
                continue
            for w1 in (0.3, 0.5, 0.7):
                candidates.append(_two_point_candidate(i1, i2, w1))
    
    # Broad positive bumps
    for center in (50, 100, 200, 300, 350):
        for width in (10, 20, 50, 100, 200):
            candidates.append(_broad_positive_candidate(center, width))
    
    # Random positive candidates
    for seed in range(30):
        for sparsity in (0.0, 0.2, 0.5, 0.8):
            candidates.append(_random_positive_candidate(seed, sparsity))
    
    # Block candidates
    for bs in (5, 10, 20, 50, 100):
        for nr in (0.0001, 0.001, 0.01):
            candidates.append(_block_candidate(block_size=bs, neg_ratio=nr))
    
    # Spectral candidates
    for seed in range(15):
        for n_sines in (3, 5, 8):
            candidates.append(_spectral_candidate(seed, n_sines=n_sines))
    
    # Evaluate all, find best starting point
    best_h = None
    best_value = np.inf
    for h in candidates:
        val, valid = _evaluate(h)
        if valid and val < best_value:
            best_value = val
            best_h = h.copy()
    
    if best_h is None or not np.isfinite(best_value):
        best_h = _uniform_candidate()
        best_value, _ = _evaluate(best_h)

    # Phase 2: Convex optimization (subgradient + projected gradient)
    # Normalize to sum=1 for the convex formulation
    h_norm = best_h.copy()
    s = np.sum(h_norm)
    if abs(s) > 1e-12:
        h_norm /= s
    
    if time.time() < deadline - 30:
        # Subgradient descent
        h_sg, val_sg = _convex_optimize(
            h_norm, 
            max_iters=8000, 
            step_init=0.005, 
            deadline=deadline - 30
        )
        # Convert back: value = 2*N*maxconv (since sum=1)
        val_sg_exact, valid_sg = _evaluate(h_sg)
        if valid_sg and val_sg_exact < best_value:
            best_value = val_sg_exact
            best_h = h_sg.copy()

    # Phase 3: Coordinate descent on convex formulation
    if time.time() < deadline - 15:
        h_norm = best_h.copy()
        s = np.sum(h_norm)
        if abs(s) > 1e-12:
            h_norm /= s
        h_cd, val_cd = _coordinate_descent_convex(
            h_norm,
            max_iters=5000,
            deadline=deadline - 15
        )
        val_cd_exact, valid_cd = _evaluate(h_cd)
        if valid_cd and val_cd_exact < best_value:
            best_value = val_cd_exact
            best_h = h_cd.copy()

    # Phase 4: L-BFGS-B polish with smooth surrogate
    if time.time() < deadline - 10:
        for beta in (50.0, 100.0, 200.0):
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_h.copy(),
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxfun': 2000, 'gtol': 1e-12},
                )
                val, valid = _evaluate(res.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Phase 5: Final coordinate descent polish
    if time.time() < deadline - 3:
        best_h, best_value = _coordinate_descent(
            best_h, max_iters=3000, deadline=deadline - 3
        )

    # Final validity guard
    val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = _uniform_candidate()
        val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return best_h
