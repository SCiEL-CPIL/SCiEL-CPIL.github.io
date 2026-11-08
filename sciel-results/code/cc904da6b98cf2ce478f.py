"""
Combined model-and-optimize approach for the second autocorrelation inequality.

Key structural facts:
  * h >= 0, so g = conv(h,h) >= 0.
  * norm_1 = sum(g)/(m+1) = sum(h)^2/(m+1).
  * norm_inf = max(g).
  * The objective is scale-invariant: value depends only on shape of h.
  * We want to maximize l2 / (norm_1 * norm_inf).

Strategy:
  1. Smooth log-sum-exp surrogate for norm_inf to make the objective differentiable.
     This allows L-BFGS-B to converge to sharper optima than the kinked max.
  2. Multi-stage optimization:
     a. Diverse structured starts (Gaussian, two-peak, semicircle, etc.)
     b. Progressive-beta L-BFGS-B (20 -> 50 -> 100 -> 200) for global coverage
     c. Nelder-Mead polish on exact objective
     d. Final high-precision L-BFGS-B with high beta
  3. DCT-based representation for dimensionality reduction in some phases.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 149.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _smooth_value(h, beta=100.0):
    """
    Smooth surrogate for the value function using log-sum-exp for norm_inf.
    Returns negative value for minimization.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0

    # Smooth max via log-sum-exp
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))

    if norm_1 <= 0 or lse <= 0:
        return 0.0

    val = l2 / (norm_1 * lse)
    return -val


def _neg_value(h):
    """Negative exact value for minimization."""
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _dct_basis(n_modes):
    """Return DCT-II basis matrix of shape (N, n_modes)."""
    k = np.arange(N)
    n = np.arange(n_modes)
    basis = np.cos(np.pi * np.outer(2 * k + 1, n) / (2 * N))
    return basis


def _dct_to_h(coeffs, basis):
    """Reconstruct h from DCT coefficients, enforcing nonnegativity."""
    h = basis @ coeffs
    return np.maximum(0.0, h)


def _h_to_dct(h, basis):
    """Project h onto DCT basis."""
    return basis.T @ h


def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian centered
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Sharp peak in center
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 10)) ** 2))
    # Power law
    starts.append(np.power(np.maximum(0.01, 1 - np.abs(x - N / 2) / (N / 2)), 2.0))
    # Sparse spikes
    for frac in [0.2, 0.3, 0.4]:
        h = np.zeros(N)
        center = int(N * frac)
        width = max(1, int(N * 0.05))
        h[max(0, center - width):min(N, center + width)] = 1.0
        starts.append(h)
    # Random smooth profiles
    for _ in range(15):
        freqs = RNG.randint(1, 8, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))

    return starts


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)  # safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # --- Phase 0: Quick baselines ---
    track(np.ones(N))
    for s in _generate_starts():
        track(s)

    bounds = [(0, None)] * N
    betas = [20.0, 50.0, 100.0, 200.0]

    # --- Phase 1: Progressive-beta L-BFGS-B from diverse starts ---
    for s in _generate_starts():
        if time.time() > deadline - 30:
            break

        x0 = np.maximum(0.0, np.asarray(s, dtype=float))

        for beta in betas:
            if time.time() > deadline - 20:
                break

            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value(x, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                track(res.x)
                x0 = res.x  # chain: use previous result as next start
            except Exception:
                pass

    # --- Phase 2: DCT-based optimization for global structure ---
    coarse_modes_list = [3, 5, 8]

    for n_modes in coarse_modes_list:
        if time.time() > deadline - 25:
            break

        basis = _dct_basis(n_modes)

        starts = []
        starts.append(_h_to_dct(best_h, basis))
        s2 = np.zeros(n_modes)
        s2[0] = 1.0
        starts.append(s2)
        starts.append(RNG.normal(0.5, 0.3, n_modes))
        starts.append(RNG.normal(0.3, 0.5, n_modes))

        for start in starts:
            if time.time() > deadline - 15:
                break

            try:
                res = opt.minimize(
                    lambda c: _smooth_value(_dct_to_h(c, basis), beta=100.0),
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_dct = _dct_to_h(res.x, basis)
                track(h_dct)
            except Exception:
                pass

    # --- Phase 3: Nelder-Mead polish on exact objective ---
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        scale = 0.02 + 0.03 * RNG.rand()
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                _neg_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            track(res.x)
        except Exception:
            pass

    # --- Phase 4: Final high-precision L-BFGS-B with high beta ---
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        pert = best_h + RNG.normal(0.0, 0.05, N)
        pert = np.maximum(0.0, pert)

        try:
            res = opt.minimize(
                lambda x: _smooth_value(x, beta=300.0),
                pert,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            track(res.x)
        except Exception:
            pass

    # --- Phase 5: Final Nelder-Mead on best ---
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _neg_value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
