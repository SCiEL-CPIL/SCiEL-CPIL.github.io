"""Signed autocorrelation C3-prime: minimize max|autoconv| / (sum)^2.
Structural change: discrete hypercube search via max-pooling stochastic gradient
steps (flip single signs to reduce the dominant lag), combined with spectral
initialization and FFT-accelerated coordinate descent for multi-flip moves.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 145.0


def _check_deadline(margin=3.0):
    return (time.time() - START_TIME) > (TIME_LIMIT - margin)


def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C


def _score(x):
    """C3' = max|autoconv| / (sum)^2."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv_fft(x)
    return float(np.max(np.abs(c)) / (s * s))


def _soft_max_and_grad(x, beta):
    """
    Compute soft-max surrogate for max|ac| and its gradient via FFT cross-correlation.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)

    m = np.max(abs_ac)
    exps = np.exp(beta * (abs_ac - m))
    log_sum = np.log(np.sum(exps))
    soft_val = m + log_sum / beta

    weights = exps / np.sum(exps)
    signs = np.sign(ac)
    signs[ac == 0] = 1.0
    grad_ac = weights * signs  # length 2N-1

    # grad_x[n] = sum_k grad_ac[k] * 2 * x[k-n] for valid k-n
    L_corr = 1
    while L_corr < 4 * N:
        L_corr *= 2

    g_padded = np.zeros(L_corr)
    g_padded[:2 * N - 1] = grad_ac

    X_f = np.fft.rfft(x, n=L_corr)
    G_f = np.fft.rfft(g_padded, n=L_corr)

    corr = np.fft.irfft(G_f * np.conj(X_f), n=L_corr)
    grad_x = 2.0 * corr[:N]

    return soft_val, grad_x


def _objective_and_grad(x, beta):
    """Compute objective and gradient."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10, np.zeros(N)
    soft_val, grad_soft = _soft_max_and_grad(x, beta)
    val = soft_val / (s * s)
    grad = grad_soft / (s * s) - 2.0 * soft_val * np.ones(N) / (s * s * s)
    return val, grad


def _spectral_init(rng, mode=0):
    """Generate initial x via spectral phase retrieval."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    n_freq = L_fft // 2 + 1

    if mode == 0:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.ones(n_freq)
        taper = np.minimum(1.0, np.abs(freqs) * L_fft / 5)
        mag *= taper
    elif mode == 1:
        freqs = np.fft.rfftfreq(L_fft)
        mag = 0.5 + 0.5 * np.cos(2 * np.pi * freqs * L_fft * 0.5)
    elif mode == 2:
        mag = 1.0 + 0.3 * rng.normal(0, 1, size=n_freq)
        mag = np.maximum(0.1, mag)
    elif mode == 3:
        freqs = np.fft.rfftfreq(L_fft)
        mag = np.exp(-20 * (freqs * L_fft / 2) ** 2)
    elif mode == 4:
        freqs = np.fft.rfftfreq(L_fft)
        cutoff = 0.4
        transition = 0.05
        mag = np.where(np.abs(freqs) < cutoff - transition, 1.0,
                       np.where(np.abs(freqs) < cutoff + transition,
                                0.5 * (1 + np.cos(np.pi * (np.abs(freqs) - cutoff) / transition)),
                                0.1))
    else:
        mag = np.ones(n_freq)

    phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    phase[0] = 0

    spectrum = mag * np.exp(1j * phase)
    spectrum[0] = mag[0]

    x_full = np.fft.irfft(spectrum, n=L_fft)
    x = x_full[:N]

    s = np.sum(x)
    if s < 0:
        x = -x

    return x


def _snap_to_signs(x):
    """Snap continuous values to +/-1, preserving sign."""
    return np.sign(x).astype(np.float64)


def _max_pooling_stochastic_flip(x, deadline, max_steps=50000):
    """
    Discrete local search: at each step, find the single sign flip that most
    reduces max|autoconv|, flip it, and repeat. Uses gradient guidance to
    prioritize candidate positions.
    """
    x = x.copy().astype(np.float64)
    best = _score(x)
    n = len(x)
    rng = np.random.default_rng(12345)

    for step in range(max_steps):
        if time.time() > deadline:
            break

        # Compute current autoconvolution and identify dominant lag
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_idx = int(np.argmax(abs_ac))
        max_val = abs_ac[max_idx]

        # Compute gradient to guide which positions to try
        if max_val < 1e-10:
            break

        # Use soft-max gradient to rank positions
        beta = 50.0
        _, grad = _soft_max_and_grad(x, beta)

        # Candidate positions: top-k by |gradient| plus random
        k = 20
        top_k = np.argsort(np.abs(grad))[-k:]
        rand_k = rng.integers(0, n, size=k)
        candidates = np.unique(np.concatenate([top_k, rand_k]))

        best_flip = None
        best_flip_score = best

        for j in candidates:
            x[j] = -x[j]
            sc = _score(x)
            if sc < best_flip_score:
                best_flip_score = sc
                best_flip = j
            x[j] = -x[j]

        if best_flip is not None:
            x[best_flip] = -x[best_flip]
            best = best_flip_score
        else:
            # No single flip helps; try a small random perturbation set
            for _ in range(5):
                j = int(rng.integers(n))
                x[j] = -x[j]
                sc = _score(x)
                if sc < best:
                    best = sc
                else:
                    x[j] = -x[j]

    return x, best


def _fft_coordinate_descent(x, deadline, max_rounds=200):
    """
    FFT-accelerated coordinate descent: try flipping groups of consecutive
    positions (blocks of size 1-8) to reduce the objective.
    """
    x = x.copy().astype(np.float64)
    best = _score(x)
    n = len(x)
    rng = np.random.default_rng(777)

    block_sizes = [1, 2, 4, 8]

    for round_num in range(max_rounds):
        if time.time() > deadline:
            break

        # Pick a random block size
        bs = block_sizes[int(rng.integers(len(block_sizes)))]
        nb = n // bs

        for b in range(nb):
            if time.time() > deadline:
                break

            start = b * bs
            end = start + bs

            # Try flipping the entire block
            x[start:end] *= -1
            sc = _score(x)
            if sc < best:
                best = sc
            else:
                x[start:end] *= -1

            # Try flipping half the block
            if bs >= 2:
                mid = start + bs // 2
                x[start:mid] *= -1
                sc = _score(x)
                if sc < best:
                    best = sc
                else:
                    x[start:mid] *= -1

                x[mid:end] *= -1
                sc = _score(x)
                if sc < best:
                    best = sc
                else:
                    x[mid:end] *= -1

    return x, best


def _continuous_optimize(x0, beta, deadline, maxiter=100):
    """Run L-BFGS-B from x0."""
    if time.time() > deadline:
        return x0, _score(x0)

    def obj(x):
        v, _ = _objective_and_grad(x, beta)
        return v

    def grad(x):
        _, g = _objective_and_grad(x, beta)
        return g

    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': 300,
                                'ftol': 1e-10, 'gtol': 1e-8,
                                'iprint': -1})
        x_opt = res.x
    except Exception:
        x_opt = x0

    return x_opt, _score(x_opt)


def _structured_seeds(rng):
    """Generate diverse initial seeds."""
    seeds = []
    seeds.append(np.ones(N, dtype=np.float64))
    seeds.append(np.tile([1.0, -1.0], N // 2).astype(np.float64))
    seeds.append(np.tile([1.0, 1.0, -1.0, -1.0], N // 4).astype(np.float64))
    seeds.append(np.tile([1.0, -1.0, -1.0, 1.0], N // 4).astype(np.float64))
    for _ in range(6):
        seeds.append(rng.choice([-1.0, 1.0], size=N).astype(np.float64))
    for bs in [4, 8, 16, 32]:
        if N % bs == 0:
            nb = N // bs
            seeds.append(np.repeat(rng.choice([-1.0, 1.0], size=nb), bs).astype(np.float64))
    return seeds


def construct():
    """Return the sequence of signed step heights minimizing C3'."""
    deadline = START_TIME + TIME_LIMIT - 5.0
    rng = np.random.default_rng(42)

    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # Phase 1: Generate diverse seeds and run continuous optimization + snap
    seeds = _structured_seeds(rng)
    # Add spectral seeds
    for mode in range(5):
        seeds.append(_spectral_init(rng, mode))

    betas = [30.0, 80.0]

    for s in seeds:
        if time.time() > deadline - 20:
            break

        for beta in betas:
            if time.time() > deadline - 15:
                break

            x_cont, sc_cont = _continuous_optimize(s, beta, deadline - 10, maxiter=80)
            x_s = _snap_to_signs(x_cont)
            sc_s = _score(x_s)

            if sc_s < best_score:
                best_score = sc_s
                best_x = x_s.copy()

    # Phase 2: Max-pooling stochastic flip search (core structural change)
    if time.time() < deadline - 10:
        x_mp, sc_mp = _max_pooling_stochastic_flip(best_x, deadline - 10, max_steps=80000)
        if sc_mp < best_score:
            best_score = sc_mp
            best_x = x_mp.copy()

    # Phase 3: FFT coordinate descent for multi-flip moves
    if time.time() < deadline - 5:
        x_cd, sc_cd = _fft_coordinate_descent(best_x, deadline - 5, max_rounds=100)
        if sc_cd < best_score:
            best_score = sc_cd
            best_x = x_cd.copy()

    # Phase 4: Final max-pooling polish
    if time.time() < deadline - 2:
        x_final, sc_final = _max_pooling_stochastic_flip(best_x, deadline - 2, max_steps=50000)
        if sc_final < best_score:
            best_x = x_final
            best_score = sc_final

    return best_x
