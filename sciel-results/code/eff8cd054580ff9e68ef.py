"""Fresh design: low-rank Gaussian mixture profile + bounded differential evolution.

We represent the nonnegative step function as a smooth profile (a small sum of
Gaussians, optionally plus a small constant floor) sampled onto the 600 equal
intervals of [-1/4, 1/4].  The objective is the normalized peak of the
autoconvolution, peak(h*h) / ||h||_1, computed via the convolution theorem with
a zero-padded FFT.  A wall-clock guarded differential-evolution loop optimizes
the mixture parameters.
"""
import time
import numpy as np

N = 600
DEADLINE = 140.0  # safety margin below the 150 s kill limit


def _profile(params):
    """Build the nonnegative step-height array from mixture parameters.

    params layout: [c, mu_1..mu_K, sig_1..sig_K, amp_1..amp_K]
    """
    K = 3
    c = params[0]
    rest = params[1:]
    mus = rest[:K]
    sigs = rest[K:2 * K]
    amps = rest[2 * K:3 * K]

    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.25 / N)  # interval centers
    h = np.full(N, c, dtype=float)
    for i in range(K):
        h += amps[i] * np.exp(-0.5 * ((x - mus[i]) / sigs[i]) ** 2)
    return h


def _objective(params):
    """Normalized peak of the autoconvolution (lower is better)."""
    h = _profile(params)
    if h.max() <= 0.0:
        return 1e9
    m = h.sum()  # ||h||_1
    if m <= 0.0:
        return 1e9
    fftsize = 2 * N  # zero-pad to avoid circular wrap-around
    H = np.fft.rfft(h, fftsize)
    # peak of h*h = max_t |FFT(H)|^2 / fftsize ; normalize by mass
    peak = float(np.max(np.abs(H) ** 2) / fftsize)
    return peak / m


def _rand_in(lo, hi):
    return np.random.uniform(lo, hi)


def _init_population(popsize):
    pop = []
    for _ in range(popsize):
        p = np.empty(1 + 3 * 3)
        p[0] = _rand_in(0.0, 0.15)            # constant floor
        for i in range(3):
            p[1 + i] = _rand_in(-0.25, 0.25)  # mu
            p[1 + 3 + i] = _rand_in(0.02, 0.12)  # sigma
            p[1 + 6 + i] = _rand_in(0.3, 1.5)    # amplitude
        pop.append(p)
    return pop


def construct():
    t0 = time.time()
    np.random.seed(0)

    # Seed candidate: the constant step function (value ~ 1.0).
    best = np.full(1 + 9, 0.0)
    best_val = _objective(best)  # ~ 1.0

    popsize = 40
    F = 0.7
    CR = 0.9

    pop = _init_population(popsize)
    fitness = np.array([_objective(p) for p in pop])
    bi = int(np.argmin(fitness))
    if fitness[bi] < best_val:
        best_val = fitness[bi]
        best = pop[bi].copy()

    while time.time() < DEADLINE:
        # Differential evolution: DE/rand/1
        for i in range(popsize):
            if time.time() > DEADLINE:
                break
            a, b, cidx = np.random.choice(popsize, 3, replace=False)
            while a == b or a == cidx:
                cidx = np.random.randint(popsize)
            v = pop[a] + F * (pop[b] - pop[cidx])
            mask = np.random.rand(len(v)) < CR
            if not mask.any():
                mask[np.random.randint(len(v))] = True
            trial = np.where(mask, v, pop[i])
            ft = _objective(trial)
            if ft <= fitness[i]:
                pop[i] = trial
                fitness[i] = ft
                if ft < best_val:
                    best_val = ft
                    best = trial.copy()

    return _profile(best)


# ---- quick self-check (not used by evaluator) ----
if __name__ == "__main__":
    t = time.time()
    out = construct()
    print("time", time.time() - t, "n", len(out), "min", min(out), "max", max(out))
    h = np.array(out)
    fftsize = 2 * N
    H = np.fft.rfft(h, fftsize)
    peak = float(np.max(np.abs(H) ** 2) / fftsize)
    print("value", peak / h.sum())
