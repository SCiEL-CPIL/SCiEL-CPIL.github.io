"""Third autocorrelation inequality – signed step on 400 bins over [-1/4, 1/4].

Goal: minimise the autoconvolution upper bound on C3.
We represent the step function by its 400 signed heights (each in {-1, +1}).
The autoconvolution is computed via FFT.  We use a hybrid optimiser:

  1. Start from several structured initialisers (alternating blocks, sine signs,
     random with a few large blocks).
  2. Greedy coordinate descent: repeatedly flip the single coordinate that
     gives the largest decrease in the autoconvolution L1 (or L∞) proxy.
     Each flip cost is O(N log N) via FFT; with a wall-clock deadline we do as
     many as fit.
  3. Simulated-annealing polish: random flips accepted by Metropolis criterion,
     cooling geometrically, to escape shallow local minima.
  4. Return the best sign vector found.

The returned sequence is the list/array of 400 signed step heights.
The evaluator computes the raw autoconvolution upper bound; lower is better.
Score = seed / value, so 1.0 == seed; >1.0 is improvement.
"""
import numpy as np
import time

N = 400
DEADLINE = 120.0          # evaluator kills at 120 s
SAFETY = 5.0              # stop 5 s early

# ---------------------------------------------------------------------------
# Autoconvolution proxy
# ---------------------------------------------------------------------------
def _autoconv_energy(s):
    """Return a scalar proxy for the autoconvolution upper bound.

    For a step function f on [-1/4,1/4] with N equal bins, the autoconvolution
    (f * f)(t) = ∫ f(x) f(t-x) dx.  The relevant upper bound C3 involves the
    L∞ norm of the autoconvolution.  We use the L1 norm of the discrete
    autoconvolution as a cheap, differentiable proxy that correlates well
    with the true bound (minimising L1 tends to minimise L∞ as well).
    """
    # Pad to avoid circular convolution aliasing
    size = 1 << (2 * N - 1).bit_length()   # next power of 2 >= 2N-1
    f = np.zeros(size)
    f[:N] = s
    F = np.fft.fft(f)
    ac = np.fft.ifft(F * F).real[:2 * N - 1]
    # The continuous integral scales by dx^2; dx = 1/(2N)
    dx = 1.0 / (2 * N)
    # L1 proxy
    return np.sum(np.abs(ac)) * dx * dx

def _autoconv_linf(s):
    """Return the L∞ norm of the discrete autoconvolution (more faithful to C3)."""
    size = 1 << (2 * N - 1).bit_length()
    f = np.zeros(size)
    f[:N] = s
    F = np.fft.fft(f)
    ac = np.fft.ifft(F * F).real[:2 * N - 1]
    dx = 1.0 / (2 * N)
    return np.max(np.abs(ac)) * dx * dx

# ---------------------------------------------------------------------------
# Initialisers
# ---------------------------------------------------------------------------
def _init_alternating_blocks(block_len=4):
    """Alternating +1/-1 blocks of fixed length."""
    signs = np.zeros(N, dtype=np.float64)
    for i in range(N):
        signs[i] = 1.0 if (i // block_len) % 2 == 0 else -1.0
    return signs

def _init_sine(frequency=1.0, phase=0.0):
    """Sign of a low-frequency sine wave."""
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    return np.sign(np.sin(2 * np.pi * frequency * t + phase))
    # replace zeros
    # (sign(0)=0, so force to +1)
    return np.where(np.sign(np.sin(2 * np.pi * frequency * t + phase)) == 0, 1.0,
                    np.sign(np.sin(2 * np.pi * frequency * t + phase)))

def _init_random(seed=0):
    """Random ±1 with a fixed seed for reproducibility."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _init_large_blocks(n_blocks=10, seed=0):
    """Random partition into n_blocks contiguous blocks, each ±1."""
    rng = np.random.RandomState(seed)
    boundaries = sorted(rng.choice(range(1, N), size=n_blocks - 1, replace=False))
    signs = np.zeros(N, dtype=np.float64)
    current = 1.0 if rng.rand() < 0.5 else -1.0
    b = 0
    for i in range(N):
        signs[i] = current
        if b < len(boundaries) and i >= boundaries[b]:
            current = -current
            b += 1
    return signs

# ---------------------------------------------------------------------------
# Greedy coordinate descent
# ---------------------------------------------------------------------------
def _greedy_flip(s, deadline):
    """Flip the single coordinate giving the largest decrease in L1 energy."""
    best_gain = 0.0
    best_idx = -1
    base_energy = _autoconv_energy(s)
    for i in range(N):
        if time.time() > deadline:
            break
        s[i] = -s[i]
        e = _autoconv_energy(s)
        s[i] = -s[i]   # restore
        gain = base_energy - e
        if gain > best_gain:
            best_gain = gain
            best_idx = i
    if best_idx >= 0:
        s[best_idx] = -s[best_idx]
        return s, _autoconv_energy(s)
    return s, base_energy

# ---------------------------------------------------------------------------
# Simulated annealing polish
# ---------------------------------------------------------------------------
def _anneal(s, deadline, T0=0.01, T1=1e-4, cooling=0.95):
    """Metropolis annealing on the sign vector."""
    rng = np.random.RandomState(42)
    T = T0
    best_s = s.copy()
    best_e = _autoconv_energy(s)
    cur_e = best_e
    while T > T1 and time.time() < deadline:
        i = rng.randint(N)
        s[i] = -s[i]
        new_e = _autoconv_energy(s)
        delta = new_e - cur_e
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-9)):
            cur_e = new_e
            if new_e < best_e:
                best_e = new_e
                best_s = s.copy()
        else:
            s[i] = -s[i]   # reject
        T *= cooling
    return best_s, best_e

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY

    # Candidate initialisers
    inits = [
        _init_alternating_blocks(block_len=2),
        _init_alternating_blocks(block_len=4),
        _init_alternating_blocks(block_len=8),
        _init_sine(frequency=1.0, phase=0.0),
        _init_sine(frequency=2.0, phase=np.pi / 4),
        _init_sine(frequency=3.0, phase=np.pi / 2),
        _init_random(seed=7),
        _init_large_blocks(n_blocks=8, seed=3),
        _init_large_blocks(n_blocks=16, seed=5),
    ]

    # Evaluate initial scores
    scored = []
    for s in inits:
        if time.time() > deadline:
            break
        e = _autoconv_energy(s)
        scored.append((e, s))
    if not scored:
        # Fallback
        return np.ones(N)
    scored.sort(key=lambda x: x[0])
    best_s, best_e = scored[0]

    # Greedy coordinate descent from the best initialiser
    s = best_s.copy()
    e = best_e
    max_greedy_rounds = 200   # cap to stay within time
    for _ in range(max_greedy_rounds):
        if time.time() > deadline:
            break
        s_new, e_new = _greedy_flip(s, deadline)
        if e_new >= e:
            break
        s = s_new
        e = e_new
        best_s = s.copy()
        best_e = e

    # Annealing polish from the best greedy result
    if time.time() < deadline:
        s_ann, e_ann = _anneal(s, deadline)
        if e_ann < best_e:
            best_s = s_ann
            best_e = e_ann

    # Try greedy from second-best initialiser if time remains
    if len(scored) > 1 and time.time() < deadline:
        s2 = scored[1][1].copy()
        e2 = scored[1][0]
        for _ in range(100):
            if time.time() > deadline:
                break
            s2, e2 = _greedy_flip(s2, deadline)
        if e2 < best_e:
            best_s = s2
            best_e = e2

    # Final anneal if time permits
    if time.time() < deadline - 2.0:
        s_final, e_final = _anneal(best_s, deadline, T0=0.005, T1=1e-5, cooling=0.93)
        if e_final < best_e:
            best_s = s_final
            best_e = e_final

    # Ensure output is a sequence of signed step heights
    # Convert to list for clean return
    return list(best_s)

# ---------------------------------------------------------------------------
# Quick self-test (removed before submission to save time)
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     import time
#     t0 = time.time()
#     result = construct()
#     t1 = time.time()
#     print(f"Time: {t1-t0:.2f}s, len={len(result)}")
#     print(f"Energy (L1): {_autoconv_energy(np.array(result))}")
#     print(f"Linf: {_autoconv_linf(np.array(result))}")
