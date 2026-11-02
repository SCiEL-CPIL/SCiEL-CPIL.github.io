"""Fresh design: minimize the autoconvolution upper bound C3 = max_k |c_k|
for a signed step function (400 equal intervals on [-1/4,1/4]).

c_k = D * sum_j h_j h_{j+k}  (D = interval width = 1/100).
Minimizing max_k |c_k|  <=>  minimizing max_k |(h*h)_k|  (D is a constant factor),
which is exactly the third autocorrelation inequality upper bound.

Algorithm: random +-1 start + coordinate-flip local search with O(N) incremental
updates of the max-lag objective, repeated with random restarts under a wall-clock
deadline guard. Exact integer arithmetic via np.convolve (h in {+-1}).
"""
import time
import numpy as np

N = 400
DEADLINE = 150.0          # evaluator wall-clock limit
MARGIN = 6.0              # safety margin (s)
RNG_SEED = 12345


def _max_abs_conv(h):
    """Exact max over lags of | (h*h)_k | for h in {+-1}."""
    c = np.convolve(h, h, mode='full')          # length 2N-1, all positive ints
    return int(np.max(np.abs(c)))


def _construct():
    t0 = time.time()
    limit = DEADLINE - MARGIN
    rng = np.random.default_rng(RNG_SEED)

    best_h = None
    best_val = None

    # Phase 1: a few quick random starts to establish a baseline immediately.
    for _ in range(4):
        h = rng.choice(np.array([-1, 1], dtype=np.int64), size=N)
        v = _max_abs_conv(h)
        if best_val is None or v < best_val:
            best_val, best_h = v, h.copy()
        if time.time() > limit:
            break

    # Phase 2: coordinate-flip local search with random restarts.
    while time.time() < limit:
        # fresh random start (restart)
        h = rng.choice(np.array([-1, 1], dtype=np.int64), size=N)
        c = np.convolve(h, h, mode='full')
        L = 2 * N - 1
        cur = int(np.max(np.abs(c)))

        # local search
        while time.time() < limit:
            improved = False
            # random order of indices each pass
            order = rng.permutation(N)
            for i in order:
                # flip h[i]; c_k changes by 2*h[i]*h[j] for k in [i-(N-1), i+N-1]
                lo = i - (N - 1)
                hi = i + (N - 1)
                lo = max(0, lo)
                hi = min(L - 1, hi)
                if lo > hi:
                    continue
                jlo = lo - i
                jhi = hi - i
                delta = 2 * int(h[i]) * h[jlo:jhi + 1]
                new_c = c[lo:hi + 1] + delta
                new_max = int(np.max(np.abs(new_c)))
                if new_max < cur:
                    c[lo:hi + 1] = new_c
                    h[i] = -h[i]
                    cur = new_max
                    improved = True
            if not improved:
                break

        if cur < best_val:
            best_val, best_h = cur, h.copy()
            # keep improving from this best with occasional restarts handled by loop

    return best_h


def construct():
    return _construct()


# ---- local self-test (not called by evaluator) ----
if __name__ == '__main__':
    t = time.time()
    h = construct()
    print("time", time.time() - t, "N", len(h), "max_abs_conv", _max_abs_conv(h))
