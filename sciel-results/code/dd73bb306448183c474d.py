import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak
    autocorrelation sidelobe.

    New approach:
    - Start from multiple random restarts.
    - Use a fast FFT-based peak sidelobe metric.
    - Local search via single-bit flips and small block flips.
    - Accept improvements greedily; occasionally allow small worsening moves
      to escape local minima (light tabu-like diversification via restarts).
    """
    rng = np.random.default_rng(12345)

    def peak_sidelobe(f):
        # Linear autocorrelation via FFT (zero-padded)
        F = np.fft.rfft(f, n=2 * N)
        acf = np.fft.irfft(F * np.conj(F), n=2 * N)
        # Nonzero lags: 1..N-1 and -(N-1)..-1
        # In zero-padded length 2N:
        # acf[0] is lag 0
        # acf[1:N] are lags 1..N-1
        # acf[N+1:2*N] are lags -(N-1)..-1
        pos = acf[1:N]
        neg = acf[N + 1:2 * N]
        if pos.size == 0 and neg.size == 0:
            return 0.0
        return float(np.max(np.abs(np.concatenate((pos, neg)))))

    best_f = None
    best_val = float("inf")

    # Time budget: stay safely under 120s
    t0 = time.time()
    time_budget = 110.0  # seconds

    # Number of restarts we can afford depends on runtime; we'll loop until time budget.
    # Each restart does a local search for a fixed number of iterations or until no improvement.
    max_iter_per_restart = 20000

    # We'll do several restarts; for diversity, we can also try different initial structures:
    # 1) random signs
    # 2) alternating signs
    # 3) blocks of +1/-1
    # 4) chirp-like (quadratic phase) signs

    def random_init():
        return rng.choice([-1, 1], size=N).astype(np.int8)

    def alternating_init():
        f = np.empty(N, dtype=np.int8)
        f[::2] = 1
        f[1::2] = -1
        return f

    def block_init(block_len):
        f = np.empty(N, dtype=np.int8)
        sign = 1
        for i in range(0, N, block_len):
            f[i:i + block_len] = sign
            sign = -sign
        return f

    def chirp_init():
        # sign = +1 if quadratic phase in [0, pi), else -1
        x = np.linspace(0, 1, N, endpoint=False)
        phase = 2 * np.pi * x * x
        f = np.sign(np.cos(phase)).astype(np.int8)
        # Replace zeros with +1
        f[f == 0] = 1
        return f

    init_strategies = [
        random_init,
        alternating_init,
        lambda: block_init(16),
        lambda: block_init(32),
        lambda: block_init(64),
        chirp_init,
    ]

    # We'll iterate over strategies repeatedly.
    strat_idx = 0
    while (time.time() - t0) < time_budget:
        init_fn = init_strategies[strat_idx % len(init_strategies)]
        strat_idx += 1

        f = init_fn().astype(np.int8).copy()
        cur_val = peak_sidelobe(f)
        if cur_val < best_val:
            best_val = cur_val
            best_f = f.copy()

        # Local search
        # We'll alternate between single flips and small block flips.
        # Use a simple acceptance rule: accept if improves; occasionally accept small worsenings.
        # To avoid getting stuck, we cap iterations and rely on restarts.
        no_improve = 0
        max_no_improve = 5000

        for it in range(max_iter_per_restart):
            if (time.time() - t0) >= time_budget:
                break

            # Choose move type: mostly single flips, some block flips
            if rng.random() < 0.8:
                i = int(rng.integers(N))
                f[i] = -f[i]
                new_val = peak_sidelobe(f)
                if new_val < cur_val:
                    cur_val = new_val
                    no_improve = 0
                    if cur_val < best_val:
                        best_val = cur_val
                        best_f = f.copy()
                else:
                    # revert
                    f[i] = -f[i]
                    no_improve += 1
                    # small chance to accept a worsening move (light diversification)
                    if no_improve > 1000 and rng.random() < 0.001:
                        f[i] = -f[i]
                        cur_val = new_val
                        no_improve = 0
            else:
                # block flip of length 2..8
                blen = int(rng.integers(2, 9))
                start = int(rng.integers(0, N - blen + 1))
                f[start:start + blen] = -f[start:start + blen]
                new_val = peak_sidelobe(f)
                if new_val < cur_val:
                    cur_val = new_val
                    no_improve = 0
                    if cur_val < best_val:
                        best_val = cur_val
                        best_f = f.copy()
                else:
                    f[start:start + blen] = -f[start:start + blen]
                    no_improve += 1
                    if no_improve > 1000 and rng.random() < 0.0005:
                        f[start:start + blen] = -f[start:start + blen]
                        cur_val = new_val
                        no_improve = 0

            if no_improve >= max_no_improve:
                break

    if best_f is None:
        best_f = random_init()

    return best_f
# EVOLVE-BLOCK-END
