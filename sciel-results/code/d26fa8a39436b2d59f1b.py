import numpy as np
import time

N = 400


def _fast_score(x):
    """Compute the ratio: max|autocorr| / (sum(x))^2. Lower is better."""
    # Autocorrelation via FFT
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return max_abs_ac / (s * s)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_x = np.ones(N)
    best_score = _fast_score(best_x)

    # Generate diverse initializations
    inits = []
    inits.append(best_x.copy())  # all ones
    inits.append(np.random.choice([-1, 1], size=N))
    inits.append(np.sign(np.random.randn(N)))
    # Tent / triangular
    t = np.linspace(-1, 1, N)
    inits.append(np.sign(1 - np.abs(t)))
    # Multi-frequency
    inits.append(np.sign(np.sin(2 * np.pi * 3 * np.linspace(0, 1, N)) + 0.5 * np.sin(2 * np.pi * 7 * np.linspace(0, 1, N))))
    # Sparse
    sp = np.zeros(N)
    idx = np.random.choice(N, size=N // 4, replace=False)
    sp[idx] = np.random.choice([-1, 1], size=len(idx))
    inits.append(sp)

    # Evaluate initializations
    for x in inits:
        s = _fast_score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()

    # Gradient-free random perturbation search (fast objective-only)
    np.random.seed(42)
    x = best_x.copy()
    cur_score = best_score

    # Use a soft-max surrogate: log-sum-exp of |ac| components, scaled
    def _soft_score(x, beta=100):
        n = len(x)
        f = np.fft.fft(x, n=2 * n)
        ac = np.fft.ifft(f * np.conj(f)).real[:n]
        # Soft max of |ac| via log-sum-exp
        log_sum_exp = beta * np.log(np.sum(np.exp(-beta * np.abs(ac) / beta)))  # placeholder
        # Actually use: log-sum-exp(beta * |ac|) / beta as soft max
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        if m < 1e-300:
            max_abs_ac = 0.0
        else:
            max_abs_ac = m + np.log(np.sum(np.exp(beta * (abs_ac - m)))) / beta
        s = np.sum(x)
        if abs(s) < 1e-12:
            return np.inf
        return max_abs_ac / (s * s)

    # Random perturbation phase with fast scoring
    step_size = 1.0
    for i in range(500000):
        if time.time() > deadline:
            break
        # Random perturbation
        pert = np.random.randn(N) * step_size
        x_new = x + pert
        # Clip to [-2, 2] to keep values bounded
        x_new = np.clip(x_new, -2, 2)

        sc = _fast_score(x_new)
        if sc < cur_score:
            x = x_new
            cur_score = sc
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
            # Slightly increase step size on success (optional)
            step_size = min(step_size * 1.01, 3.0)
        else:
            step_size = max(step_size * 0.99, 0.1)

    # Polish phase with soft-max surrogate for smoother landscape
    if time.time() < deadline - 5:
        x = best_x.copy()
        cur_score = _soft_score(x, beta=100)
        best_soft = cur_score
        step_size = 0.1
        for i in range(200000):
            if time.time() > deadline:
                break
            pert = np.random.randn(N) * step_size
            x_new = np.clip(x + pert, -2, 2)
            sc = _soft_score(x_new, beta=100)
            if sc < cur_score:
                x = x_new
                cur_score = sc
                step_size = min(step_size * 1.02, 1.0)
                if _fast_score(x) < best_score:
                    best_score = _fast_score(x)
                    best_x = x.copy()
            else:
                step_size = max(step_size * 0.98, 0.01)

    # Final high-precision polish: local coordinate ascent on sign flips
    if time.time() < deadline - 3:
        x = best_x.copy()
        cur_score = _fast_score(x)
        for i in range(100000):
            if time.time() > deadline:
                break
            improved = False
            # Try flipping a random subset
            for _ in range(5):
                idx = np.random.randint(0, N)
                x[idx] = -x[idx]
                sc = _fast_score(x)
                if sc < cur_score:
                    cur_score = sc
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_x = x.copy()
                else:
                    x[idx] = -x[idx]  # revert
            if not improved:
                break

    # Return the best sequence found
    return best_x
