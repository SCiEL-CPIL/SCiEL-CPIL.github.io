import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Represent h as a chirp (frequency + linear phase ramp) + DC offset for global structure.
    2. Use FFT-based fast convolution and analytic gradient for L-BFGS-B.
    3. Global grid search over chirp parameters to find good basins.
    4. Local refinement with L-BFGS-B from diverse initial points.
    5. Coordinate descent polish.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    # Precompute FFT sizes
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        using FFT-based cross-correlation.
        """
        h_rev = h[::-1]
        fft_size_cc = 1
        while fft_size_cc < N + 2 * N - 1:
            fft_size_cc *= 2

        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)

        grad_lse = 2.0 * conv_result[N - 1:2 * N - 1]
        return grad_lse

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    def smooth_objective_and_grad(x, beta=80.0):
        """Compute both objective and gradient in one pass."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)

        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)

        exp_terms = np.exp(beta * (shifted - max_shifted))
        sum_weights = np.sum(exp_terms)
        if sum_weights < 1e-300:
            return 1e10, np.zeros_like(h)

        lse = max_c + max_shifted + np.log(sum_weights) / beta
        f = 2.0 * N * lse / (s * s)

        if not np.isfinite(f):
            return 1e10, np.zeros_like(h)

        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)

        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)

        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh

        return f, grad

    def smooth_objective(x, beta=80.0):
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=80.0):
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    # Evaluate seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, N, endpoint=False)
    i_arr = np.arange(N)

    # --- Chirp-based global search ---
    # h[i] = A * cos(2*pi*f*t[i] + chirp_rate * t[i]^2) + dc
    # Parameters: f (frequency), chirp_rate, dc (DC offset), A (amplitude)
    chirp_candidates = []

    # Grid search over frequency and chirp rate
    freq_range = np.arange(2, 30, 2)  # frequencies 2 to 28
    chirp_range = np.arange(-50, 51, 10)  # chirp rates
    dc_range = np.arange(-0.5, 0.51, 0.2)  # DC offsets
    amp_range = [1.0]

    for f in freq_range:
        for cr in chirp_range:
            for dc in dc_range:
                for A in amp_range:
                    h_chirp = A * np.cos(2 * np.pi * f * t + cr * t ** 2) + dc
                    s = np.sum(h_chirp)
                    if abs(s) < 1e-12:
                        continue
                    conv = fast_convolve(h_chirp)
                    val = abs(2.0 * N * np.max(conv) / (s * s))
                    chirp_candidates.append((val, h_chirp.copy()))

    # Sort by value and keep top candidates
    chirp_candidates.sort(key=lambda x: x[0])
    for val_c, h_c in chirp_candidates[:30]:
        if val_c < best_value:
            best_value = val_c
            best_h = h_c.copy()

    # --- Generate diverse initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Best chirp candidates
    for val_c, h_c in chirp_candidates[:15]:
        initial_points.append(h_c)

    # Random uniform
    for _ in range(15):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(15):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)

    # Low-frequency cosines
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))

    # Parameterized bump profiles
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)
            h_off = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) + 0.3 * np.ones(N)
            initial_points.append(h_off)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7, 10]:
        for _ in range(5):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Structured patterns: blocks of +1 and -1
    for block_size in [10, 20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(5):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b * block_size:(b + 1) * block_size] = sign
            initial_points.append(h_blocks)

    # Smooth random via Gaussian filter
    for _ in range(15):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1

        try:
            result = minimize(
                smooth_objective,
                init_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-5}
            )

            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass

        # Also try exact evaluation of the initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()

    # --- Local polish using coordinate descent on the best solution ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.05
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in [lr, -lr, lr * 0.5, -lr * 0.5]:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.7

    return best_h
