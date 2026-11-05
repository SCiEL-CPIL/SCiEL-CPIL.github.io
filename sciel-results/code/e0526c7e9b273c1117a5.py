"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    # Efficient gradient of LSE via FFT:
    # grad_lse[i] = 2 * sum_j weights[i+j] * h[j] (for linear conv indices)
    # This is a cross-correlation, computable via FFT.
    def lse_grad_via_fft(h, weights):
        # weights has length 2N-1 (full autoconv length)
        # grad[i] = 2 * sum_{j} weights[i+j] * h[j] where 0<=i<N, 0<=j<N, 0<=i+j<2N-1
        # This is the cross-correlation of weights with h, reversed appropriately.
        # Use FFT: pad both to fft_size, multiply, ifft
        W = np.fft.rfft(weights, n=fft_size)
        H = np.fft.rfft(h, n=fft_size)
        # cross_corr[i] = sum_j weights[i+j]*h[j] = ifft(conj(W)*H)[i] shifted
        # Actually: sum_j w[i+j]*h[j] = (w * h)[i] where * is cross-correlation
        # In FFT: cross_corr = ifft(conj(fft(w)) * fft(h))
        # But we need w[i+j], not w[j-i]. Let's think carefully.
        # Let g[i] = sum_j w[i+j] * h[j] for i=0..N-1
        # Define w_rev[k] = w[2N-2-k] for k=0..2N-2
        # Then g[i] = sum_j w_rev[2N-2-i-j] * h[j] = (w_rev * h)[2N-2-i] (convolution)
        # So g = ifft(fft(w_rev)*fft(h))[2N-2-i] for i=0..N-1
        # Equivalently, g = reverse(ifft(fft(w_rev)*fft(h))[:N])
        w_rev = weights[::-1]
        W_rev = np.fft.rfft(w_rev, n=fft_size)
        H = np.fft.rfft(h, n=fft_size)
        conv_wrev_h = np.fft.irfft(W_rev * H, n=fft_size)
        # conv_wrev_h[k] = sum_j w_rev[k-j]*h[j] = sum_j w[2N-2-k+j]*h[j]
        # We want g[i] = sum_j w[i+j]*h[j]
        # Set 2N-2-k+j = i+j => k = 2N-2-i
        # So g[i] = conv_wrev_h[2N-2-i]
        g = conv_wrev_h[2*N-2 : 2*N-2-N : -1]
        return 2.0 * g

    # --- Smooth objective with log-sum-exp, using FFT for gradient ---
    BETA = 300.0

    def smooth_objective_and_grad(x):
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

        beta = BETA
        scaled = beta * (shifted - max_shifted)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        if sum_exp < 1e-300:
            return 1e10, np.zeros_like(h)

        lse = max_c + max_shifted + np.log(sum_exp) / beta
        weights = exp_scaled / sum_exp

        # Gradient via FFT
        grad_lse = lse_grad_via_fft(h, weights)

        f_val = 2.0 * N * lse / (s * s)
        if not np.isfinite(f_val):
            return 1e10, np.zeros_like(h)

        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)

        if f_val >= 0:
            grad = df_dh
        else:
            grad = -df_dh

        return abs(f_val), grad

    # Cache to avoid recomputation when objective and grad are called separately
    _cache = {'x': None, 'val': None, 'grad': None}

    def smooth_objective_cached(x):
        if _cache['x'] is None or not np.array_equal(_cache['x'], x):
            _cache['val'], _cache['grad'] = smooth_objective_and_grad(x)
            _cache['x'] = x.copy()
        return _cache['val']

    def smooth_grad_cached(x):
        if _cache['x'] is None or not np.array_equal(_cache['x'], x):
            _cache['val'], _cache['grad'] = smooth_objective_and_grad(x)
            _cache['x'] = x.copy()
        return _cache['grad']

    # --- Generate diverse initial points ---
    rng = np.random.RandomState(42)
    initial_points = []
    initial_points.append(np.ones(N).astype(float))

    # Random
    for _ in range(10):
        initial_points.append(rng.uniform(-1, 1, N))

    # Fourier-synthesized
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * (freq + 1) * t))

    # Multi-frequency with random phases
    for n_comp in [5, 8, 12, 15]:
        for _ in range(3):
            freqs = rng.choice(np.arange(1, 30), size=n_comp, replace=False)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Bimodal
    for _ in range(6):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Sort by initial value
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE - 5:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1

        # Clear cache for each new starting point
        _cache['x'] = None

        try:
            result = minimize(
                smooth_objective_cached,
                init_h,
                jac=smooth_grad_cached,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-7}
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

        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()

    # --- Final polish with exact objective ---
    if time.time() - start_time < DEADLINE - 3:
        h_curr = best_h.copy()

        def exact_objective(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10
            conv = np.convolve(h, h)
            max_conv = np.max(conv)
            val = abs(2.0 * N * max_conv / (s * s))
            return val if np.isfinite(val) else 1e10

        try:
            result = minimize(
                exact_objective,
                h_curr,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-9}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # --- Perturbation phase ---
    if time.time() - start_time < DEADLINE - 2:
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 2:
                break
            h_pert = best_h + 0.03 * rng.normal(0, 1, N)
            _cache['x'] = None
            val_pert, valid = evaluate(h_pert)
            if valid and val_pert < best_value:
                best_value = val_pert
                best_h = h_pert
            else:
                try:
                    result = minimize(
                        smooth_objective_cached,
                        h_pert,
                        jac=smooth_grad_cached,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-7}
                    )
                    h_opt = result.x
                    val_opt, valid = evaluate(h_opt)
                    if valid and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    return best_h
