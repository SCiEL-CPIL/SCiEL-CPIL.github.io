import numpy as np
import time

N = 400
FFT_SIZE = 1024  # power of 2 >= 2N-1 = 799

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    def _eval(h):
        """Compute the actual objective value."""
        S = np.sum(h)
        if S == 0:
            return np.inf
        r = np.convolve(h, h)
        return abs(2.0 * N * np.max(r) / (S * S))

    def _fft_convolve(h):
        """FFT-accelerated full autoconvolution."""
        H = np.fft.rfft(h, FFT_SIZE)
        r = np.fft.irfft(H * H, FFT_SIZE)[:2 * N - 1]
        return r

    def _objective(h, beta):
        """Smoothed objective: LSE of r / S^2, with gradient."""
        S = np.sum(h)
        r = _fft_convolve(h)
        r_max = np.max(r)
        exp_r = np.exp(beta * (r - r_max))
        sum_exp = np.sum(exp_r)
        lse = r_max + np.log(sum_exp) / beta
        val = lse / (S * S)
        
        # Gradient: d/dh [lse / S^2]
        # d(lse)/dh = 2 * convolve(h, w) where w_k = exp(beta*(r_k - r_max))/sum_exp
        w = exp_r / sum_exp
        # grad_lse = 2 * (h * w) in the sense of: sum_k w_k * dr_k/dh_m
        # dr_k/dh_m = h_{k-m} + h_{m-k} (with zero padding)
        # So grad_lse[m] = 2 * sum_k w_k * h_{k-m} (with bounds)
        # This is 2 * ifft(fft(h) * fft(w_padded))
        w_padded = np.zeros(FFT_SIZE)
        w_padded[:len(w)] = w
        H_fft = np.fft.rfft(h, FFT_SIZE)
        W_fft = np.fft.rfft(w_padded, FFT_SIZE)
        grad_lse = 2.0 * np.fft.irfft(H_fft * W_fft, FFT_SIZE)[:N]
        
        # d(lse/S^2)/dh = grad_lse/S^2 - 2*lse*h/S^3
        grad = grad_lse / (S * S) - 2.0 * lse * h / (S * S * S)
        return val, grad

    def _adam_step(h, grad, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        h_new = h - lr * m_hat / (np.sqrt(v_hat) + eps)
        return h_new, m, v

    # Best tracking
    best_h = np.ones(N) / N
    best_val = 2.0

    np.random.seed(42)

    # Restart configurations: (init_type, init_params)
    restarts = []
    # Random initializations with varying scales
    for scale in [0.1, 0.2, 0.5, 1.0]:
        restarts.append(('random', scale))
    # Low-frequency structured
    restarts.append(('lowfreq', 0.3))
    restarts.append(('lowfreq', 0.5))
    # High-frequency
    restarts.append(('highfreq', 0.2))
    # Mixed
    restarts.append(('mixed', 0.3))
    # Near-uniform perturbation
    restarts.append(('near_uniform', 0.1))
    restarts.append(('near_uniform', 0.05))

    for ri, (init_type, param) in enumerate(restarts):
        if time.time() > deadline - 10.0:
            break

        # Generate initial h
        if init_type == 'random':
            h = np.random.randn(N) * param
        elif init_type == 'lowfreq':
            # Sum of a few low-frequency cosines
            h = np.zeros(N)
            for k in range(1, 5):
                h += param * np.cos(2 * np.pi * k * np.arange(N) / N)
        elif init_type == 'highfreq':
            h = np.zeros(N)
            for k in range(N // 4, N // 2):
                h += param * 0.1 * np.cos(2 * np.pi * k * np.arange(N) / N)
        elif init_type == 'mixed':
            h = np.random.randn(N) * param * 0.5
            h += param * 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
        elif init_type == 'near_uniform':
            h = np.ones(N) / N + np.random.randn(N) * param / N
        else:
            h = np.random.randn(N) * param

        # Normalize to sum = 1 for numerical stability
        S = np.sum(h)
        if abs(S) < 1e-10:
            h = np.ones(N) / N
        else:
            h = h / S

        # Adam optimizer with cosine-annealed beta
        n_steps = 2000
        m = np.zeros(N)
        v = np.zeros(N)
        base_lr = 0.01
        beta_start = 10.0
        beta_end = 500.0

        for step in range(n_steps):
            if time.time() > deadline - 10.0:
                break

            # Cosine anneal beta
            progress = step / max(n_steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * (1 + np.cos(np.pi * progress)) / 2
            # Learning rate schedule
            lr = base_lr * (1 + np.cos(np.pi * progress)) / 2 + 1e-5

            val, grad = _objective(h, beta)

            # Check actual value periodically
            if step % 100 == 0 or step == n_steps - 1:
                actual_val = _eval(h)
                if actual_val < best_val:
                    best_val = actual_val
                    best_h = h.copy()

            # Adam step
            h, m, v = _adam_step(h, grad, m, v, step + 1, lr)

            # Keep sum = 1 (renormalize to prevent drift)
            S = np.sum(h)
            if abs(S) > 1e-10:
                h = h / S

        # Final evaluation after this restart
        actual_val = _eval(h)
        if actual_val < best_val:
            best_val = actual_val
            best_h = h.copy()

    # Final refinement: take best_h and do a short high-temperature optimization
    if time.time() < deadline - 15.0:
        h = best_h.copy()
        S = np.sum(h)
        if abs(S) < 1e-10:
            h = np.ones(N) / N
        else:
            h = h / S

        m = np.zeros(N)
        v = np.zeros(N)
        n_refine = 500
        for step in range(n_refine):
            if time.time() > deadline - 5.0:
                break
            progress = step / max(n_refine - 1, 1)
            beta = 200.0 + 300.0 * (1 + np.cos(np.pi * progress)) / 2
            lr = 0.005 * (1 + np.cos(np.pi * progress)) / 2 + 1e-5

            val, grad = _objective(h, beta)
            h, m, v = _adam_step(h, grad, m, v, step + 1, lr)

            S = np.sum(h)
            if abs(S) > 1e-10:
                h = h / S

        actual_val = _eval(h)
        if actual_val < best_val:
            best_val = actual_val
            best_h = h.copy()

    # Ensure finite and nonzero sum
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        best_h = np.ones(N) / N

    return best_h
