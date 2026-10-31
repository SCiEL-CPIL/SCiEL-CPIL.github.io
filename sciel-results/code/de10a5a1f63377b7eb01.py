import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    p = e / np.sum(e)
    return p

def _project_sum(h, target_sum=N/2.0):
    """Project h onto {0<=h<=1, sum(h)=target_sum} via bisection on clip offset."""
    h = h.copy()
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_clipped = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_clipped)
        if s > target_sum:
            hi = mid
        else:
            lo = mid
    return np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)

def _generate_highpass_noise(cutoff_freq, amplitude, seed=42):
    """Generate high-pass filtered white noise with given cutoff and amplitude."""
    np.random.seed(seed)
    white_noise = np.random.randn(N)
    white_noise -= np.mean(white_noise)  # Remove DC component
    
    # Apply high-pass filter in frequency domain
    fft_noise = np.fft.fft(white_noise)
    freqs = np.fft.fftfreq(N)
    
    # High-pass filter: zero out low frequencies
    cutoff_idx = int(abs(cutoff_freq) * N / 2)
    for i in range(len(fft_noise)):
        if abs(freqs[i]) < cutoff_freq:
            fft_noise[i] = 0
    
    filtered = np.fft.ifft(fft_noise).real
    filtered -= np.mean(filtered)  # Ensure zero mean
    
    # Scale to desired amplitude
    std = np.std(filtered)
    if std > 1e-10:
        filtered = filtered / std * amplitude
    
    h = 0.5 + filtered
    h = _project_sum(h)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    h_best = np.full(N, 0.5)
    score_best = _score(h_best)
    
    # First, try high-pass filtered white noise with grid search
    cutoffs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    amplitudes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    for cutoff in cutoffs:
        if time.time() > deadline - 20:
            break
        for amp in amplitudes:
            if time.time() > deadline - 10:
                break
            for seed in [42, 123, 456]:
                h_candidate = _generate_highpass_noise(cutoff, amp, seed)
                sc = _score(h_candidate)
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()
    
    # Then, refine using Fourier basis optimization on the best candidate
    x = np.arange(N)
    
    for K in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > deadline - 15:
            break

        n_params = 2 * K

        basis = np.zeros((N, 2*K))
        for k in range(1, K+1):
            basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * x / N)
            basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * x / N)

        def get_h(params):
            return 0.5 + basis @ params

        def project(h):
            return _project_sum(h)

        def objective(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)
            return _smooth_max(overlap, alpha=200.0)

        def objective_grad(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)

            p = _smooth_max_grad(overlap, alpha=200.0)

            grad_hc = np.zeros(N)
            h_next = np.zeros(N)
            h_next[:-1] = h_c[1:]
            grad_hc += p * (1.0 - h_next)
            h_prev = np.zeros(N)
            h_prev[1:] = h_c[:-1]
            p_prev = np.zeros(N)
            p_prev[1:] = p[:-1]
            grad_hc -= p_prev * h_c

            mask = (h > 0.01) & (h < 0.99)
            grad_h = grad_hc * mask

            return basis.T @ grad_h

        # Initialize from the best high-pass noise candidate
        if time.time() > deadline - 10:
            break
            
        # Try to fit Fourier coefficients to h_best
        try:
            # Least squares fit: h_best ≈ 0.5 + basis @ params
            A = basis
            b = h_best - 0.5
            if np.all(np.abs(A) > 1e-10):
                params_init, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            else:
                params_init = np.zeros(2*K)
        except:
            params_init = np.zeros(2*K)

        # Also try small random initializations
        init_candidates = [params_init]
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            init_candidates.append(np.random.randn(2*K) * 0.05)

        for h0 in init_candidates:
            if time.time() > deadline - 8:
                break

            try:
                res = minimize(
                    objective,
                    h0,
                    jac=objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
                )

                h_candidate = project(get_h(res.x))
                sc = _score(h_candidate)
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()

            except Exception:
                pass

    # Final polish: try direct coordinate descent on the best solution
    if time.time() < deadline - 5:
        h_polish = h_best.copy()
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                for delta in [-0.01, 0.01, -0.005, 0.005, -0.001, 0.001]:
                    h_test = h_polish.copy()
                    h_test[i] = np.clip(h_test[i] + delta, 0, 1)
                    h_test = _project_sum(h_test)
                    sc_test = _score(h_test)
                    if sc_test < score_best - 1e-12:
                        h_polish = h_test
                        score_best = sc_test
                        improved = True
                        break
            if not improved:
                break
        h_best = h_polish

    return h_best
