import numpy as np
import time

N = 600
TIME_LIMIT = 135  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def _fft_convolve(a, b):
    n = len(a) + len(b) - 1
    size = 1
    while size < n:
        size *= 2
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    return np.fft.irfft(fa * fb, size)[:n]

def _smooth_max(conv, t):
    return t * np.log(np.sum(np.exp(conv / t)))

def _smooth_max_grad(conv, t):
    exp_c = np.exp(conv / t)
    weights = exp_c / np.sum(exp_c)
    return weights

def _fista_optimize(f0, max_iter=5000, time_limit=80.0):
    """FISTA for min max(conv) s.t. f>=0, sum(f)=1."""
    f = f0.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f
    f = f / s

    # Use a moderate smoothness parameter
    t = 0.001  # temperature for smooth max

    # FISTA parameters
    L = 1.0  # Lipschitz constant (rough estimate)
    y = f.copy()
    z = f.copy()
    t_k = 1.0

    start_time = time.time()
    best_f = f.copy()
    best_score = compute_score(f)

    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        # Gradient of smooth max of conv
        conv = _fft_convolve(z, z)
        grad = _smooth_max_grad(conv, t)
        
        # Gradient of smooth max w.r.t. f:
        # d/dz (t * log(sum(exp(conv_k/t)))) = sum_k weights_k * dconv_k/dz
        # dconv_k/dz_j = f[k-j] + f[j-k] (approximately, for real convolution)
        # Actually, conv = f * f (full), so dconv/dz is the gradient
        # Let me compute it properly using FFT
        # grad_conv_k = f reversed appropriately
        
        # Simpler: use finite difference or analytical
        # For conv_k = sum_i f_i f_{k-i}, dconv_k/dz_j = f_{k-j} + f_{j-k} (if in range)
        # This is expensive for all k. Let's use FFT approach.
        # Actually, the gradient of conv with respect to z is:
        # grad_z[full] = FFT^{-1}(weights_full * FFT(z_full) + FFT(z_full) * FFT(weights_full_reversed))
        
        # Let's just do a direct but vectorized computation for the gradient
        # We need: for each j, grad[j] = sum_k w_k * (f[k-j] + f[j-k])
        # This is like a convolution of w with f, plus f convolved with reversed w
        
        w = grad  # weights from smooth max
        # Gradient of sum_k w_k conv_k w.r.t f_j
        # = sum_k w_k (f[k-j] + f[j-k])
        # = (w * f)[j] + (w_rev * f)[j]
        # where w_rev[k] = w[-k] = w[N-1-k] approximately
        
        # Use FFT to compute these convolutions
        w_pad = np.zeros(2 * N - 1)
        w_pad[:N] = w
        f_pad = np.zeros(2 * N - 1)
        f_pad[:N] = z
        
        size2 = 1
        while size2 < 4 * N - 1:
            size2 *= 2
        
        wf = np.fft.rfft(w_pad, size2)
        ff = np.fft.rfft(f_pad, size2)
        conv_wf = np.fft.irfft(wf * ff, size2)[:2*N-1]
        
        # For the second term, reverse w
        w_rev = w[::-1]
        w_rev_pad = np.zeros(2 * N - 1)
        w_rev_pad[:N] = w_rev
        wrf = np.fft.rfft(w_rev_pad, size2)
        conv_wr_f = np.fft.irfft(wrf * ff, size2)[:2*N-1]
        
        total_grad = conv_wf[:N] + conv_wr_f[:N]
        
        # Step size
        step = 1.0 / L
        y_new = z - step * total_grad
        
        # Project onto simplex {f >= 0, sum(f) = 1}
        y_proj = _project_simplex(y_new)
        
        # FISTA update
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
        z = y_proj
        y = z + (t_k - 1.0) / t_new * (z - y)
        t_k = t_new
        
        # Check score
        score = compute_score(z)
        if score < best_score:
            best_score = score
            best_f = z.copy()

    return best_f

def _project_simplex(v):
    """Project v onto {f >= 0, sum(f) = 1}."""
    v = np.maximum(v, 0.0)
    s = np.sum(v)
    if s > 1e-15:
        v = v / s
    else:
        v = np.ones(N) / N
    return v

def _cosine_basis_signal(freqs, amps):
    """
    Build a nonnegative signal from a cosine basis.
    f(x) = 1 + sum_k a_k cos(2*pi*f_k*x + phi_k)
    We ensure nonnegativity by keeping the DC term dominant.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.ones(N)
    for k in range(len(amps)):
        freq = freqs[k]
        amp = amps[k]
        f += amp * np.cos(2 * np.pi * freq * x)
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    return f

def _parametric_to_step(params):
    """
    Convert parametric representation to a 600-step vector.
    Uses sum of Gaussian bumps for flexibility.
    """
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.zeros(N)
    n_bumps = len(params) // 3
    for i in range(n_bumps):
        c = params[3*i]
        w = params[3*i+1]
        a = params[3*i+2]
        w = max(abs(w), 0.005)
        f += a * np.exp(-0.5 * ((x - c) / w) ** 2)
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s > 1e-15:
        f = f / s
    return f

def _objective_parametric(params):
    """Objective function for parametric optimization."""
    try:
        f = _parametric_to_step(params)
        return compute_score(f)
    except Exception:
        return 1e10

def _coordinate_descent(f0, max_iter=2000, time_limit=30.0):
    """Coordinate descent on individual bins."""
    f = f0.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f
    f = f / s
    
    start_time = time.time()
    best_score = compute_score(f)
    best_f = f.copy()
    
    # Try modifying pairs of bins
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random pair of bins to swap mass
        j1 = np.random.randint(0, N)
        j2 = np.random.randint(0, N)
        if j1 == j2:
            continue
        
        # Try moving a small amount from j1 to j2
        delta = 0.01 * f[j1]
        if delta < 1e-10:
            continue
        
        f_trial = f.copy()
        f_trial[j1] -= delta
        f_trial[j2] += delta
        f_trial = np.maximum(f_trial, 0.0)
        s_trial = np.sum(f_trial)
        if s_trial < 1e-15:
            continue
        f_trial = f_trial / s_trial
        
        score_trial = compute_score(f_trial)
        if score_trial < best_score:
            best_score = score_trial
            best_f = f_trial.copy()
            f = f_trial
    
    return best_f

def construct():
    start_time = time.time()
    
    # Initialize with a good starting point: single Gaussian
    params_init = [0.0, 0.05, 1.0]
    f_init = _parametric_to_step(params_init)
    best_f = f_init.copy()
    best_score = compute_score(f_init)
    
    # Phase 1: Local FISTA optimization
    remaining = TIME_LIMIT - (time.time() - start_time) - 10.0
    if remaining > 20.0:
        f_fista = _fista_optimize(best_f, max_iter=3000, time_limit=min(remaining - 5.0, 80.0))
        score_fista = compute_score(f_fista)
        if score_fista < best_score:
            best_score = score_fista
            best_f = f_fista
    
    # Phase 2: Coordinate descent for fine-tuning
    remaining = TIME_LIMIT - (time.time() - start_time) - 5.0
    if remaining > 10.0:
        f_cd = _coordinate_descent(best_f, max_iter=1000, time_limit=min(remaining - 3.0, 30.0))
        score_cd = compute_score(f_cd)
        if score_cd < best_score:
            best_score = score_cd
            best_f = f_cd
    
    # Final cleanup
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
