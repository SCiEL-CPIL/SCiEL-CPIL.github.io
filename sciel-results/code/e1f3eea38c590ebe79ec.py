import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def score_from_z(z):
        """Compute score from complex vector z where h = |z|^2."""
        h = np.abs(z)**2
        return score(h)

    def autoconv_peak_from_z(z):
        """Compute max of autoconvolution of h=|z|^2 using FFT."""
        h = np.abs(z)**2
        # Use FFT to compute autoconvolution
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
        return np.max(conv)

    def objective_from_z(z_flat):
        """Negative log-ratio to minimize. z_flat is real vector of length 2N (real+imag parts)."""
        z = z_flat[:N] + 1j * z_flat[N:]
        h = np.abs(z)**2
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 1e10
        l2_sq = np.sum(h**2)
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 1e10
        ratio = l2_sq / (l1 * max_conv)
        return -np.log(ratio)

    def gradient_objective_from_z(z_flat):
        """Compute gradient of negative log-ratio using finite differences or analytic."""
        z = z_flat[:N] + 1j * z_flat[N:]
        h = np.abs(z)**2
        l1 = np.sum(h)
        if l1 < 1e-10:
            return np.zeros(2*N)
        l2_sq = np.sum(h**2)
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return np.zeros(2*N)
        
        ratio = l2_sq / (l1 * max_conv)
        log_ratio = np.log(ratio)
        
        # Analytic gradient via chain rule
        # d log_ratio / dz_i = (1/ratio) * d ratio / dz_i
        # ratio = l2_sq / (l1 * max_conv)
        # d log_ratio = d l2_sq/l2_sq - d l1/l1 - d max_conv/max_conv
        
        # d l2_sq / dz_i = 4 * |z_i|^2 * z_i.conj() * ... actually let's use real representation
        # h_i = |z_i|^2 = x_i^2 + y_i^2 where z_i = x_i + i*y_i
        # d h_i / dx_i = 2*x_i, d h_i / dy_i = 2*y_i
        
        # d l2_sq / dx_i = 2*h_i * 2*x_i = 4*x_i*h_i
        # d l1 / dx_i = 2*x_i
        # d max_conv / dx_i: need to find which lag achieves max
        
        peak_idx = np.argmax(conv)
        shift = peak_idx - (N - 1)
        
        # max_conv = sum_j h_j * h_{j+shift} (with bounds)
        # d max_conv / dx_i = 2*x_i * (h_{i+shift} + h_{i-shift}) if both in range
        d_maxconv_dx = np.zeros(N)
        d_maxconv_dy = np.zeros(N)
        for i in range(N):
            coeff = 0.0
            if 0 <= i + shift < N:
                coeff += h[i + shift]
            if 0 <= i - shift < N:
                coeff += h[i - shift]
            d_maxconv_dx[i] = 2 * z[i].real * coeff
            d_maxconv_dy[i] = 2 * z[i].imag * coeff
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        for i in range(N):
            dl2sq_dx = 4 * z[i].real * h[i]
            dl1_dx = 2 * z[i].real
            dlog = dl2sq_dx / l2_sq - dl1_dx / l1 - d_maxconv_dx[i] / max_conv
            grad_x[i] = -dlog  # negative because we minimize -log_ratio
            
            dl2sq_dy = 4 * z[i].imag * h[i]
            dl1_dy = 2 * z[i].imag
            dlog = dl2sq_dy / l2_sq - dl1_dy / l1 - d_maxconv_dy[i] / max_conv
            grad_y[i] = -dlog
        
        return np.concatenate([grad_x, grad_y])

    def optimize_z(z_init, max_iter=500, lr_init=0.01):
        """Optimize z using gradient descent with momentum."""
        z_flat = np.concatenate([z_init.real, z_init.imag])
        best_z_flat = z_flat.copy()
        best_obj = objective_from_z(z_flat)
        best_h = np.abs(z_init)**2
        
        velocity = np.zeros(2*N)
        lr = lr_init
        beta = 0.9
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            obj = objective_from_z(z_flat)
            if obj < best_obj:
                best_obj = obj
                best_z_flat = z_flat.copy()
                best_h = np.abs(z_flat[:N] + 1j * z_flat[N:])**2
            
            grad = gradient_objective_from_z(z_flat)
            
            # Adaptive learning rate
            if it % 50 == 0 and it > 0:
                obj_prev = objective_from_z(z_flat - velocity * lr)
                if obj_prev < obj:
                    lr *= 0.5
                else:
                    lr *= 1.5
                    lr = min(lr, 0.1)
            
            velocity = beta * velocity + (1 - beta) * grad
            z_flat = z_flat - lr * velocity
            
            # Convergence check
            if it > 50 and abs(obj - best_obj) < 1e-12:
                break
        
        return best_h, score(best_h)

    # Generate initial candidates for z
    rng = np.random.RandomState(42)
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various structured initializations
    # 1. Sparse support with random phases
    for k in [3, 4, 5, 6, 7, 8, 10, 12]:
        for seed in range(10):
            if time.time() > deadline - 10.0:
                break
            rng_s = np.random.RandomState(seed * 1000 + k)
            support = sorted(rng_s.choice(N, k, replace=False).tolist())
            z = np.zeros(N, dtype=complex)
            for i in support:
                z[i] = 1.0 / np.sqrt(k) * np.exp(1j * rng_s.uniform(0, 2*np.pi))
            h_opt, s = optimize_z(z, max_iter=300)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    # 2. Random complex vectors
    for seed in range(50):
        if time.time() > deadline - 10.0:
            break
        rng_s = np.random.RandomState(seed)
        z = (rng_s.randn(N) + 1j * rng_s.randn(N)) / np.sqrt(2*N)
        h_opt, s = optimize_z(z, max_iter=200)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # 3. Try to reduce support after optimization
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        # Try removing small elements
        support = list(np.where(best_h > 1e-10)[0])
        if len(support) <= 2:
            break
        # Sort by magnitude
        mags = [(best_h[i], i) for i in support]
        mags.sort(reverse=True)
        # Try removing the smallest
        i_remove = mags[-1][1]
        h_trial = best_h.copy()
        h_trial[i_remove] = 0
        s_trial = score(h_trial)
        if s_trial >= best_score * 0.99:
            best_h = h_trial
            best_score = s_trial
    
    # Final polish with coordinate ascent
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        improved = False
        support = list(np.where(best_h > 1e-12)[0])
        for i in support:
            for delta in [0.01, -0.01, 0.005, -0.005]:
                h_trial = best_h.copy()
                h_trial[i] = max(0, h_trial[i] + delta)
                s = score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return list(best_h)
