import numpy as np
import time

N = 600
DEADLINE_SECONDS = 115.0

def construct():
    start_time = time.time()
    
    # Compute autoconvolution peak (normalized)
    def score(f):
        n_fft = 2 * N
        F = np.fft.rfft(f, n=n_fft)
        autoconv = np.fft.irfft(F * np.conj(F), n=n_fft)
        peak = np.max(autoconv[:N])
        integral = np.sum(f)
        if integral <= 0:
            return np.inf
        return peak / (integral * integral)
    
    # Use spectral approach: minimize peak of IFFT(|FFT(f)|^2)
    # Parameterize f = exp(z) to ensure nonnegativity
    # Use smooth surrogate: log-sum-exp of autoconvolution values
    
    def smooth_score_and_grad(z, alpha=10.0):
        f = np.exp(z)
        n_fft = 2 * N
        F = np.fft.rfft(f, n=n_fft)
        autoconv = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
        
        # Normalize
        integral = np.sum(f)
        ac_norm = autoconv / (integral * integral)
        
        # Smooth max via log-sum-exp
        max_val = np.max(ac_norm)
        shifted = ac_norm - max_val
        log_sum_exp = max_val + np.log(np.sum(np.exp(alpha * shifted)) / N)
        score_val = log_sum_exp
        
        # Gradient of log-sum-exp w.r.t. ac_norm
        weights = np.exp(alpha * (ac_norm - max_val)) / np.sum(np.exp(alpha * (ac_norm - max_val)))
        
        # d(ac_norm)/df: ac_norm[k] = autoconv[k] / integral^2
        # d(autoconv[k])/df[j] = f[j+k] (if j+k<N) + f[j-k] (if j-k>=0)
        # d(integral^2)/df[j] = 2 * integral
        
        # d(ac_norm)/df[j] = (d(autoconv)/df[j] * integral^2 - autoconv * 2*integral) / integral^4
        # = d(autoconv)/df[j] / integral^2 - 2*autoconv / integral^3
        
        grad_ac = np.zeros(N)
        for k in range(N):
            wk = weights[k]
            if wk < 1e-10:
                continue
            # d(autoconv[k])/df[j]
            for j in range(N):
                val = 0.0
                if j + k < N:
                    val += f[j + k]
                if j - k >= 0:
                    val += f[j - k]
                grad_ac[j] += wk * (val / (integral * integral) - 2.0 * autoconv[k] / (integral ** 3))
        
        # Chain rule: d(score)/dz = d(score)/df * df/dz = grad_ac * f
        grad_z = grad_ac * f
        
        return score_val, grad_z
    
    # Adam optimizer
    z = np.zeros(N)
    m = np.zeros(N)
    v = np.zeros(N)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.05
    
    best_score = score(np.exp(z))
    best_f = np.exp(z).copy()
    
    it = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        score_val, grad_z = smooth_score_and_grad(z, alpha=10.0)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_z
        v = beta2 * v + (1 - beta2) * grad_z ** 2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        z -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        it += 1
        
        f = np.exp(z)
        cur_score = score(f)
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()
        
        # Reduce learning rate occasionally
        if it % 100 == 0:
            lr *= 0.95
    
    best_f = np.maximum(best_f, 0)
    return list(best_f)
