import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s margin for safety
    
    best_h = np.ones(N)
    best_val = 2.0
    
    def compute_value(h):
        c = np.convolve(h, h)
        S = np.sum(h)
        if abs(S) < 1e-12:
            return np.inf
        return abs(2 * N * np.max(c) / S**2)
    
    # Strategy 1: Frequency-domain optimization with parameterized spectral shape
    # h[n] = IDFT(H[k]) where H[k] is a parameterized complex spectrum
    # We want |H[k]|^2 to be flat except for a null at k=0 (zero frequency)
    
    # Use a rational function parameterization:
    # |H[k]|^2 = 1 / (1 + a * cos(2*pi*k/N)^2) for k != 0
    # This gives a spectral notch at DC (k=0) and flat elsewhere
    
    def spectral_shape(a, k):
        """Parameterized spectral shape with notch at DC"""
        cos_k = np.cos(2 * np.pi * k / N)
        return 1.0 / (1.0 + a * cos_k**2)
    
    def make_h_from_params(params):
        """Create h from spectral parameters"""
        a = params[0]  # notch depth parameter
        k = np.arange(N)
        power = spectral_shape(a, k)
        
        # Random phases for diversity
        rng = np.random.default_rng(123)
        phases = rng.uniform(0, 2*np.pi, N)
        
        H = np.sqrt(power) * np.exp(1j * phases)
        
        # Ensure zero DC component (sum h = 0 approximately)
        H[0] = 0
        
        h = np.fft.ifft(H).real
        h = h / np.sum(h)  # Normalize
        return h
    
    # Optimize over spectral parameter a
    def spectral_objective(params):
        h = make_h_from_params(params)
        return compute_value(h)
    
    # Try several values of a
    for a in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        if time.time() > deadline:
            break
        
        # Use multiple random phase initializations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            rng = np.random.default_rng(1000 + trial)
            phases = rng.uniform(0, 2*np.pi, N)
            
            def make_h_fixed_phases(a_val):
                k = np.arange(N)
                power = spectral_shape(a_val, k)
                H = np.sqrt(power) * np.exp(1j * phases)
                H[0] = 0
                h = np.fft.ifft(H).real
                h = h / np.sum(h)
                return h
            
            h0 = make_h_fixed_phases(a)
            val0 = compute_value(h0)
            if val0 < best_val:
                best_val = val0
                best_h = h0.copy()
            
            # Local optimization around this point
            if time.time() > deadline:
                break
            
            # Use gradient-free optimization on the spectral parameter
            # and also try direct gradient descent on h
            
            # Gradient descent on h with smooth surrogate
            beta = 5.0
            for step in range(200):
                if time.time() > deadline:
                    break
                
                c = np.convolve(h0, h0)
                S = np.sum(h0)
                val = abs(2 * N * np.max(c) / S**2)
                
                if val < best_val:
                    best_val = val
                    best_h = h0.copy()
                
                # Softmax surrogate for max(c)
                max_c = np.max(c)
                exp_c = np.exp(beta * (c - max_c))
                sum_exp = np.sum(exp_c)
                g = exp_c / sum_exp
                
                # Gradient of sum(g_k * c_k) w.r.t h
                # grad[i] = sum_k g[k] * (h[k-i] + h[i-k])
                # Use FFT-based correlation
                # This is equivalent to: grad = 2 * ifft(fft(h) * conj(fft(g_padded)))
                # But need to be careful with the convolution structure
                
                # For c = convolve(h, h), dc_k/dh_i = h_{k-i} + h_{i-k}
                # grad[i] = sum_k g[k] * h[k-i] + sum_k g[k] * h[i-k]
                
                # First term: sum_k g[k] h[k-i] = (g * h)[i] where * is cross-correlation
                # In FFT: ifft(fft(g) * conj(fft(h)))
                
                g_padded = np.zeros(2*N-1, dtype=complex)
                g_padded[:len(g)] = g
                h_padded = np.zeros(2*N-1, dtype=complex)
                h_padded[:N] = h0
                
                # Cross-correlation: sum_k g[k] h[k-i]
                # = ifft(conj(fft(g_padded)) * fft(h_padded))
                corr1 = np.fft.ifft(np.conj(np.fft.fft(g_padded)) * np.fft.fft(h_padded)).real
                # Second term: sum_k g[k] h[i-k] = ifft(fft(g_padded) * conj(fft(h_padded)))
                corr2 = np.fft.ifft(np.fft.fft(g_padded) * np.conj(np.fft.fft(h_padded))).real
                
                grad = corr1[:N] + corr2[:N]
                
                # Normalize gradient
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-10:
                    grad = grad / grad_norm
                
                # Adaptive step size
                alpha = 0.01
                h_new = h0 - alpha * grad
                h_new = h_new / np.sum(h_new)
                
                # Check if improvement
                val_new = compute_value(h_new)
                if val_new < val:
                    h0 = h_new
                else:
                    alpha *= 0.5
                    h0 = h0 - alpha * grad
                    h0 = h0 / np.sum(h0)
    
    # Strategy 2: Direct iterative reweighting
    # Start with random h, iteratively adjust to reduce peak autocorrelation
    
    rng = np.random.default_rng(42)
    for trial in range(10):
        if time.time() > deadline:
            break
        
        h = rng.standard_normal(N)
        h = h / np.sum(h)
        
        # Iterative reweighting: reduce weights where autocorrelation peaks
        for it in range(300):
            if time.time() > deadline:
                break
            
            c = np.convolve(h, h)
            S = np.sum(h)
            val = abs(2 * N * np.max(c) / S**2)
            
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Find where the peak is
            peak_idx = np.argmax(c)
            
            # Create a weight vector that penalizes contributions to the peak
            # c[peak_idx] = sum_i h[i] * h[peak_idx - i]
            # To reduce this, we want to reduce h[i] where h[peak_idx - i] is large
            
            # Simple approach: gradient descent with the actual objective
            # Use smooth approximation
            beta = 3.0
            max_c = np.max(c)
            exp_c = np.exp(beta * (c - max_c))
            sum_exp = np.sum(exp_c)
            g = exp_c / sum_exp
            
            # FFT-based gradient
            g_padded = np.zeros(2*N-1, dtype=complex)
            g_padded[:len(g)] = g
            h_padded = np.zeros(2*N-1, dtype=complex)
            h_padded[:N] = h
            
            corr1 = np.fft.ifft(np.conj(np.fft.fft(g_padded)) * np.fft.fft(h_padded)).real
            corr2 = np.fft.ifft(np.fft.fft(g_padded) * np.conj(np.fft.fft(h_padded))).real
            
            grad = corr1[:N] + corr2[:N]
            
            # Normalize
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Adaptive step with restart
            alpha = 0.02
            for sub_iter in range(5):
                h_new = h - alpha * grad
                h_new = h_new / np.sum(h_new)
                val_new = compute_value(h_new)
                
                if val_new < val:
                    h = h_new
                    break
                else:
                    alpha *= 0.5
            
            # Occasionally add noise to escape local minima
            if it % 50 == 49:
                noise = rng.standard_normal(N) * 0.01
                h = h + noise
                h = h / np.sum(h)
    
    return best_h
