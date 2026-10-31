import numpy as np
import time

N = 600
DEADLINE = 115

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)

    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0
    
    # Parametric families for initialization
    def make_bimodal(sigma, alpha):
        """Bimodal: two Gaussians at positions alpha*N and (1-alpha)*N"""
        pos1 = alpha * N
        pos2 = (1 - alpha) * N
        g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
        g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
        return g1 + g2
    
    def make_poly(c, d, e):
        """Polynomial: c + d*(x-center)^2 + e*(x-center)^4, clipped to nonneg"""
        x = (i - center) / (N / 2)
        val = c + d * x**2 + e * x**4
        return np.maximum(val, 0.01)
    
    # Generate diverse initializations
    inits = []
    
    # Uniform
    inits.append(np.ones(N))
    
    # Bimodal with various parameters
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            inits.append(make_bimodal(sigma, alpha))
    
    # Polynomial shapes
    for c, d, e in [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 0.5, 1), (2, 1, 0), (1, 3, 0)]:
        inits.append(make_poly(c, d, e))
    
    # U-shaped
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)
    
    # W-shape (two bumps)
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))
    
    # Random smooth
    np.random.seed(42)
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))
    
    best_h = None
    best_val = np.inf
    
    # Stage 1: Evaluate all initializations
    for init in inits:
        if time_left() < 10:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)
    
    # Stage 2: Cosine series parameterization with Remez-like equioscillation
    # Parameterize h as a low-order cosine series: h_k = sum_{m=0}^{M} a_m * cos(2*pi*m*k/N)
    # This ensures smooth optimization in coefficient space
    
    if time_left() > 30:
        M = 12  # Number of cosine terms
        # Initial coefficients from best_h via DCT
        H_freq = np.fft.rfft(best_h)
        coeffs = np.zeros(M + 1)
        coeffs[0] = H_freq[0].real / N
        for m in range(1, M + 1):
            if m < len(H_freq):
                coeffs[m] = 2 * H_freq[m].real / N
        
        def coeffs_to_h(c):
            """Convert cosine coefficients to heights."""
            h = np.zeros(N)
            for m in range(len(c)):
                h += c[m] * np.cos(2 * np.pi * m * i / N)
            return h
        
        def h_to_coeffs(h):
            """Convert heights to cosine coefficients."""
            H_freq = np.fft.rfft(h)
            c = np.zeros(M + 1)
            c[0] = H_freq[0].real / N
            for m in range(1, M + 1):
                if m < len(H_freq):
                    c[m] = 2 * H_freq[m].real / N
            return c
        
        def compute_conv_weights(h, beta=30.0):
            """Compute softmax weights on autoconvolution for smooth max."""
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            m = np.max(conv)
            exp_conv = np.exp((conv - m) * beta)
            weights = exp_conv / np.sum(exp_conv)
            return weights, m, H
        
        def compute_grad_coeff(c, beta=30.0):
            """Compute gradient of smooth max autoconvolution w.r.t. coefficients."""
            h = coeffs_to_h(c)
            h = np.maximum(h, 1e-10)
            s = np.sum(h)
            h_norm = h / s
            
            weights, m_val, H = compute_conv_weights(h_norm, beta)
            
            size = 2 * N
            w_full = np.zeros(size)
            w_full[:len(weights)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
            
            # Chain rule: d(conv_max)/dc = d(conv_max)/dh * dh/dc
            # dh/dc_m = cos(2*pi*m*i/N) for m >= 0
            grad_c = np.zeros(len(c))
            for m in range(len(c)):
                grad_c[m] = np.dot(grad_h, np.cos(2 * np.pi * m * i / N))
            
            return grad_c, m_val + np.log(np.sum(weights)) / beta
        
        # Remez-like equioscillation: identify peaks in autoconvolution and adjust coefficients
        # to minimize the maximum, promoting equal oscillation at multiple points
        
        def remez_step(c, beta=30.0):
            """Perform one Remez-like step: find peaks and adjust to equioscillate."""
            h = coeffs_to_h(c)
            h = np.maximum(h, 1e-10)
            s = np.sum(h)
            h_norm = h / s
            
            weights, m_val, H = compute_conv_weights(h_norm, beta)
            
            # Find top K peaks in the autoconvolution
            K = min(8, 2 * N)
            top_indices = np.argsort(weights)[-K:]
            top_vals = np.fft.ifft(H * H).real[top_indices]
            max_val = np.max(top_vals)
            
            # For equioscillation, we want to reduce the highest peaks
            # Compute a weighted gradient that emphasizes the peaks
            peak_mask = np.zeros(2 * N)
            peak_mask[top_indices] = weights[top_indices]
            peak_mask = peak_mask / np.sum(peak_mask)
            
            w_full = np.zeros(2 * N)
            w_full[:len(peak_mask)] = peak_mask
            W = np.fft.fft(w_full, n=2 * N)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
            
            # Project gradient onto coefficient space
            grad_c = np.zeros(len(c))
            for m in range(len(c)):
                grad_c[m] = np.dot(grad_h, np.cos(2 * np.pi * m * i / N))
            
            return grad_c, m_val
        
        # Optimize using combined gradient + Remez approach
        current_coeffs = coeffs.copy()
        current_val = best_val
        
        # Phase 1: Gradient descent to get close to optimum
        eta = 0.05
        for k in range(150):
            if time_left() < 15:
                break
            
            h_curr = coeffs_to_h(current_coeffs)
            h_curr = np.maximum(h_curr, 1e-10)
            h_curr = h_curr / np.sum(h_curr)
            curr_val = compute_autoconv_max(h_curr)
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()
            
            grad_c, smooth_val = compute_grad_coeff(current_coeffs)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad_c)
            if grad_norm > 1e-10:
                grad_c = grad_c / grad_norm
            
            new_coeffs = current_coeffs - eta * grad_c
            
            # Check if new heights are non-negative
            new_h = coeffs_to_h(new_coeffs)
            if np.any(new_h < -1e-6):
                eta *= 0.5
                if eta < 1e-8:
                    break
                continue
            
            current_coeffs = new_coeffs
        
        # Phase 2: Remez-like refinement
        if time_left() > 10:
            for k in range(100):
                if time_left() < 5:
                    break
                
                h_curr = coeffs_to_h(current_coeffs)
                h_curr = np.maximum(h_curr, 1e-10)
                h_curr = h_curr / np.sum(h_curr)
                curr_val = compute_autoconv_max(h_curr)
                
                if curr_val < best_val:
                    best_val = curr_val
                    best_h = h_curr.copy()
                
                grad_c, m_val = remez_step(current_coeffs)
                
                # Normalize gradient
                grad_norm = np.linalg.norm(grad_c)
                if grad_norm < 1e-10:
                    break
                
                grad_c = grad_c / grad_norm
                
                new_coeffs = current_coeffs - 0.01 * grad_c
                
                new_h = coeffs_to_h(new_coeffs)
                if np.any(new_h < -1e-6):
                    break
                
                current_coeffs = new_coeffs
    
    # Stage 3: Iterative Proportional Fitting (IPF) to minimize L1 norm of autoconvolution
    # This leverages the fact that reducing overall spread of convolution mass lowers peak
    # more effectively for near-uniform distributions
    
    if time_left() > 20:
        h_curr = best_h.copy()
        
        def ipf_step(h, beta=50.0):
            """
            Perform one IPF-like step:
            1. Compute autoconvolution
            2. Compute "target" distribution proportional to exp(-beta * conv)
            3. Update h to be proportional to sqrt(target * h) (geometric mean)
            This reduces peaks in conv while maintaining sum=1 and non-negativity
            """
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            
            # Target: low where conv is high
            m = np.max(conv)
            target_full = np.exp(-(conv - m) * beta)
            target = target_full[:N]
            
            # Geometric mean update: h_new ~ sqrt(h * target)
            # This is a multiplicative update that balances current h with target
            h_new = np.sqrt(h * target + 1e-30)
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new / s
            else:
                h_new = h.copy()
            
            return h_new, np.max(conv)
        
        # Run IPF iterations
        for k in range(200):
            if time_left() < 5:
                break
            
            h_curr, conv_max = ipf_step(h_curr)
            
            # Evaluate
            val = compute_autoconv_max(h_curr)
            if val < best_val:
                best_val = val
                best_h = h_curr.copy()
            
            # Early stopping if converged
            if k > 10 and abs(conv_max - best_val) < 1e-6:
                break
    
    # Stage 4: Final refinement with direct gradient descent
    if time_left() > 10:
        def smooth_max_and_grad_h(h, beta=100.0):
            """
            Compute smooth max of autoconvolution and its gradient w.r.t h.
            h is assumed to be non-negative and sum to 1.
            Returns: (smooth_max_value, grad_h)
            """
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            
            m = np.max(conv)
            exp_conv = np.exp((conv - m) * beta)
            sum_exp = np.sum(exp_conv)
            s_max = m + np.log(sum_exp) / beta
            
            weights = exp_conv / sum_exp
            
            # Gradient of smooth max w.r.t h
            W = np.fft.fft(weights, n=size)
            corr = np.fft.ifft(W * np.conj(H)).real
            grad_h = 2.0 * corr[:N]
            
            return s_max, grad_h
        
        current_h = best_h.copy()
        current_val = best_val
        eta = 0.01
        
        for k in range(100):
            if time_left() < 5:
                break
                
            s_max, grad_h = smooth_max_and_grad_h(current_h, 100.0)
            
            # Project gradient to maintain sum=1 and non-negativity
            new_h = current_h - eta * grad_h
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s > 0:
                new_h /= s
            else:
                break
                
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.9
                if eta < 1e-6:
                    break
    
    # Final fallback
    if best_h is None:
        best_h = np.ones(N) / N
    
    return best_h.tolist()
