import numpy as np
import time

N = 600
DEADLINE = 115  # seconds, keep safety margin before 150s kill

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
    
    def smooth_max(h, beta=50.0):
        """Smooth approximation of max using log-sum-exp."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        m = np.max(conv)
        if m < -1e10:
            return 0.0
        s = np.sum(np.exp((conv - m) * beta))
        return m + np.log(s) / beta
    
    def objective_and_grad(params, beta):
        """Objective and gradient for optimizer. params in log-space."""
        h = np.exp(params)
        s = np.sum(h)
        h_norm = h / s
        
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h_norm
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        
        obj = smooth_max(h_norm, beta)
        
        m = np.max(conv)
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)
        
        w_full = np.zeros(size)
        w_full[:len(conv)] = weights
        W = np.fft.fft(w_full, n=size)
        grad_h_pad = 2.0 * np.fft.ifft(H * W).real[:N]
        
        dot_g = np.dot(grad_h_pad, h_norm)
        grad_params = grad_h_pad * h_norm - h_norm * dot_g
        
        return obj, grad_params
    
    # Initializations
    inits = []
    i = np.arange(N)
    center = N / 2.0
    
    # Uniform
    inits.append(np.ones(N))
    
    # Raised cosine
    rc = 0.5 * (1 + np.cos(2 * np.pi * (i - center) / N))
    rc = np.maximum(rc, 0.01)
    inits.append(rc)
    
    # Gaussians
    for sigma in [N/6, N/5, N/4, N/3]:
        gauss = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
        inits.append(gauss)
    
    # U-shaped
    u_shape = 1.0 + 3.0 * (1 - np.sin(np.pi * i / N))
    inits.append(u_shape)
    
    # Triangular
    tri = 1 - np.abs(i - center) / (N/2)
    tri = np.maximum(tri, 0.01)
    inits.append(tri)
    
    # W-shape
    w_shape = 1.0 + 2.0 * np.cos(4 * np.pi * i / N)
    w_shape = np.maximum(w_shape, 0.01)
    inits.append(w_shape)
    
    best_h = None
    best_val = np.inf
    
    for init in inits:
        if time_left() < 15:
            break
        
        init = np.maximum(init, 1e-10)
        init = init / np.sum(init)
        params_init = np.log(init)
        
        for beta in [30.0, 50.0, 100.0]:
            if time_left() < 8:
                break
            
            try:
                def obj_wrapper(p):
                    return objective_and_grad(p, beta)[0]
                
                def grad_wrapper(p):
                    return objective_and_grad(p, beta)[1]
                
                from scipy.optimize import minimize
                res = minimize(
                    obj_wrapper,
                    params_init,
                    jac=grad_wrapper,
                    method='L-BFGS-B',
                    options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 80}
                )
                
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                val = compute_autoconv_max(h_opt)
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
                    
            except Exception:
                continue
    
    # Fallback
    if best_h is None:
        for init in inits:
            h = np.maximum(init, 1e-10)
            h = h / np.sum(h)
            val = compute_autoconv_max(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N) / N
    
    # Gradient descent refinement
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val
        eta = 0.03
        
        for k in range(80):
            if time_left() < 5:
                break
            
            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = current_h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real
            
            c_val = np.max(conv)
            idx_max = np.argmax(conv)
            
            idx = np.arange(N)
            pos1 = idx_max - idx
            pos2 = idx - idx_max
            
            grad = np.zeros(N)
            mask1 = (pos1 >= 0) & (pos1 < N)
            if np.any(mask1):
                grad[mask1] += current_h[pos1[mask1]]
            mask2 = (pos2 >= 0) & (pos2 < N)
            if np.any(mask2):
                grad[mask2] += current_h[pos2[mask2]]
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            new_h = current_h - eta * grad
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s > 0:
                new_h /= s
            
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.8
                if eta < 1e-6:
                    break
    
    return best_h.tolist()
