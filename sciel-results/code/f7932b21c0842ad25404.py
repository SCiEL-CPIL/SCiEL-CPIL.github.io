"""Optimize signed step heights via projected gradient descent on soft-max autoconv peak."""
import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x."""
    n = len(x)
    # Use FFT for speed
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    return ac

def _soft_max_abs(ac, temperature):
    """Smooth approximation of max |ac| using log-sum-exp."""
    abs_ac = np.abs(ac)
    if temperature <= 0:
        return np.max(abs_ac)
    # log-sum-exp trick for numerical stability
    m = np.max(abs_ac)
    if m == 0:
        return 0.0
    return m + temperature * np.log(np.sum(np.exp((abs_ac - m) / temperature)))

def _soft_max_abs_grad(ac, temperature):
    """Gradient of soft-max |ac| with respect to ac."""
    abs_ac = np.abs(ac)
    if temperature <= 0:
        g = np.zeros_like(ac)
        idx = np.argmax(abs_ac)
        g[idx] = np.sign(ac[idx]) if ac[idx] != 0 else 1.0
        return g
    m = np.max(abs_ac)
    if m == 0:
        return np.zeros_like(ac)
    weights = np.exp((abs_ac - m) / temperature)
    weights /= np.sum(weights)
    # d/dac of max|ac| approx: weights * sign(ac)
    signs = np.sign(ac)
    signs[signs == 0] = 1.0
    return weights * signs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initialize with small random perturbation around 1
    rng = np.random.default_rng(42)
    x = np.ones(N)
    
    # Track best
    best_x = x.copy()
    best_val = float('inf')
    
    # Compute initial objective
    ac = _autoconv(x)
    init_val = np.max(np.abs(ac)) / (np.sum(x))**2
    best_val = init_val
    
    # Adaptive temperature schedule
    temp_start = 5.0
    temp_end = 0.1
    n_iter = 3000
    
    # Learning rate schedule
    lr_start = 0.01
    lr_end = 0.001
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Current temperature
        frac = it / max(n_iter - 1, 1)
        temp = temp_start * (temp_end / temp_start) ** frac
        lr = lr_start * (lr_end / lr_start) ** frac
        
        # Compute autoconvolution
        ac = _autoconv(x)
        
        # Soft-max objective and its gradient w.r.t. ac
        obj = _soft_max_abs(ac, temp)
        grad_ac = _soft_max_abs_grad(ac, temp)
        
        # Gradient of autoconvolution w.r.t. x
        # ac[k] = sum_i x[i] * x[i+k]
        # dac[k]/dx[j] = x[j+k] + x[j-k] (with boundary handling)
        # More precisely: d(sum_i x[i]*x[i+k])/dx[j] = x[j+k] + x[j-k] if both in range
        # We can compute this via: grad_x = ifft(conj(fft(grad_ac)) * ... ) 
        # Actually: ac = ifft(|fft(x)|^2), so grad_x[j] = ifft(fft(grad_ac) * 2*conj(fft(x)))... 
        # Let's use the direct convolution structure:
        # dac/dx[j] = sum_k grad_ac[k] * (x[j+k] + x[j-k]) where indices are valid
        # This is equivalent to: grad_x = 2 * (grad_ac * x) (circular conv) for full conv
        # For linear convolution, we need to be careful. Let's use FFT approach:
        # ac = ifft(fft(x, 2N) * conj(fft(x, 2N)))[:N]
        # d(ac)/dx[j] can be computed as: ifft(2 * fft(x, 2N) * conj(fft(grad_ac_padded, 2N)))
        # where grad_ac is padded to length 2N
        
        # Pad grad_ac to length 2N
        ga_padded = np.zeros(2 * N)
        ga_padded[:N] = grad_ac
        
        fx = np.fft.rfft(x, n=2 * N)
        fga = np.fft.rfft(ga_padded, n=2 * N)
        
        # Gradient of ac w.r.t. x: 
        # ac[k] = sum_i x[i]*x[i+k], so dac/dx[j] involves cross terms
        # Using FFT: ac = ifft(|Fx|^2) where Fx = fft(x, 2N)
        # d(ac)/dx = ifft(2 * Fx * conj(Fga)) ... let me think more carefully
        # Actually, the gradient of the scalar obj w.r.t. x[j] is:
        # sum_k grad_ac[k] * d(ac[k])/dx[j]
        # d(ac[k])/dx[j] = x[j+k] + x[j-k] (for linear conv, with zero padding)
        # This equals 2 * (x * grad_ac)[j] in circular sense if we handle boundaries
        # For linear convolution with zero padding:
        # grad_x[j] = sum_k grad_ac[k] * (x[j+k] + x[j-k])
        # = sum_k grad_ac[k]*x[j+k] + sum_k grad_ac[k]*x[j-k]
        # = (grad_ac * x)[j] + (grad_ac_flipped * x)[j]  ... this is getting complex
        # 
        # Simpler: use the fact that for ac = x*x (linear), 
        # grad_x = 2 * (x * grad_ac) but only for the valid part
        # Let's just compute it directly for clarity, or use the FFT formula:
        # 
        # The correct formula: if ac[k] = sum_i x[i] x[i+k], then
        # d(obj)/dx[j] = sum_k grad_ac[k] (x[j+k] + x[j-k])
        # This is the same as: 2 * ifft(fft(x, 2N) * conj(fft(grad_ac, 2N)))[:N] 
        # Wait, let me verify: 
        # ifft(fft(x) * conj(fft(ga)))[j] = sum_k x[k] ga[j-k] = (x * ga_flipped)[j]
        # That's not quite right either.
        
        # Let me just do it with a simple approach: 
        # grad_x[j] = sum_k grad_ac[k] * (x[j+k] if j+k < N else 0 + x[j-k] if j-k >= 0 else 0)
        # This is O(N^2) which is fine for N=400
        
        grad_x = np.zeros(N)
        for k in range(N):
            ga_k = grad_ac[k]
            if ga_k == 0:
                continue
            # x[j+k] term: for j in [0, N-1-k], x[j+k] contributes
            # x[j-k] term: for j in [k, N-1], x[j-k] contributes
            if k == 0:
                grad_x += 2 * ga_k * x
            else:
                # x[j+k] for j=0..N-1-k
                grad_x[:N-k] += ga_k * x[k:]
                # x[j-k] for j=k..N-1
                grad_x[k:] += ga_k * x[:N-k]
        
        # Project gradient onto hyperplane sum(x) = N
        # Subtract mean of gradient to ensure step preserves sum
        grad_x -= np.mean(grad_x)
        
        # Take gradient step
        x_new = x - lr * grad_x
        
        # Project back to sum(x) = N
        x_new += (N - np.sum(x_new)) / N
        
        # Update
        x = x_new
        
        # Evaluate actual objective (not soft-max) for tracking
        ac_check = _autoconv(x)
        val = np.max(np.abs(ac_check)) / (np.sum(x))**2
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Final: try a few more iterations with very low temperature for polish
    if time.time() < deadline - 5:
        for it in range(500):
            if time.time() > deadline - 2:
                break
            ac = _autoconv(x)
            obj = np.max(np.abs(ac))
            idx_max = np.argmax(np.abs(ac))
            # Exact gradient at the max
            grad_ac = np.zeros(N)
            grad_ac[idx_max] = np.sign(ac[idx_max]) if ac[idx_max] != 0 else 1.0
            
            grad_x = np.zeros(N)
            k = idx_max
            if k == 0:
                grad_x += 2 * grad_ac[k] * x
            else:
                grad_x[:N-k] += grad_ac[k] * x[k:]
                grad_x[k:] += grad_ac[k] * x[:N-k]
            
            grad_x -= np.mean(grad_x)
            
            lr = 0.0005
            x = x - lr * grad_x
            x += (N - np.sum(x)) / N
            
            ac_check = _autoconv(x)
            val = np.max(np.abs(ac_check)) / (np.sum(x))**2
            if val < best_val:
                best_val = val
                best_x = x.copy()
    
    return best_x
