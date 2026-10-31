import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import convolve1d

N = 600
DEADLINE = 145  # safety margin before 150s hard limit

def construct():
    start_time = time.time()
    
    # The metric is J(h) = max_k (h*h)_k / (sum h)^2
    # We fix sum(h) = 1, so J(h) = max_k (h*h)_k
    # We use softmax parameterization: h = softmax(x)
    # Objective: Log-Sum-Exp of convolution to smooth the max
    
    def objective(x):
        # Compute h from x using softmax
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        h /= np.sum(h)
        
        # Compute autoconvolution
        # h*h is full convolution, length 2N-1
        c = np.convolve(h, h, mode='full')
        
        # Log-Sum-Exp approximation of max
        # Temperature T controls smoothness. Higher T -> closer to max.
        # We can anneal T, but a fixed high T like 50-100 is often sufficient.
        T = 50.0
        c_scaled = T * c
        # Avoid overflow in exp
        c_max = np.max(c_scaled)
        lse = c_max + np.log(np.sum(np.exp(c_scaled - c_max)))
        return lse / T
        
    def gradient(x):
        # Gradient of LSE w.r.t x
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        h /= np.sum(h)
        
        c = np.convolve(h, h, mode='full')
        
        T = 50.0
        c_scaled = T * c
        c_max = np.max(c_scaled)
        exp_c = np.exp(c_scaled - c_max)
        sum_exp = np.sum(exp_c)
        
        # Gradient of LSE w.r.t c is softmax(T*c)
        grad_c = exp_c / sum_exp
        
        # Gradient of c w.r.t h
        # c_k = sum_i h_i h_{k-i}
        # d c_k / d h_j = h_{k-j} + h_{k-j+N} (with wraparound or zero padding logic)
        # Actually, c = h * h (full).
        # d/dc_k (h_j) = h_{k-j} if 0 <= k-j < N, else 0.
        # More efficiently, grad_h = c * h + c * h (due to symmetry and chain rule)
        # Specifically, grad_h_j = sum_k (d c_k / d h_j) * grad_c_k
        # d c_k / d h_j = h_{k-j} (if index valid in full conv context)
        # This is equivalent to grad_h = convolve(grad_c, h, 'valid' + padding?)
        # Let's use the property: grad_h = c * h (in the sense of cross-correlation)
        # Actually, for c = h*h, grad_h = c * h (cross-correlation)
        # Let's compute it directly for correctness if N is small, but N=600 is ok for FFT or direct if efficient.
        # Direct:
        # grad_h[j] = sum_{k} grad_c[k] * h[k-j]
        # This is convolution of grad_c and h.
        # Since c is full length 2N-1, and h is N.
        # Let's use np.convolve for the gradient step too, it's fast enough.
        
        # The gradient of the convolution c_k = sum_i h_i h_{k-i} w.r.t h_j is:
        # sum_k grad_c[k] * (h_{k-j} + h_{k-j}) ... wait.
        # c_k = sum_{i=0}^{N-1} h_i h_{k-i}
        # d c_k / d h_j = h_{k-j} + h_{k-j} ? No.
        # If j is in the first sum: term is h_j h_{k-j}. Deriv is h_{k-j}.
        # If j is in the second sum: term is h_i h_{j} where i = k-j. Deriv is h_i = h_{k-j}.
        # So d c_k / d h_j = 2 * h_{k-j} (if k-j is valid index for h).
        # So grad_h_j = 2 * sum_k grad_c[k] h_{k-j}
        # This is 2 * cross_correlation(grad_c, h)
        
        # Let's compute cross_correlation
        # grad_c is length 2N-1. h is length N.
        # We need sum_k grad_c[k] h_{k-j} for j=0..N-1.
        # This is equivalent to np.convolve(grad_c, h, 'valid')? No.
        # Let's just use the FFT or a simple loop if it's not too slow, or np.convolve trick.
        # np.convolve(grad_c, h, 'full') will give the correlation?
        # Actually, simpler: grad_h = np.convolve(grad_c, h, 'valid')?
        # Let's verify dimensions.
        # grad_c: 2N-1. h: N.
        # We want grad_h of size N.
        # grad_h[j] = sum_{k=j}^{j+N-1} grad_c[k] h[k-j] ? No.
        # The indices k in c range from 0 to 2N-2.
        # h index i ranges 0 to N-1.
        # c_k = sum_i h_i h_{k-i}.
        # For a fixed j, d c_k / d h_j is non-zero when j is an index in h.
        # Term 1: h_j * h_{k-j}. Exists if 0 <= k-j < N. Deriv: h_{k-j}.
        # Term 2: h_i * h_j where i = k-j. Exists if 0 <= k-j < N. Deriv: h_{k-j}.
        # So d c_k / d h_j = 2 * h_{k-j} if 0 <= k-j < N, else 0.
        # grad_h_j = 2 * sum_k grad_c[k] h_{k-j} * I(0 <= k-j < N)
        # Let m = k-j. Then k = m+j.
        # grad_h_j = 2 * sum_m grad_c[m+j] h[m] * I(0 <= m < N)
        # This is the cross-correlation of grad_c and h, shifted.
        # Specifically, it's the "valid" part of the convolution of grad_c and reversed h?
        # Let's just use np.correlate or manual FFT if needed.
        # For N=600, a simple O(N^2) is 360,000 ops, which is very fast in Python/NumPy vectorized.
        
        # Vectorized computation of grad_h
        # grad_h[j] = 2 * sum_{m=0}^{N-1} h[m] * grad_c[m+j]
        # This is a sliding dot product.
        # We can use stride tricks or just a loop if it's fast enough.
        # Or use FFT.
        
        # Let's use FFT for speed and robustness.
        # grad_h = 2 * ifft(fft(grad_c, 2*N) * conj(fft(h, 2*N)))[:N]
        
        size = 2 * N
        F_gc = np.fft.rfft(grad_c, size)
        F_h = np.fft.rfft(h, size)
        grad_h_fft = F_gc * np.conj(F_h)
        grad_h = np.fft.irfft(grad_h_fft, size)[:N] * 2.0
        
        # Now gradient w.r.t x
        # h_i = exp(x_i) / S, S = sum exp(x_j)
        # d h_i / d x_k = h_i (delta_{ik} - h_k)
        # grad_x_k = sum_i grad_h_i * d h_i / d x_k
        # = sum_i grad_h_i h_i (delta_{ik} - h_k)
        # = grad_h_k h_k - h_k sum_i grad_h_i h_i
        # = h_k (grad_h_k - dot(grad_h, h))
        
        dot_grad_h_h = np.dot(grad_h, h)
        grad_x = h * (grad_h - dot_grad_h_h)
        
        return grad_x

    # Initial guess: Gaussian
    x_grid = np.linspace(-3, 3, N)
    h_init = np.exp(-x_grid**2)
    h_init /= np.sum(h_init)
    # Invert softmax to get x_init
    # h_i = exp(x_i) / sum(exp(x_j))
    # log(h_i) = x_i - log(sum(exp(x_j)))
    # x_i = log(h_i) + C. We can set C=0 for simplicity, or just use log(h_init) as x_init.
    # Since softmax is invariant to shifts, x_init = log(h_init) is a valid starting point.
    x0 = np.log(h_init + 1e-10)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Optimize
    # Use L-BFGS-B
    # maxiter and maxfun to control time
    # We check time in a callback
    
    state = {'elapsed': 0}
    
    def callback(xk):
        state['elapsed'] = time.time() - start_time
        val = objective(xk)
        if val < best_val:
            nonlocal best_val, best_x
            best_val = val
            best_x = xk.copy()
        return state['elapsed'] > DEADLINE - 5 # Return True to stop if deadline near

    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                       options={'maxiter': 1000, 'maxfun': 1000, 'ftol': 1e-10, 'gtol': 1e-8},
                       callback=callback)
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception as e:
        pass # Keep best_x

    # Final check and maybe a few local restarts if time permits
    if time.time() - start_time < DEADLINE - 10:
        # Try a few random restarts
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Perturb x0
            x_try = x0 + 0.1 * np.random.randn(N)
            val_try = objective(x_try)
            if val_try < best_val:
                # Optimize from this point
                try:
                    res2 = minimize(objective, x_try, method='L-BFGS-B', jac=gradient, 
                                   options={'maxiter': 200, 'maxfun': 200},
                                   callback=callback)
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_x = res2.x
                except:
                    pass

    # Convert best_x to h
    x_shifted = best_x - np.max(best_x)
    h_best = np.exp(x_shifted)
    h_best /= np.sum(h_best)
    
    return h_best.tolist()
