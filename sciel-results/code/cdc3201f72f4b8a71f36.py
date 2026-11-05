import numpy as np
import time
from scipy.fft import dct, idct, fft, ifft
from scipy.optimize import minimize

N = 400

def construct():
    deadline = time.time() + 145.0
    
    # Target sum
    target_sum = 1.0
    
    # Precompute DCT basis? No, use scipy functions.
    
    def get_h(x):
        # x is length N, x[0] is fixed to ensure sum=1
        # We pass x[1:] as variables
        x_full = np.zeros(N)
        x_full[0] = 1.0 / np.sqrt(N) # For norm='ortho', this gives mean 1/N, sum 1
        x_full[1:] = x
        h = idct(x_full, type=2, norm='ortho')
        return h

    def objective_and_grad(x, beta):
        h = get_h(x)
        S = np.sum(h)
        # Compute auto-correlation
        r = np.convolve(h, h)
        
        # LogSumExp for max
        # Shift r to avoid overflow
        r_max = np.max(r)
        exp_r = np.exp(beta * (r - r_max))
        sum_exp = np.sum(exp_r)
        lse = r_max + np.log(sum_exp) / beta
        
        # Penalty for sum deviation (should be 0 if DCT is correct, but numerical errors?)
        # Actually, with ortho norm, the sum is exactly 1 if x[0] is fixed.
        # So we don't need penalty.
        
        # Gradient calculation
        # f = lse
        # df/dr = exp(beta * (r - r_max)) / sum_exp
        w = exp_r / sum_exp
        
        # Gradient of r w.r.t h
        # r_k = sum_j h_j h_{k-j}
        # dr_k/dh_m = h_{k-m} + h_{m-k} (with 0 padding)
        # Sum_k w_k dr_k/dh_m = 2 * convolve(h, w_padded)[m] ?
        # Let's verify.
        # r = h * h
        # dr/dh is the derivative of the convolution.
        # It is equivalent to convolving h with the gradient of the loss w.r.t r.
        # grad_h = 2 * np.convolve(h, w, mode='full') ? 
        # The length of r is 2N-1.
        # np.convolve(h, w, 'full') has length 2N-1 + N - 1 = 3N-2? No.
        # If h is N, w is 2N-1.
        # We need the gradient w.r.t h (length N).
        # The operation is:
        # grad_h[m] = sum_k w_k (h_{k-m} + h_{m-k})
        # This is the convolution of h with w, but w is symmetric? No, w is not symmetric.
        # However, r is symmetric if h is real? No, r is symmetric around N-1.
        # w is not symmetric.
        
        # Let's use the property:
        # L = sum_k w_k r_k
        # L = sum_k w_k sum_j h_j h_{k-j}
        # Let u = w.
        # L = sum_j h_j (sum_k w_k h_{k-j})
        # The term in parenthesis is (w * h)_j ?
        # (w * h)_j = sum_k w_k h_{j-k}.
        # Indices:
        # r_k = sum_{j=0}^{N-1} h_j h_{k-j} (with h_i=0 for i<0 or i>=N)
        # L = sum_k w_k sum_j h_j h_{k-j}
        # Swap sums: sum_j h_j sum_k w_k h_{k-j}
        # Let l = k-j => k = l+j.
        # sum_k w_k h_{k-j} = sum_l w_{l+j} h_l.
        # This is the cross-correlation of w and h?
        # Actually, it's simpler to just compute the gradient via FFT or direct matrix multiplication if N is small.
        # N=400. 399 variables.
        # Finite difference might be okay for a few steps, but L-BFGS-B needs many steps.
        
        # Let's compute grad_h analytically.
        # grad_h[m] = 2 * sum_{k} w_k h_{k-m} (with appropriate bounds)
        # This is the convolution of h with w, but reversed?
        # Let's use np.convolve.
        # grad_h = 2 * np.convolve(h, w, mode='same')?
        # w has length 2N-1. h has length N.
        # np.convolve(h, w, 'same') returns length max(N, 2N-1) = 2N-1?
        # We need length N.
        
        # Correct approach:
        # grad_h = 2 * ifft(fft(h, 2*N) * fft(w, 2*N))[:N] ?
        # w is defined on lags 0..2N-2.
        # Let's zero pad w to 2N.
        w_padded = np.zeros(2*N)
        w_padded[:len(w)] = w
        H_fft = fft(h, 2*N)
        W_fft = fft(w_padded, 2*N)
        grad_h_fft = 2 * H_fft * W_fft
        grad_h = ifft(grad_h_fft, 2*N).real[:N]
        
        # Transform grad_h to DCT domain
        grad_x = dct(grad_h, type=2, norm='ortho')
        grad_x[0] = 0 # x[0] is fixed
        
        return lse, grad_x[1:]

    best_h = np.ones(N) / N
    best_val = 2.0 # Seed value is 2.0 (scaled by sum^2=1, 2*N*max(1/N^2)/1 = 2/N * N^2? No.)
    # For h=1/N, sum=1. r_max = 1/N. V = 2*N*(1/N)/1 = 2.
    
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        x0 = np.random.randn(N-1) * 0.1
        
        # Anneal beta
        betas = [5, 10, 20, 40]
        
        x_current = x0
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            
            # Minimize with current beta
            res = minimize(
                objective_and_grad, 
                x_current, 
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            x_current = res.x
            
            # Evaluate true value
            h_curr = get_h(x_current)
            r_curr = np.convolve(h_curr, h_curr)
            S_curr = np.sum(h_curr)
            val_curr = abs(2 * N * np.max(r_curr) / (S_curr ** 2))
            
            if val_curr < best_val:
                best_val = val_curr
                best_h = h_curr.copy()
                
    return best_h
