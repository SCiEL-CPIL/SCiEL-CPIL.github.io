import numpy as np
from scipy.optimize import minimize
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Uniform
    # Using log-space for positivity and sum-to-1 constraint
    # x = softmax(y)
    y = np.zeros(N)
    
    # Soft-max temperature for the objective (max of sidelobes)
    tau_obj = 0.1
    
    # Learning rate
    lr = 0.01
    momentum = 0.9
    
    v = np.zeros(N)
    
    best_score = np.inf
    best_y = y.copy()
    
    # Precompute FFT sizes
    fft_size = 2 * N
    
    def compute_circ_conv(x):
        # Circular convolution via FFT
        X = np.fft.fft(x, n=fft_size)
        C = np.fft.ifft(X * X).real
        # We need linear convolution, but for sidelobes k < N, 
        # circular conv matches linear conv if we handle the wrap around correctly.
        # Actually, for k in 1..N-1, c_lin[k] = c_circ[k] - c_circ[k+N]?
        # No, c_circ[k] = sum x_i x_{i+k mod N}.
        # c_lin[k] = sum x_i x_{i+k} (where x_j=0 for j<0 or j>=N).
        # For k < N, c_lin[k] = sum_{i=0}^{N-1-k} x_i x_{i+k}.
        # c_circ[k] = sum_{i=0}^{N-1} x_i x_{i+k mod N}
        #            = sum_{i=0}^{N-1-k} x_i x_{i+k} + sum_{i=N-k}^{N-1} x_i x_{i+k-N}
        #            = c_lin[k] + c_lin[N-k]
        # So c_lin[k] = c_circ[k] - c_lin[N-k].
        # This coupling makes gradient descent tricky if we use circular directly.
        
        # Better: Compute Linear Convolution directly using FFT with zero padding.
        X = np.fft.fft(x, n=fft_size)
        C_lin = np.fft.ifft(X * X).real
        # C_lin has length fft_size. The linear conv is in C_lin[0:N] and C_lin[N:2N-1] is mirrored?
        # np.fft.ifft of X*X gives circular conv of size fft_size.
        # Since fft_size >= 2N-1, it contains the linear conv.
        # c_lin[k] is at index k.
        return C_lin

    def objective_and_grad(y):
        x = np.exp(y - np.max(y))
        x_sum = np.sum(x)
        x = x / x_sum
        
        c = compute_circ_conv(x)
        # Sidelobes are indices 1 to N-1
        sidelobes = c[1:N]
        
        # Soft max of sidelobes
        max_sidelobe = np.max(sidelobes)
        exp_sidelobes = np.exp((sidelobes - max_sidelobe) / tau_obj)
        sum_exp = np.sum(exp_sidelobes)
        obj = tau_obj * np.log(sum_exp) + max_sidelobe
        
        # Gradient of soft max w.r.t sidelobes
        p = exp_sidelobes / sum_exp
        
        # Gradient w.r.t x
        # d obj / d x = 2 * (p * x)  (for linear conv, correlation with p)
        # p is defined on lags 1..N-1. We need to extend p to full length for convolution.
        # p_full[0] = 0, p_full[1:N] = p, p_full[N:] = 0 (and symmetric part?)
        # The gradient of sum c_k p_k w.r.t x_i is 2 sum_k p_k (x_{k-i} + x_{i-k})
        # This is 2 * (p_full * x) where p_full is symmetric?
        # Let's construct p_full of size fft_size.
        p_full = np.zeros(fft_size)
        p_full[1:N] = p
        p_full[fft_size-(N-1):fft_size] = p[::-1] # Symmetric part for linear conv gradient?
        # Actually, the gradient of linear correlation is 2 * corr(x, p_full)
        
        # Using FFT for gradient
        P_full = np.fft.fft(p_full, n=fft_size)
        X = np.fft.fft(x, n=fft_size)
        grad_x = 2 * np.real(np.fft.ifft(P_full * X))
        
        # Gradient w.r.t y
        # x = softmax(y)
        # grad_y = diag(x) grad_x - x (x^T grad_x)
        grad_y = x * grad_x - x * np.dot(x, grad_x)
        
        return obj, grad_y

    # Optimization loop
    for i in range(10000):
        if time.time() > deadline:
            break
        
        obj, grad_y = objective_and_grad(y)
        
        # Update momentum
        v = momentum * v + (1 - momentum) * grad_y
        y -= lr * v
        
        # Track best
        # We need to evaluate the TRUE max sidelobe for the score
        x = np.exp(y - np.max(y))
        x = x / np.sum(x)
        c = compute_circ_conv(x)
        true_max_sidelobe = np.max(c[1:N])
        
        if true_max_sidelobe < best_score:
            best_score = true_max_sidelobe
            best_y = y.copy()
            
        # Anneal tau_obj
        if i % 100 == 0 and tau_obj > 0.01:
            tau_obj *= 0.9

    # Final polish with best_y
    x = np.exp(best_y - np.max(best_y))
    x = x / np.sum(x)
    
    return list(x)

