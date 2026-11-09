import numpy as np
import time
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START

def construct():
    """
    Minimizes the normalized maximum autoconvolution using gradient-based optimization
    on a smooth surrogate objective.
    
    The objective is approximately max(f*f) / (sum(f))^2.
    Since scaling f by c scales max(f*f) by c^2 and (sum(f))^2 by c^2, the ratio is scale-invariant.
    To avoid division by zero and ensure numerical stability, we fix the L1 norm (sum of heights) to 1.
    Then we minimize max(f*f).
    
    We use a soft-max approximation for the maximum to make the objective differentiable:
    soft_max(z, T) = T * log(sum(exp(z_i / T)))
    As T -> 0, this approaches max(z_i).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Compute the full autoconvolution kernel (linear convolution)
    # f has length N. f*f has length 2N-1.
    # We want the maximum value of this convolution.
    # For efficiency, we can use FFT.
    
    # Initialize with a Gaussian bump centered in the domain
    x = np.linspace(0, N - 1, N)
    center = (N - 1) / 2.0
    sigma = N / 10.0  # Initial width
    h0 = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    h0 /= h0.sum()  # Normalize L1 norm to 1
    
    def objective_and_grad(h):
        """
        Computes the soft-max of the autoconvolution and its gradient.
        h is assumed to be non-negative and L1-normalized (or close to it).
        """
        # Ensure non-negativity for the calculation, though optimizer handles bounds
        h_pos = np.maximum(h, 0)
        
        # FFT for autoconvolution
        # Length of linear conv is 2N-1. FFT size should be next power of 2 for efficiency.
        fft_len = 1 << (2 * N - 1).bit_length()
        
        H = np.fft.rfft(h_pos, fft_len)
        H2 = H * H
        conv = np.fft.irfft(H2, fft_len)[:2 * N - 1]
        
        # Soft-max temperature
        # Start with a reasonable temperature, maybe anneal if needed, 
        # but for simplicity, let's pick a T that balances smoothness and accuracy.
        # The values in conv are roughly O(1/N) if sum(h)=1.
        # Let's inspect the scale.
        # If h ~ 1/N, conv ~ 1/N.
        # Let's set T relative to the max value.
        max_val = np.max(conv)
        if max_val <= 0:
            return 0.0, np.zeros_like(h)
        
        # Temperature T. Smaller T is more accurate but steeper gradients.
        # Let's try T = 0.05 * max_val initially? Or a fixed small value.
        # Since we are minimizing, we want the gradient to point away from the peak.
        # Let's use T = 1.0e-3 * max_val? 
        # Actually, let's just use a fixed small T relative to the signal.
        # If max_val is ~1e-3, T=1e-5 might be good.
        
        T = 1e-4 * max_val + 1e-10
        
        # Soft max: T * log(sum(exp(conv / T)))
        # To avoid overflow, subtract max
        shifted = conv / T
        shifted -= np.max(shifted)
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        soft_max_val = T * log_sum_exp
        
        # Gradient of soft-max w.r.t conv
        # d(soft_max)/d(conv_i) = exp(conv_i / T) / sum(exp(conv_j / T))
        # Note: shifted = conv/T - max(conv/T)
        # exp(shifted) = exp(conv/T - max) = exp(conv/T) / exp(max)
        # sum(exp(shifted)) = sum(exp(conv/T)) / exp(max)
        # ratio = exp(shifted_i) / sum(exp(shifted))
        weights = np.exp(shifted)
        weights /= weights.sum()
        
        # Gradient w.r.t h
        # conv = h * h
        # d(conv_k)/d(h_i) = h_{k-i} + h_{k+i} (with appropriate bounds)
        # This is a linear operator. We can use FFT for the gradient step too.
        # grad_conv_k = weights_k
        # grad_h_i = sum_k weights_k * d(conv_k)/d(h_i)
        
        # Using FFT:
        # grad_h = ifft( fft(grad_conv) * H ) * 2 ? 
        # Let's derive:
        # conv = ifft( H^2 )
        # d(conv)/d(h) is a linear operator L.
        # <grad_h, h> = <weights, conv>
        # In frequency domain:
        # grad_H = 2 * H * weights_freq ? No.
        # Let's use the property that if y = x*x, grad_x = 2 * ifft(fft(y_grad) * X)
        # Here y = conv, y_grad = weights.
        # X = H.
        # So grad_h = 2 * ifft( fft(weights) * H )
        
        W = np.fft.rfft(weights, fft_len)
        grad_h = 2.0 * np.fft.irfft(W * H, fft_len)[:N]
        
        # The objective is soft_max. We want to minimize it.
        # So the gradient to return is grad_h.
        
        return soft_max_val, grad_h

    # Optimization
    # We need to handle the constraint sum(h)=1 and h>=0.
    # L-BFGS-B can handle bounds.
    # We can fix the sum=1 constraint by optimizing N-1 variables? 
    # Or just use a penalty? 
    # Or simpler: since the objective is scale invariant, we can just minimize max(f*f) 
    # with the constraint sum(h)=1.
    # Let's use L-BFGS-B with bounds [0, inf] and add a penalty for sum(h) != 1?
    # Actually, L-BFGS-B doesn't handle linear equality constraints well.
    # Alternative: Parameterize h such that sum(h)=1 automatically?
    # Or, since scale invariance holds, we can just minimize max(f*f) + lambda * (sum(h)-1)^2.
    # But the gradient of the main term is what matters.
    
    # Let's try a simpler approach: 
    # Minimize F(h) = soft_max(f*f) 
    # Subject to h >= 0.
    # The scale invariance means that if we find a local minimum, we can normalize it.
    # However, the gradient flow might drift in scale.
    # Let's add a strong penalty for deviating from sum(h)=1.
    
    def penalized_objective_and_grad(h):
        val, grad = objective_and_grad(h)
        s = np.sum(h)
        # Penalty for sum(h) != 1
        # P = 100 * (s - 1)^2
        # dP/dh = 200 * (s - 1) * 1
        penalty = 100.0 * (s - 1.0) ** 2
        grad_penalty = 200.0 * (s - 1.0) * np.ones_like(h)
        
        return val + penalty, grad + grad_penalty

    # Initial guess
    x0 = h0.copy()
    
    # Bounds: h_i >= 0
    bounds = [(0, None) for _ in range(N)]
    
    # Run optimization
    best_h = x0
    best_val = penalized_objective_and_grad(x0)[0]
    
    try:
        res = minimize(
            penalized_objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'disp': False, 'gtol': 1e-5}
        )
        
        if res.success or res.fun < best_val:
            best_h = res.x
            best_val = res.fun
            
            # Check time
            if time.time() < deadline:
                # Try a second pass with a different initialization or smaller T?
                # Let's try a slightly narrower Gaussian to see if we can escape local minima
                sigma2 = N / 20.0
                h0_2 = np.exp(-0.5 * ((x - center) / sigma2) ** 2)
                h0_2 /= h0_2.sum()
                
                res2 = minimize(
                    penalized_objective_and_grad,
                    h0_2,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 500, 'maxfun': 5000, 'disp': False, 'gtol': 1e-5}
                )
                
                if res2.fun < best_val:
                    best_h = res2.x
                    best_val = res2.fun
                    
    except Exception:
        pass

    # Final cleanup: ensure non-negative and normalize
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.ones(N) / N

    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
