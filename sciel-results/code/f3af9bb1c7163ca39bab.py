import numpy as np
import time
from scipy.fft import dct, idct

N = 400

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Initialize h
    # Try an antisymmetric initialization
    h = np.zeros(N)
    h[:N//2] = 1.0
    h[N//2:] = -1.0
    
    # Add some noise to break symmetry slightly and help optimization
    rng = np.random.default_rng(42)
    h += 0.1 * rng.standard_normal(N)
    
    # Normalize to sum 1
    h = (h - np.mean(h)) + 1.0/N
    
    best_h = h.copy()
    best_val = np.inf
    
    # Precompute DCT basis? No, let's just use time domain gradient descent.
    # It's simpler and N=400 is small enough.
    
    def compute_value(h):
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        sum_h = np.sum(h)
        if sum_h == 0:
            return np.inf
        val = abs(2 * N * max_conv / (sum_h ** 2))
        return val
    
    best_val = compute_value(best_h)
    
    # Gradient Descent
    # We want to minimize max(conv).
    # Use LSE surrogate.
    tau = 0.1
    lr = 0.01
    
    # Number of steps depends on time
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        conv = np.convolve(h, h)
        # LSE gradient
        # L = tau * log(sum(exp(conv/tau)))
        # dL/dconv_i = exp(conv_i/tau) / sum(exp(conv_j/tau))
        # We can compute this stably
        max_conv = np.max(conv)
        exp_conv = np.exp((conv - max_conv) / tau)
        sum_exp = np.sum(exp_conv)
        grad_conv = exp_conv / sum_exp
        
        # Gradient of conv w.r.t h
        # conv_k = sum_i h_i h_{k-i}
        # d conv_k / d h_j = h_{k-j} + h_{k-j} (if distinct) ... 
        # Actually, grad_h = convolve(grad_conv, h) + convolve(grad_conv, h)??
        # No.
        # conv = h * h.
        # d (h*h) / d h = 2 * (h * dh/dh)?
        # Let's use the property:
        # grad_h = convolve(grad_conv, h) + convolve(grad_conv[::-1], h)?
        # Simpler: 
        # grad_h_j = sum_k grad_conv_k * (d conv_k / d h_j)
        # d conv_k / d h_j = h_{k-j} (with bounds)
        # So grad_h = convolve(grad_conv, h) + convolve(grad_conv, h[::-1])?
        # Let's verify.
        # If conv = h*h, then grad_h = 2 * (h * grad_conv) is for circular?
        # For linear:
        # grad_h = np.convolve(grad_conv, h, mode='same') + np.convolve(grad_conv, h, mode='same')?
        # Actually, the gradient of the convolution $y = x*x$ with respect to $x$ is $2 x * \frac{\partial L}{\partial y}$.
        # Wait, $y_k = \sum_i x_i x_{k-i}$.
        # $\frac{\partial y_k}{\partial x_j} = x_{k-j} + x_{k-j}$? No.
        # If $i=j$, term is $x_j x_{k-j}$. Deriv is $x_{k-j}$.
        # If $k-i=j \implies i=k-j$, term is $x_{k-j} x_j$. Deriv is $x_j$.
        # So if $j \neq k-j$, we have two terms?
        # No, the sum is over $i$.
        # $\frac{\partial}{\partial x_j} \sum_i x_i x_{k-i}$.
        # Term $i=j$: $x_j x_{k-j} \to x_{k-j}$.
        # Term $i=k-j$: $x_{k-j} x_j \to x_j$.
        # So if $j \neq k-j$, sum is $x_{k-j} + x_j$.
        # If $j = k-j$ (i.e. $2j=k$), term is $x_j^2 \to 2x_j$.
        # This is exactly $2 (x * g)_j$ where $g$ is the gradient vector?
        # Let's check: $(x * g)_j = \sum_k x_{j-k} g_k$.
        # This matches the structure.
        # So grad_h = 2 * np.convolve(h, grad_conv, mode='same')?
        # Note: `np.convolve(h, g)` is $h * g$.
        # We need to be careful with boundaries.
        
        grad_h = 2 * np.convolve(h, grad_conv, mode='same')
        
        # Update h
        h_new = h - lr * grad_h
        
        # Project to sum 1
        h_new = (h_new - np.mean(h_new)) + 1.0/N
        
        # Check improvement
        val_new = compute_value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            # Decay learning rate
            lr *= 0.999
            # Decay tau
            tau *= 0.99
        else:
            # Revert
            h = best_h.copy()
            lr *= 0.9
            tau *= 0.95
        
        # If tau is very small, we are effectively minimizing max
        if tau < 1e-4:
            break

    return best_h
