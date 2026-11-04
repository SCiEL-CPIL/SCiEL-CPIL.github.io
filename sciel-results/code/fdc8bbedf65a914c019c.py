"""
Strategy: Spectral Peak Minimization via Convex Relaxation and Gradient Descent.

The task is to minimize the maximum absolute autoconvolution (C3') divided by the squared integral.
By homogeneity, we can fix the sum of heights to 1 (or any non-zero constant) and minimize the 
maximum absolute autoconvolution.

The autoconvolution $c[k]$ is related to the power spectral density $P(\omega) = |X(\omega)|^2$ 
by the Wiener-Khinchin theorem: $c[k] = \mathcal{F}^{-1}[P](k)$.
Minimizing $\max_k |c[k]|$ is equivalent to flattening the power spectrum $P(\omega)$. 
If the spectrum is perfectly flat (constant $C$), then the autoconvolution is a delta function 
(non-zero only at $k=0$), and $\max_k |c[k]| = C = \sum x^2 / N$.

This problem is non-convex in the time domain $x$, but we can approach it by:
1. Initializing with a "chaotic" sequence (random signs) to break symmetry.
2. Using a smooth approximation of the max function (Soft-Max) to make the objective differentiable.
3. Performing gradient descent in the time domain to minimize the spectral peakiness.
4. The gradient of the autoconvolution $c[k]$ with respect to $x$ is efficiently computable via FFT 
   or direct convolution logic. Specifically, $\nabla_x c[k] = 2 \cdot \text{circshift}(x, k)$ (with appropriate boundary handling for linear convolution, which is effectively zero-padding).
   
   Actually, for linear convolution $c = x * x$ (reversed), the gradient is simply $2 \cdot \text{flip}(x)$ convolved appropriately? 
   Let's use the FFT derivative:
   $c = \text{IFFT}(|X|^2)$.
   $\frac{\partial c}{\partial x} = 2 \text{IFFT}(X \cdot \text{conj}(X) \cdot \dots)$? No.
   
   Simpler: 
   $c[k] = \sum_j x[j] x[j+k]$.
   $\frac{\partial c[k]}{\partial x[i]} = x[i+k] + x[i-k]$ (with $x[m]=0$ for $m \notin [0, N-1]$).
   
   We minimize $f(x) = \text{SoftMax}(|c|)$.
   $\nabla_x f = \sum_k w_k \frac{\partial |c[k]|}{\partial x}$, where $w_k$ are the soft-max weights.
   $\frac{\partial |c[k]|}{\partial x} = \text{sign}(c[k]) \frac{\partial c[k]}{\partial x}$.
   
   This allows for efficient gradient steps. We also enforce the constraint $\sum x = 1$ by projecting 
   or using a Lagrange multiplier, or simply normalizing $x$ after each step (which is valid due to homogeneity).
"""

import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _compute_conv(x):
    """Compute linear autoconvolution using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv_full = np.fft.irfft(fx * fx, _FFT_SIZE)
    # Extract valid lags
    # Lag 0 is at index 0
    # Lag k (1..N-1) is at index k
    # Lag -k (-(N-1)..-1) is at index _FFT_SIZE - k
    conv_pos = conv_full[:N]
    conv_neg = conv_full[_FFT_SIZE - (N - 1):]
    # Note: conv_neg[0] corresponds to lag -(N-1), conv_neg[N-2] corresponds to lag -1
    # We want the array to represent lags -N+1 .. N-1 or just collect all 2N-1 values.
    # For max absolute value, order doesn't matter.
    return np.concatenate([conv_pos, conv_neg])

def _compute_grad_conv(x, conv):
    """
    Compute gradients of all autoconvolution lags with respect to x.
    Returns a tensor G where G[k, i] = d conv[k] / d x[i].
    This is O(N^2) memory and time, which is 160,000 for N=400. Acceptable.
    """
    # G has shape (2*N - 1, N)
    G = np.zeros((2 * N - 1, N))
    
    # Lags 0 to N-1 (indices 0 to N-1 in G)
    for k in range(N):
        # d conv[k] / d x[i] = x[i+k] + x[i-k]
        # x[i+k] is non-zero for i in 0..N-1-k
        # x[i-k] is non-zero for i in k..N-1
        G[k, :N-k] += x[k:]
        G[k, k:] += x[:N-k]
        
    # Lags -(N-1) to -1 (indices N to 2N-2 in G)
    # Lag -k corresponds to index N + (N-1-k) ? 
    # Let's map lag -k to index in conv array.
    # conv_neg[0] is lag -(N-1). conv_neg[m] is lag -(N-1-m).
    # So lag -k is at conv_neg[N-1-k].
    # In the concatenated array, this is at index N + (N-1-k) = 2N - 1 - k.
    for k in range(1, N):
        idx = 2 * N - 1 - k
        kk = k
        # d conv[-k] / d x[i] = x[i-kk] + x[i+kk]
        # x[i-kk] non-zero for i in kk..N-1
        # x[i+kk] non-zero for i in 0..N-1-kk
        G[idx, kk:] += x[:N-kk]
        G[idx, :N-kk] += x[kk:]
        
    return G

def construct():
    deadline = time.time() + 145.0
    
    # Initialize with random signs (Walsh-like or just random) to break symmetry
    # Random signs give a flat spectrum on average, which is a good starting point.
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Try a few random initializations
    for start_iter in range(5):
        if time.time() > deadline - 10:
            break
            
        if start_iter == 0:
            x = rng.choice([-1, 1], size=N).astype(float)
        elif start_iter == 1:
            # Binary sequence with more structure
            x = (rng.random(N) > 0.5).astype(float) * 2 - 1
        elif start_iter == 2:
            # Random continuous
            x = rng.normal(size=N)
        elif start_iter == 3:
            # Cosine sweep
            x = np.cos(np.linspace(0, 2 * np.pi * 10, N))
        else:
            x = rng.choice([-1, 1], size=N).astype(float)
            
        # Normalize sum to 1 to fix scale
        s = np.sum(x)
        if abs(s) < 1e-12:
            continue
        x = x / s
        
        # Gradient Descent on Soft-Max of |conv|
        # Objective: min SoftMax(|c|)
        # SoftMax(z) = T * log(sum(exp(z/T)))
        
        T = 0.05 # Temperature
        lr = 0.01
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
                
            conv = _compute_conv(x)
            abs_conv = np.abs(conv)
            
            # Soft max
            max_abs = np.max(abs_conv)
            if max_abs > 1e6:
                break
            exp_term = np.exp((abs_conv - max_abs) / T)
            sum_exp = np.sum(exp_term)
            weights = exp_term / sum_exp
            obj_val = max_abs + T * np.log(sum_exp)
            
            # Gradient of SoftMax(|c|)
            # d/dx SoftMax = sum_k weights_k * d/dx |c[k]|
            # d/dx |c[k]| = sign(c[k]) * d c[k] / d x
            # d c[k] / d x is the k-th row of G
            
            G = _compute_grad_conv(x, conv)
            signs = np.sign(conv)
            # grad_x = sum_k weights_k * signs_k * G[k, :]
            # This is a weighted sum of rows of G
            grad_x = (weights * signs) @ G
            
            # Update
            x -= lr * grad_x
            
            # Project back to sum=1
            s = np.sum(x)
            if abs(s) < 1e-12:
                # Fallback
                x = rng.choice([-1, 1], size=N).astype(float)
                s = np.sum(x)
                if abs(s) < 1e-12: continue
                x = x / s
                continue
            x = x / s
            
            # Check improvement
            true_conv = _compute_conv(x)
            true_val = np.max(np.abs(true_conv)) / (np.sum(x) ** 2)
            
            if true_val < best_val:
                best_val = true_val
                best_x = x.copy()
                
            # Adaptive learning rate? 
            # If objective increases, reduce lr
            # Simple: keep lr constant for now, maybe reduce if stuck
            
        # Final check for this start
        if best_x is not None:
             final_conv = _compute_conv(best_x)
             final_val = np.max(np.abs(final_conv)) / (np.sum(best_x) ** 2)
             if final_val < best_val:
                 best_val = final_val

    # If no good result, return uniform (worst case)
    if best_x is None:
        best_x = np.ones(N)
        
    return best_x
