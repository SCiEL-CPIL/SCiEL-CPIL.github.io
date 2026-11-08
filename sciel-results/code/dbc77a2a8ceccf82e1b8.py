"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.signal import fftconvolve

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Problem setup:
    # N=600 equal intervals on [-1/4, 1/4].
    # Interval width dx = 0.5 / N = 1/1200.
    dx = 0.5 / N
    
    # Helper to compute the normalized maximum of autoconvolution.
    # The score is upper bound on C1.
    # C1 relates to sup_t (f*f)(t) / (||f||_1)^2.
    # For step functions with heights h[i] on intervals of width dx:
    # ||f||_1 = dx * sum(h)
    # (f*f)(t) is a piecewise linear function. Its maximum occurs at a vertex.
    # The vertices are at boundaries of the intervals of the convolution.
    # The convolution of two step functions on N intervals has 2N-1 "cells" or we can think of it differently.
    # Let's use the discrete convolution as a proxy but scaled correctly.
    # Actually, the exact maximum of the continuous autoconvolution of a step function
    # is the maximum of the discrete autoconvolution values? No.
    # Let's look at the inspiration 1: it used np.fft and returned max(conv)/dx.
    # Let's verify the scaling.
    # If f is constant 1 on [0,1], f*f is triangle peak 1. ||f||_1 = 1. Ratio 1.
    # Discrete h = [1, ..., 1] (N times). sum(h)=N. ||f||_1 = N*dx = 0.5.
    # Discrete conv max is N.
    # If we use formula: max_discrete_conv * dx / (sum(h)*dx)^2 ?
    # max_discrete_conv = N. dx = 0.5/N.
    # Num = N * (0.5/N) = 0.5.
    # Denom = (0.5)^2 = 0.25.
    # Ratio = 0.5 / 0.25 = 2.
    # Wait, for f=1 on [0,1], sup (f*f) = 1. ||f||_1 = 1. Ratio 1.
    # My discrete approx gave 2. Why?
    # The continuous convolution of two step functions is a piecewise linear spline.
    # The discrete convolution sum approximates the integral.
    # (f*f)(t) = ∫ f(t-s)f(s) ds.
    # If f is constant h on interval, (f*f)(t) changes linearly.
    # The maximum is at the center of the overlap.
    # For uniform h=1, N=600, dx=1/1200.
    # Peak of continuous conv is at t=0 (if centered) or center of domain.
    # Value = ∫ f(s)f(t-s) ds.
    # For uniform, it's a triangle. Peak height = ∫_{-0.25}^{0.25} 1*1 ds = 0.5.
    # ||f||_1 = 0.5.
    # Ratio = 0.5 / (0.5)^2 = 2.
    # So the seed program (uniform) gives score 2?
    # But the task says seed score is 1.0.
    # "score is the upper bound on C1".
    # "search score is seed/value".
    # If seed returns [1.0]*N and score is 1.0, then value(seed) = seed_value.
    # Maybe the "value" is defined differently.
    # Let's assume the evaluator calculates the value such that uniform gives 1.0.
    # Or maybe C1 is defined as 1/2 * sup ...?
    # Let's look at the "First autocorrelation inequality".
    # Usually C1 is the constant in ||f*f||_inf <= C1 ||f||_1^2.
    # For f=1 on [0,1], ||f*f||_inf = 1, ||f||_1^2 = 1. So C1 >= 1.
    # For f=1 on [-1/4, 1/4], ||f||_1 = 1/2. ||f*f||_inf = 1/2.
    # Ratio = (1/2) / (1/4) = 2.
    # So C1 >= 2?
    # If the seed score is 1.0, and seed is uniform, then the "value" returned by the evaluator
    # for uniform must be such that seed/value = 1.0.
    # If value = 2.0, then 1.0/2.0 = 0.5? No, score is seed/value.
    # If score is 1.0, then value = seed.
    # Maybe the seed value is not 2.0?
    # Let's re-read: "Choose a nonnegative step function ... to minimize the normalized maximum ... this gives an upper bound on C1."
    # "score is the upper bound on C1".
    # "search score is seed/value".
    # If seed program returns [1.0]*N, and its "value" (upper bound on C1) is V_seed.
    # Then search score = V_seed / V_current.
    # If search score is 1.0, then V_current = V_seed.
    # So I just need to minimize V.
    # What is V?
    # Likely V = sup_t (f*f)(t) / (||f||_1)^2.
    # For uniform, V = 2.
    # So if I find a function with V = 1.5, score = 2.0 / 1.5 = 1.33.
    # So I should minimize V.
    
    def compute_value(h):
        h = np.array(h, dtype=float)
        h[h < 0] = 0
        s = np.sum(h)
        if s == 0:
            return float('inf')
        # Normalize to ||f||_1 = 1? Or just compute the ratio.
        # Ratio = max_conv_continuous / (dx * s)^2
        # We need max_conv_continuous.
        # For step functions, the autoconvolution is a piecewise linear function.
        # The maximum is at a peak.
        # The discrete convolution c[k] = sum_i h[i] h[i+k] approximates the integral.
        # Specifically, (f*f)(t_k) ≈ dx * c[k] where t_k are the centers?
        # Actually, the continuous convolution of two step functions is a spline.
        # The values at the "nodes" (boundaries of the convolution intervals) can be computed.
        # Let's use the property that max_continuous_conv = max_discrete_conv * dx?
        # Let's test with uniform.
        # h = [1]*N. c[N-1] = N.
        # max_discrete_conv = N.
        # dx = 0.5/N.
        # max_cont ≈ N * dx = 0.5.
        # ||f||_1 = N * dx = 0.5.
        # Ratio = 0.5 / (0.5)^2 = 2.
        # This matches the continuous calculation.
        # So V = max_discrete_conv * dx / (dx * s)^2 = max_discrete_conv / (dx * s^2).
        # Wait.
        # max_cont = max_disc * dx.
        # ||f||_1 = s * dx.
        # V = (max_disc * dx) / (s * dx)^2 = max_disc * dx / (s^2 * dx^2) = max_disc / (s^2 * dx).
        
        # Let's re-verify.
        # max_cont = max_disc * dx.
        # V = max_cont / (||f||_1)^2 = (max_disc * dx) / (s * dx)^2 = max_disc * dx / (s^2 * dx^2) = max_disc / (s^2 * dx).
        
        # For uniform: max_disc = N, s = N, dx = 0.5/N.
        # V = N / (N^2 * 0.5/N) = N / (0.5 * N) = 2.
        # Correct.
        
        # So to minimize V, we minimize max_disc / s^2.
        # Since dx is constant, we can just minimize max_disc / s^2.
        
        # Compute discrete autoconvolution
        # Using FFT for speed
        f = np.fft.fft(h, n=2*N)
        conv = np.fft.ifft(f * f).real
        # conv has length 2N. The valid part is roughly N to 2N-1?
        # np.convolve(h, h, 'full') has length 2N-1.
        # The FFT result is circular.
        # We need the linear convolution.
        # Zero padding to 2N is sufficient for linear conv of length 2N-1.
        # The linear convolution is contained in the first 2N-1 elements of the IFFT result?
        # Actually, the IFFT of F*F gives the circular convolution.
        # If we pad to 2N, the circular convolution is the same as linear convolution
        # for the first 2N-1 indices?
        # Let's check.
        # Linear conv length 2N-1.
        # Circular conv length 2N.
        # Indices 0 to 2N-2 correspond to linear conv.
        # Index 2N-1 is wrap-around?
        # Actually, if we pad to M >= 2N-1, then the first 2N-1 elements of the IFFT
        # are the linear convolution.
        # So we take max of conv[0:2N-1].
        
        max_disc = np.max(conv[:2*N-1])
        return max_disc / (s*s * dx)

    # Strategy:
    # 1. Start with a Gaussian.
    # 2. Use subgradient descent to minimize V.
    # 3. Maybe try some random restarts or different initializations if time permits.
    
    # Initial Gaussian
    x = np.arange(N)
    mu = N / 2.0
    # Optimal sigma?
    # For a Gaussian, the convolution is also Gaussian.
    # f(x) = exp(-x^2/sigma^2).
    # f*f ~ exp(-x^2/(2*sigma^2)).
    # Peak height scales as sigma.
    # L1 norm scales as sigma.
    # Ratio ~ sigma / sigma^2 = 1/sigma.
    # So smaller sigma (narrower peak) gives smaller ratio?
    # Wait.
    # If f is delta-like, f*f is delta-like.
    # sup (f*f) / ||f||_1^2.
    # If f = delta, sup is inf? No, delta is not a function.
    # If f is a narrow box of width w, height 1/w.
    # ||f||_1 = 1.
    # f*f is a triangle of width 2w, height 1/w.
    # Ratio = (1/w) / 1 = 1/w.
    # As w -> 0, ratio -> inf.
    # So we want WIDE functions.
    # Uniform is the widest.
    # But uniform gives 2.
    # Can we do better than 2?
    # The inequality is ||f*f||_inf <= C1 ||f||_1^2.
    # For L1 functions, C1 is not bounded?
    # No, for step functions on a fixed domain, maybe?
    # Actually, the "First autocorrelation inequality" usually refers to:
    # ||f*f||_inf <= ||f||_1 ||f||_inf? No.
    # Maybe it's related to the Wiener algebra or something.
    # Let's check the literature.
    # "First autocorrelation inequality" might be:
    # max_t (f*f)(t) <= C * (integral f)^2 ?
    # If so, for f=1, C=2.
    # Is it possible to get C < 2?
    # If we take f to be a cosine bump?
    # f(x) = cos(pi x) on [-0.5, 0.5]?
    # Let's just optimize.
    
    # Start with uniform, then perturb?
    # Or start with Gaussian and let it widen?
    # Subgradient descent:
    # V(h) = max_k (h*h)[k] / (sum h)^2 * (1/dx).
    # Let M = max_k conv[k].
    # grad V with respect to h[j].
    # This is tricky because of the max and the denominator.
    # Let's use the method from Inspiration 1.
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try a few initializations
    inits = []
    inits.append(np.ones(N))
    # Gaussian with various widths
    for sigma in [0.1, 0.2, 0.3, 0.5, 0.8]:
        g = np.exp(-0.5 * ((x - mu) / (sigma * N))**2)
        inits.append(g)
    
    # Select best init
    for init in inits:
        val = compute_value(init)
        if val < best_val:
            best_val = val
            best_h = init.copy()
    
    h = best_h.copy()
    h = h / np.sum(h) # Normalize to sum=1 for optimization?
    # Note: V is homogeneous of degree 0?
    # V(c*h) = max(c*h*c*h) / (sum c*h)^2 * 1/dx = c^2 * max(h*h) / (c^2 * sum(h)^2) * 1/dx = V(h).
    # So scaling doesn't matter. We can fix sum(h)=1.
    
    # Subgradient Descent
    # We want to minimize V(h).
    # V(h) = max_k conv_k / (sum h)^2.
    # Let S = sum h.
    # Let k* be the argmax.
    # conv_k = sum_i h_i h_{i+k}.
    # d conv_k / d h_j = h_{j+k} + h_{j-k}.
    # d V / d h_j = (1/S^2) * d conv_k / d h_j - (2/S^3) * conv_k * d S / d h_j
    # d S / d h_j = 1.
    # So grad_j = (1/S^2) * (h_{j+k*} + h_{j-k*}) - (2 conv_k / S^3).
    
    eta = 0.1
    for step in range(1000):
        if time.time() > deadline - 5:
            break
            
        # Compute conv
        f = np.fft.fft(h, n=2*N)
        conv = np.fft.ifft(f * f).real
        conv = conv[:2*N-1]
        k_max = np.argmax(conv)
        val = conv[k_max] / (np.sum(h)**2 * dx)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # Gradient
        S = np.sum(h)
        g1 = np.roll(h, -k_max)
        g2 = np.roll(h, k_max)
        # Handle boundaries for roll?
        # h is defined on 0..N-1.
        # conv_k = sum_{i} h_i h_{i+k}.
        # If i+k >= N, h_{i+k} is 0.
        # np.roll wraps around. This is incorrect for linear convolution.
        # We need to be careful with the gradient.
        # Let's use explicit loops or careful array slicing.
        # Or just use numerical gradient? No, too slow.
        # Let's construct the gradient correctly.
        # grad_j = h_{j+k} + h_{j-k} (with 0 outside).
        
        grad = np.zeros(N)
        # h_{j+k}
        # j from 0 to N-1-k
        if k_max < N:
            grad[:N-k_max] += h[k_max:]
            # h_{j-k}
            # j from k to N-1
            grad[k_max:] += h[:N-k_max]
        
        # Full gradient of V
        # V = conv_k / S^2
        # dV/dh_j = (1/S^2) grad_j - (2 conv_k / S^3)
        grad_V = (1.0 / (S*S)) * grad - (2.0 * conv[k_max] / (S**3))
        
        # Update
        h_new = h - eta * grad_V
        h_new = np.clip(h_new, 0, None)
        S_new = np.sum(h_new)
        if S_new == 0:
            break
        h_new = h_new / S_new
        
        # Check if improved
        val_new = compute_value(h_new)
        if val_new < val:
            h = h_new
            eta = min(eta * 1.1, 0.5)
        else:
            eta *= 0.5
            if eta < 1e-6:
                break
                
    return best_h.tolist()
# EVOLVE-BLOCK-END
