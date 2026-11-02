"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    w = 1.0 / 300  # interval width
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    def autoconv_max(h):
        """Compute the maximum of the autoconvolution of step function h."""
        # h has N elements, each of width w
        # Autoconvolution at sample points
        # The autoconvolution is piecewise linear with breakpoints at multiples of w
        # Sample at the midpoints of the 2N-1 intervals of the autoconvolution support
        
        # Support of autoconv is [-1/2, 1/2], divided into 2N-1 intervals of width w
        # Actually, the autoconv of two step functions with N intervals each has up to 2N-1 linear pieces
        
        # For efficiency, compute autoconvolution via FFT or direct
        # Let's use the fact that autoconv of step function can be computed from cumulative sums
        
        # The autoconvolution (f*f)(t) for step function can be computed as:
        # For t in [k*w, (k+1)*w], it's linear
        # Let's compute at the breakpoints
        
        # Using convolution approach:
        # f is represented as h_i * rect on interval i
        # f*f can be computed via discrete convolution with appropriate scaling
        
        # Simpler: use scipy.signal.fftconvolve or numpy convolve
        # But we need the continuous autoconvolution
        
        # For a step function, the autoconvolution at point t is:
        # (f*f)(t) = sum_i sum_j h_i * h_j * overlap(i, j, t)
        # where overlap is the length of intersection
        
        # This is O(N^2) which is 360000 - feasible for one eval but slow for optimization
        # Let's use FFT-based approach
        
        # Actually, for step functions, we can compute the autoconvolution efficiently
        # The autoconvolution of a step function is a piecewise quadratic... no, piecewise linear
        # Wait, autoconv of two step functions: each is piecewise constant, so their product in the integral
        # gives piecewise linear
        
        # Let me just compute it directly for the max
        # Sample at enough points to capture the max
        
        # The autoconvolution is piecewise linear, with breakpoints at t = (i-j)*w for i,j in range(N)
        # So breakpoints are at multiples of w from -(N-1)*w to (N-1)*w
        # That's 2N-1 = 1199 breakpoints, giving 1199 linear pieces
        
        # The max of a piecewise linear function occurs at a breakpoint or endpoint
        # So we just need to evaluate at the breakpoints
        
        # Breakpoints: t_k = (k - (N-1)) * w for k = 0, ..., 2N-2
        # Actually, let me think again.
        
        # f is nonzero on [-1/4, 1/4] = [-(N/2)*w, (N/2)*w] since N=600, N/2=300
        # So f is on intervals indexed 0..599, with x_i = -1/4 + i*w
        
        # (f*f)(t) is nonzero for t in [-1/2, 1/2]
        # The breakpoints of (f*f) occur at t = x_i - x_j for various i,j
        # x_i - x_j = (i-j)*w
        # So breakpoints at k*w for k = -(N-1), ..., N-1
        # That's 2N-1 = 1199 points, from -(N-1)*w to (N-1)*w
        
        # But the support is [-1/2, 1/2] = [-300*w, 300*w], and (N-1)*w = 599*w ≈ 1.997, which is > 1/2
        # Wait, that's wrong. Let me recompute.
        
        # x_i = -1/4 + i*w, i=0..599
        # x_0 = -1/4, x_599 = -1/4 + 599/300 = -1/4 + 1.9967 = 1.7467
        # That's not right. 599/300 = 1.9967, so x_599 = -0.25 + 1.9967 = 1.7467
        # But we want support on [-1/4, 1/4]. So x_599 should be 1/4.
        # -1/4 + 599*w = 1/4 => 599*w = 1/2 => w = 1/(2*599) = 1/1198
        
        # Hmm, let me re-read the problem. "600 equal intervals on [-1/4, 1/4]"
        # So the total length is 1/2, divided into 600 intervals, each of width 1/1200
        
        w = 1.0 / 1200
        
        # x_i = -1/4 + i*w, i=0..599
        # x_0 = -1/4, x_599 = -1/4 + 599/1200 = -1/4 + 0.4992 = 0.2492 ≈ 1/4
        
        # Autoconv support: [-1/2, 1/2]
        # Breakpoints at (i-j)*w for i,j in 0..599, so k*w for k in -599..599
        # That's 1199 breakpoints
        
        # Evaluate autoconv at these breakpoints
        # (f*f)(t) = sum_i sum_j h_i h_j * overlap_length(i, j, t)
        
        # For t = k*w, the overlap of interval i and interval j shifted by t:
        # Interval i: [x_i, x_{i+1}] = [-1/4+i*w, -1/4+(i+1)*w]
        # Interval j shifted: [t - x_{j+1}, t - x_j] = [k*w - (-1/4+(j+1)*w), k*w - (-1/4+j*w)]
        #   = [k*w + 1/4 - (j+1)*w, k*w + 1/4 - j*w]
        #   = [1/4 + (k-j-1)*w, 1/4 + (k-j)*w]
        
        # Overlap of [-1/4+i*w, -1/4+(i+1)*w] and [1/4+(k-j-1)*w, 1/4+(k-j)*w]
        # Let d = k - j
        # Second interval: [1/4+(d-1)*w, 1/4+d*w]
        # First interval: [-1/4+i*w, -1/4+(i+1)*w]
        
        # This is getting complicated. Let me just use numerical convolution.
        
        # Use FFT to compute the autoconvolution
        # Represent f as a vector of heights, compute convolution with itself
        # The discrete convolution gives us the autoconvolution sampled at appropriate points
        
        # For step functions of width w, the autoconvolution can be computed as:
        # (f*f)(t) ≈ w * sum_i h_i * h_j where the intervals overlap
        
        # Let me use scipy.signal.fftconvolve for efficiency
        
        from scipy.signal import fftconvolve
        # Convolve h with h, but we need to account for the continuous nature
        # The discrete convolution of h with h, scaled by w, gives the autoconvolution
        # sampled at points t = (i-j)*w
        
        conv = fftconvolve(h, h, mode='full') * w
        # conv has length 2N-1 = 1199
        # The max of the autoconvolution is the max of conv
        return np.max(conv)
    
    def objective(h):
        """Smooth approximation of the max for optimization."""
        # Use log-sum-exp for smooth max
        conv = fftconvolve(h, h, mode='full') * w
        # Smooth max
        alpha = 50.0
        shifted = conv / alpha
        m = np.max(shifted)
        return alpha * (m + np.log(np.sum(np.exp(shifted - m))))
    
    def objective_grad(h):
        """Gradient of the smooth max objective."""
        conv = fftconvolve(h, h, mode='full') * w
        alpha = 50.0
        shifted = conv / alpha
        m = np.max(shifted)
        exp_shifted = np.exp(shifted - m)
        softmax = exp_shifted / np.sum(exp_shifted)
        # Gradient of alpha*(m + log(sum(exp(shifted-m)))) w.r.t. h
        # = alpha * softmax * d(conv)/dh / alpha = softmax * d(conv)/dh
        # conv = fftconvolve(h,h)*w, so d(conv)/dh is the convolution structure
        # This is complex, let's use numerical gradient or just use L-BFGS-B with finite differences
        
        # Actually, let's just use the objective without gradient and let scipy handle it
        return objective(h)
    
    # Use L-BFGS-B with bounds
    bounds = [(0, None)] * N
    
    # Try a few different initial guesses
    best_h = h0.copy()
    best_val = autoconv_max(h0)
    
    # Run optimization
    try:
        result = minimize(
            objective,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if time.time() > deadline:
            return best_h.tolist()
        
        if result.fun < best_val:
            best_h = result.x
            best_val = autoconv_max(best_h)
    except Exception:
        pass
    
    # Try a different initial guess: cosine-shaped
    if time.time() < deadline - 30:
        h_cos = np.cos(np.pi * np.arange(N) / N) ** 2
        h_cos = np.maximum(h_cos, 0)
        try:
            result2 = minimize(
                objective,
                h_cos,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() < deadline:
                if result2.fun < best_val:
                    best_h = result2.x
                    best_val = autoconv_max(best_h)
        except Exception:
            pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
