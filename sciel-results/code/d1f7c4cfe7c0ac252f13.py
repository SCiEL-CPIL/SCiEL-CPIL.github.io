import numpy as np
from scipy.optimize import minimize
import time

N = 600
DEADLINE = 120.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    # We want to minimize the maximum of the autoconvolution of a nonnegative step function
    # defined on 600 equal intervals over [-1/4, 1/4].
    
    # Let the step heights be x_0, ..., x_{N-1} >= 0.
    # The autoconvolution at lag k is sum_{i} x_i x_{i+k} (for appropriate range).
    # Actually, the continuous autoconvolution of a step function f(t) = sum x_i * 1_{[t_i, t_{i+1})}(t)
    # is a piecewise quadratic function. The maximum will occur at some point.
    
    # However, the task says "normalized maximum of its autoconvolution". 
    # Typically, for such problems, we normalize by the L1 norm or L2 norm or the integral.
    # Given the context of "upper bound on C1", and the standard formulation of such extremal problems,
    # the objective is likely max_t (f*f)(t) / (integral f)^2 or similar.
    # Let's assume the normalization is by the square of the L1 norm (integral), which is common for probability-like functions,
    # or perhaps just the raw max if the function is normalized to have integral 1.
    # If we fix the integral to 1, then we minimize max(f*f).
    
    # Let's fix sum(x_i) = 1 (since interval length is 1/2, integral is (1/2)*sum(x_i)).
    # Let's assume the "normalized" part means we can scale x arbitrarily, so the ratio is scale-invariant.
    # Thus, we can fix sum(x_i) = 1 (or any constant) and minimize max(f*f).
    
    # The autoconvolution of a step function with heights x_i on intervals of width h=1/2/N:
    # (f*f)(t) is a sum of products of heights times the length of overlap.
    # The maximum of the autoconvolution of a step function occurs at the center of the intervals?
    # Actually, for a step function, the autoconvolution is a piecewise quadratic function that is continuous.
    # Its maximum can be found by evaluating it at the midpoints of the intervals? No, the derivative is linear within each interval,
    # so the maximum is at a critical point (derivative 0) or at the boundaries.
    
    # However, there is a simpler discrete approximation often used:
    # The "discrete autoconvolution" y_k = sum_{i} x_i x_{i+k} is related to the continuous one.
    # If we assume the maximum of the continuous autoconvolution is bounded by the max of the discrete autoconvolution
    # (up to a constant factor depending on h), we can optimize the discrete version.
    
    # Let's use the discrete autoconvolution: y_k = sum_{i=max(0,-k)}^{min(N-1-k, N-1)} x_i x_{i+k}.
    # We want to minimize max_k y_k subject to x_i >= 0, sum x_i = 1.
    
    # This is a convex optimization problem?
    # The objective max_k y_k is the maximum of linear functions (since y_k is linear in x? No, y_k is quadratic in x).
    # y_k = x^T A_k x where A_k is a shift matrix. This is not linear.
    # Minimizing the maximum of quadratic forms is non-convex.
    
    # Alternative approach: Use the fact that for nonnegative x, the max of the autoconvolution is related to the L2 norm?
    # No.
    
    # Let's try a different tactic: Start with a smooth function and discretize.
    # The optimal function for minimizing the peak of the autoconvolution (with fixed integral) is often a "tent" or triangular function,
    # or a cosine bump.
    # Let's try a cosine bump: x_i = max(0, cos(pi * (i - N/2) / (N/2)))?
    # Or simply x_i = 1 (uniform).
    
    # Let's implement a local search / gradient descent on the discrete variables.
    # Objective: J(x) = max_k (sum_i x_i x_{i+k})
    # Constraint: sum x_i = 1, x_i >= 0.
    
    # We can use scipy.optimize.minimize with SLSQP or trust-constr.
    # But N=600 is large. SLSQP might be slow.
    
    # Let's try a simpler heuristic: 
    # 1. Initialize x as a triangular distribution (peak in the middle).
    # 2. Perform gradient descent on the max of the discrete autoconvolution.
    
    # To compute the autoconvolution efficiently, we can use FFT.
    # The discrete autoconvolution y is the IFFT of |FFT(x)|^2? 
    # Actually, circular convolution. We need linear convolution.
    # Zero-pad x to size 2N, compute FFT, square, IFFT.
    
    def autoconv_max(x):
        # x is a 1D array of size N
        # Compute linear autoconvolution
        size = 2 * N - 1
        fft_size = 1 << (size - 1).bit_length()
        X = np.fft.rfft(x, n=fft_size)
        Y = np.fft.irfft(X * np.conj(X), n=fft_size)[:size]
        # The autoconvolution is symmetric, so we only need to look at the first half?
        # Actually, y_k for k=0..N-1.
        # The max is over all k.
        return np.max(Y)
    
    # We want to minimize this.
    # Let's use a simple iterative improvement:
    # x_new = x - eta * grad
    # The gradient of max_k y_k with respect to x_i is the gradient of the y_k that achieves the max.
    # y_k = sum_i x_i x_{i+k}
    # dy_k/dx_j = x_{j+k} + x_{j-k} (with boundaries handled)
    
    # This is a bit complex to implement efficiently for N=600 in Python loops.
    # Let's use FFT to compute the gradient?
    # The gradient of the circular convolution is also a convolution.
    # But we have linear convolution.
    
    # Given the time constraint, let's try a few fixed shapes and see which is best.
    # Shapes:
    # 1. Uniform: x = 1/N
    # 2. Triangular: x_i proportional to 1 - |2i/N - 1|
    # 3. Cosine: x_i proportional to cos(pi * (i - N/2) / (N/2)) for i in [0, N-1], 0 otherwise?
    #    Actually, on [-1/4, 1/4], a cosine bump centered at 0: cos(pi * t / (1/2)) = cos(2 pi t).
    #    t_i = -1/4 + (i+0.5)/N * (1/2)
    #    x_i = max(0, cos(2 pi t_i))
    
    # Let's evaluate these.
    
    t = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5/N * 0.5 # midpoints?
    # Actually, the step function is constant on [t_i, t_{i+1}).
    # Let's just use the indices.
    
    x_uniform = np.ones(N) / N
    
    # Triangular
    idx = np.arange(N)
    center = N / 2
    x_tri = 1 - np.abs(idx - center) / center
    x_tri[x_tri < 0] = 0
    if np.sum(x_tri) > 0:
        x_tri = x_tri / np.sum(x_tri)
    
    # Cosine
    # t ranges from -0.25 to 0.25.
    # cos(2 * pi * t) is positive for |t| < 1/4.
    # So it is positive everywhere on the domain.
    x_cos = np.cos(2 * np.pi * t)
    x_cos = np.maximum(x_cos, 0)
    if np.sum(x_cos) > 0:
        x_cos = x_cos / np.sum(x_cos)
    
    # Evaluate
    best_x = x_uniform
    best_val = autoconv_max(x_uniform)
    
    if autoconv_max(x_tri) < best_val:
        best_x = x_tri
        best_val = autoconv_max(x_tri)
        
    if autoconv_max(x_cos) < best_val:
        best_x = x_cos
        best_val = autoconv_max(x_cos)
        
    # Try a local optimization starting from best_x
    # We can use a simple coordinate descent or gradient step.
    
    # Let's try a few gradient steps.
    # We need the gradient of the max autoconvolution.
    # Let k* be the lag that gives the max.
    # grad_i = x_{i+k*} + x_{i-k*}
    # We can compute this efficiently.
    
    def grad_autoconv_max(x):
        size = 2 * N - 1
        fft_size = 1 << (size - 1).bit_length()
        X = np.fft.rfft(x, n=fft_size)
        Y = np.fft.irfft(X * np.conj(X), n=fft_size)[:size]
        k_star = np.argmax(Y)
        # Compute gradient for this k
        # g_i = x_{i+k} + x_{i-k}
        g = np.zeros(N)
        k = k_star
        # i+k in [0, N-1] => i in [-k, N-1-k]
        # i-k in [0, N-1] => i in [k, N-1+k]
        
        # Term 1: x_{i+k}
        # For i in [0, N-1-k], g[i] += x[i+k]
        if N - 1 - k >= 0:
            g[:N-k] += x[k:]
        # Term 2: x_{i-k}
        # For i in [k, N-1], g[i] += x[i-k]
        if k > 0:
            g[k:] += x[:N-k]
            
        return g, Y[k_star]
    
    # Simple gradient descent with projection onto simplex (sum=1, x>=0)
    # Projection onto simplex is non-trivial, but we can use a simple update:
    # x_new = x - eta * grad
    # x_new = max(0, x_new)
    # x_new = x_new / sum(x_new)
    
    x = best_x.copy()
    eta = 0.01
    for _ in range(100):
        if _time_left() < 10:
            break
        g, val = grad_autoconv_max(x)
        x_new = x - eta * g
        x_new = np.maximum(x_new, 0)
        s = np.sum(x_new)
        if s > 0:
            x_new = x_new / s
        else:
            x_new = x
        new_val = autoconv_max(x_new)
        if new_val < val:
            x = x_new
            val = new_val
        else:
            eta /= 2
    
    best_x = x
    best_val = autoconv_max(x)
    
    # Return the heights
    # The task asks for a sequence of nonnegative step heights.
    # We should return the values that, when placed on the intervals, give the function.
    # Since we normalized sum(x)=1, and the interval width is 1/2/N, the integral is 1/2.
    # If the "normalized maximum" implies we should return the raw heights that give the min ratio,
    # the ratio is scale-invariant, so returning x is fine.
    
    return list(best_x)

# Test
if __name__ == "__main__":
    x = construct()
    print("Constructed", len(x))
