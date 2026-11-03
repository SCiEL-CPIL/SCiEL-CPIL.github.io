import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 600
TIME_LIMIT = 145  # Safety margin before 150s kill

# Precompute grid
# The intervals are equal on [-1/4, 1/4].
# There are N intervals.
# The step function is constant on each interval.
# We represent the function by its values at the centers of the intervals.
x_edges = np.linspace(-0.25, 0.25, N + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

# Precompute FFT size for fast convolution
# Convolution of two length-N sequences results in length 2N-1.
# We need FFT size >= 2N-1. Next power of 2 for 1199 is 2048.
FFT_SIZE = 2048

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
        
    # Use FFT for convolution
    f_padded = np.zeros(FFT_SIZE)
    f_padded[:N] = f
    F = fft(f_padded)
    conv_padded = ifft(F * F)
    conv = np.real(conv_padded[:2*N-1])
    
    # The maximum of the autoconvolution of a non-negative function 
    # supported on [-1/4, 1/4] (which is effectively compactly supported)
    # is typically at the center (lag 0), due to symmetry if we center the function.
    # However, to be safe and exact, we take the max.
    # Note: For a function centered at 0, the peak is at index N-1 (lag 0).
    max_conv = np.max(conv)
    
    return max_conv / (s * s)

def parametric_to_step(params):
    """
    Convert cosine basis coefficients to step function heights.
    We use a cosine basis that is zero at the boundaries x=-0.25 and x=0.25.
    phi_k(x) = cos(k * pi * (x + 0.25) / 0.5) for k=1, 2, ...
    This ensures f(-0.25) = f(0.25) = 0, which is often optimal for minimizing 
    boundary effects in convolution, though not strictly required. 
    Actually, the problem doesn't require zero boundary. 
    However, functions with compact support (zero at edges) tend to have 
    lower peak autoconvolution than functions that are non-zero at edges 
    (like a constant or a wide Gaussian).
    
    Let's use a basis that allows non-zero boundaries but is smooth.
    Actually, let's stick to a simple polynomial/cosine expansion centered at 0.
    f(x) = sum c_k * T_k(x) ?
    
    Let's use a standard cosine series on [-0.25, 0.25].
    f(x) = a_0 + sum_{k=1}^M a_k * cos(k * 4 * pi * x)
    This allows any value at boundaries.
    
    However, to ensure non-negativity, we can optimize on the coefficients 
    and project to non-negative space, or use a softplus activation on the coefficients?
    No, the coefficients are linear. We can just clip the result f(x) to be >= 0.
    But clipping breaks differentiability for L-BFGS.
    
    Alternative: Use a basis of non-negative functions.
    e.g., f(x) = sum c_k * exp(-((x - mu_k)/sigma_k)^2)
    This goes back to the parametric family which was too restrictive.
    
    Let's try: f(x) = (sum_{k=0}^M c_k * cos(k * 4 * pi * x))^2 + epsilon
    This guarantees non-negativity. The optimization is on c_k.
    The number of parameters M can be moderate (e.g., 10-20).
    """
    M = len(params)
    # params are c_0, c_1, ..., c_{M-1}
    # x is x_centers
    # Argument for cos: k * 4 * pi * x
    # Since x is in [-0.25, 0.25], 4*pi*x is in [-pi, pi].
    # cos(k * 4 * pi * x)
    
    # Construct the matrix of basis functions
    # k=0: cos(0) = 1
    # k=1: cos(4*pi*x)
    # ...
    k = np.arange(M)
    # Shape (M, N)
    basis = np.cos(np.outer(k, 4 * np.pi * x_centers))
    
    # Linear combination
    f_lin = basis.T @ params
    
    # Square to ensure non-negativity
    f = f_lin ** 2
    
    # Add a tiny epsilon to avoid zero sum if all params are 0
    f += 1e-10
    
    return f

def objective(params):
    """
    Objective function for optimization.
    """
    f = parametric_to_step(params)
    return compute_score(f)

def construct():
    start_time = time.time()
    
    # Number of basis functions
    # Too many leads to overfitting/oscillation. Too few is too restrictive.
    # Let's try M=15.
    M = 15
    
    # Initial guess
    # A broad bump is a good start.
    # c_0 dominates.
    # Let's initialize with a Gaussian-like profile in the cosine domain?
    # Or just random small values + large c_0.
    
    # Strategy: Multiple restarts with random initializations, then L-BFGS-B.
    
    best_score = np.inf
    best_params = None
    
    # Time budget per restart
    # We have ~145s.
    # Let's allow 10-15 restarts.
    num_restarts = 12
    
    time_per_restart = (TIME_LIMIT - 10) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Random initialization
        # Scale: c_0 should be significant. Others smaller.
        # f(x) ~ c_0^2. If c_0=1, f~1.
        # Let's try c_0 in [0.5, 2.0], others in [-0.5, 0.5]
        
        init_params = np.random.randn(M) * 0.1
        init_params[0] = np.random.uniform(0.5, 1.5)
        
        # Ensure the function is not identically zero (handled by epsilon)
        
        try:
            # Use L-BFGS-B
            # Bounds: not strictly necessary if we square, but helps stability.
            # Let's set bounds to [-5, 5] for all coefficients.
            bounds = [(-5, 5)] * M
            
            res = minimize(
                objective, 
                init_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            
            final_score = objective(res.x)
            
            if final_score < best_score:
                best_score = final_score
                best_params = res.x.copy()
                
        except Exception:
            continue
            
    # Final refinement on the best found
    if best_params is not None and time.time() - start_time < TIME_LIMIT - 5:
        try:
            res_final = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=[(-5, 5)] * M,
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            final_score = objective(res_final.x)
            if final_score < best_score:
                best_score = final_score
                best_params = res_final.x.copy()
        except Exception:
            pass

    # If optimization failed completely, fallback to a known good shape
    if best_params is None or np.isinf(best_score):
        # Fallback: Gaussian
        f_gauss = np.exp(-20 * x_centers**2)
        # We need to return the step heights.
        # The interface expects the sequence of heights.
        # We can just return f_gauss directly.
        return [float(x) for x in f_gauss]
        
    # Convert best params to step function
    f_final = parametric_to_step(best_params)
    
    # Ensure non-negative
    f_final = np.maximum(f_final, 0.0)
    
    # One last sanity check: compare with a simple analytic candidate
    # e.g. Triangle (conv of two boxes)
    # f_tri = max(0, 1 - 4*|x|)
    f_tri = np.maximum(0, 1 - 4 * np.abs(x_centers))
    s_tri = compute_score(f_tri)
    
    # e.g. Cosine bump (smooth, zero at boundaries)
    # f_cos = cos(pi * x / 0.5)^2 ? No, that's non-zero at boundaries?
    # cos(0)=1, cos(pi)=-1. Square is 1.
    # We want zero at boundaries for potentially better performance?
    # Let's try f_cos = sin(2*pi*x + pi/2)^2 ? 
    # Actually, the squared cosine expansion already handles the shape.
    # Let's just check if the triangle is better.
    
    if s_tri < best_score:
        f_final = f_tri
        best_score = s_tri
        
    # Check constant (worst case usually, but good to have)
    f_const = np.ones(N)
    s_const = compute_score(f_const)
    if s_const < best_score:
        f_final = f_const
        best_score = s_const

    # Return as list of floats
    return [float(x) for x in f_final]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
