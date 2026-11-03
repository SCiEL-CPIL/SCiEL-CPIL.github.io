"""Rewritten program: spectral coordinate descent with softplus non-negativity and epigraph smoothing."""
import numpy as np
import time
import sys

N = 600
# Time limit guard (evaluator kills at 150s, use 120s for safety)
TIME_LIMIT = 115.0

def _compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution of f.
    f is a sequence of N nonnegative step heights on equal intervals of [-1/4, 1/4].
    The autoconvolution is (f * f)(x) = integral f(y) f(x-y) dy.
    We normalize by ||f||_2^2 (or similar) to get an upper bound on C1.
    
    For a step function with N equal intervals of width h = (1/4 - (-1/4))/N = 1/(2N),
    the autoconvolution at interval boundaries can be computed via discrete convolution.
    
    The score is max_x (f*f)(x) / (integral f^2 dx) or similar normalization.
    Let's use: score = max((f*f)) / (sum(f**2) * h) where h is interval width.
    Actually, let's think about this more carefully.
    
    f is a step function: f(x) = f_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*h, h=1/(2N).
    The autoconvolution g(x) = (f*f)(x) = ∫ f(y) f(x-y) dy.
    
    For a step function, g(x) is piecewise quadratic (or linear in some formulations).
    The maximum of g(x) over all x is what we want to minimize (normalized).
    
    A common normalization: C1 = sup ||f*f||_∞ / ||f||_2^2.
    
    Let's compute this numerically.
    """
    h = 1.0 / (2 * N)
    f = np.asarray(f, dtype=np.float64)
    
    # Compute autoconvolution via FFT for efficiency
    # The continuous autoconvolution of a step function can be approximated by
    # discrete convolution scaled appropriately.
    
    # Using FFT: the autoconvolution in continuous domain
    # g(x) = ∫ f(y) f(x-y) dy
    # For step functions, we can compute this on a fine grid.
    
    # Let's use a fine grid to evaluate g(x)
    # The support of g is [-1/2, 1/2] since f has support [-1/4, 1/4]
    # Use a grid of size M for evaluation
    M = 4096  # Power of 2 for FFT efficiency
    x_eval = np.linspace(-0.5, 0.5, M, endpoint=False)
    
    # Sample f on this grid
    # f is defined on [-1/4, 1/4], zero elsewhere
    f_grid = np.zeros(M)
    # Map x_eval to the step function
    # x_eval[i] = -0.5 + i * (1.0/M)
    # f is nonzero when -1/4 <= x < 1/4
    # For each x_eval, find which interval it falls in
    idx = ((x_eval + 0.25) / h).astype(int)
    valid = (x_eval >= -0.25) & (x_eval < 0.25) & (idx >= 0) & (idx < N)
    f_grid[valid] = f[idx[valid]]
    
    # Compute autoconvolution via FFT
    # g(x) = (f*f)(x) = ∫ f(y) f(x-y) dy
    # In discrete terms with grid spacing dx = 1/M:
    # g[i] ≈ sum_j f_grid[j] * f_grid[(i-j) mod M] * dx
    # But we need to be careful with the convolution definition.
    
    # Use numpy convolve or FFT-based convolution
    # For linear convolution, we need to zero-pad
    n_fft = 2 * M
    F = np.fft.fft(f_grid, n=n_fft)
    G = F * F  # Autoconvolution in frequency domain
    g = np.fft.ifft(G).real * (1.0 / M)  # Scale by dx
    
    # The normalization: ||f||_2^2 = ∫ f^2 dx ≈ sum(f_grid^2) * dx
    norm = np.sum(f_grid ** 2) * (1.0 / M)
    
    if norm < 1e-12:
        return 1e10
    
    # The score is max(g) / norm
    # But g is the autoconvolution, and we want the normalized max
    score = np.max(g) / norm
    return score

def _score_from_coeffs(coeffs):
    """Compute score from softplus parameters."""
    f = np.exp(coeffs)  # softplus approximation: use exp for simplicity, or use np.logaddexp(0, coeffs)
    # Actually, let's use proper softplus: f_i = log(1 + exp(coeffs_i))
    # But for large positive coeffs, this is approximately coeffs_i
    # For large negative coeffs, this is approximately exp(coeffs_i)
    f = np.logaddexp(0, coeffs)
    return _compute_score(f)

def construct():
    start_time = time.time()
    
    # Initial guess: uniform
    best_f = np.ones(N)
    best_score = _compute_score(best_f)
    
    # Spectral coordinate descent
    # Work in frequency domain
    # f is real and nonnegative, so F = FFT(f) has conjugate symmetry
    # We optimize the magnitude and phase of F, then project back to time domain
    # ensuring non-negativity via softplus
    
    # Start with uniform initialization
    f = np.ones(N)
    # Normalize to have ||f||_2^2 = 1 for stability
    norm = np.sqrt(np.sum(f ** 2))
    f = f / norm
    
    # Compute FFT
    F = np.fft.fft(f)
    
    # Number of unique frequency bins (with conjugate symmetry)
    # For N even, bins 0, 1, ..., N/2 are independent
    # F[N-k] = conj(F[k])
    n_bins = N // 2 + 1  # bins 0 to N/2
    
    # Extract magnitudes and phases
    mags = np.abs(F[:n_bins])
    phases = np.angle(F[:n_bins])
    
    # Coordinate descent on magnitudes and phases
    # For each bin, adjust magnitude and phase to minimize the score
    
    # Use a simple gradient-free approach: coordinate descent with line search
    # Or use the fact that the objective in frequency domain is non-trivial
    
    # Alternative: work in time domain with gradient descent on softplus parameters
    # Let's try time-domain gradient descent with softplus
    
    # Initialize softplus parameters
    # f_i = log(1 + exp(c_i))
    # c_i = log(exp(f_i) - 1) for f_i > 0
    # For f_i = 1, c_i = log(e - 1)
    c = np.full(N, np.log(np.e - 1))
    
    # Gradient descent
    lr = 0.1
    best_c = c.copy()
    best_f = np.logaddexp(0, c)
    best_score = _compute_score(best_f)
    
    # Track best
    iterations = 0
    while time.time() - start_time < TIME_LIMIT - 10:
        # Compute gradient numerically
        # Use central differences for a few coordinates at a time (stochastic gradient)
        # Or full gradient if feasible
        
        # For efficiency, use a subset of coordinates per iteration
        batch_size = min(50, N)
        idx = np.random.choice(N, batch_size, replace=False)
        
        eps = 1e-4
        grad = np.zeros(batch_size)
        for j, i in enumerate(idx):
            c_plus = c.copy()
            c_plus[i] += eps
            s_plus = _score_from_coeffs(c_plus)
            
            c_minus = c.copy()
            c_minus[i] -= eps
            s_minus = _score_from_coeffs(c_minus)
            
            grad[j] = (s_plus - s_minus) / (2 * eps)
        
        # Update
        c[idx] -= lr * grad
        
        # Check convergence
        f_new = np.logaddexp(0, c)
        score_new = _compute_score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            best_c = c.copy()
            # Adaptive learning rate
            lr = min(lr * 1.1, 1.0)
        else:
            lr = max(lr * 0.9, 0.01)
        
        iterations += 1
    
    # Final polish: use the best solution found
    # Ensure non-negativity
    best_f = np.maximum(best_f, 0)
    
    # Return as a sequence of floats
    return [float(x) for x in best_f]

# Keep the original function signature
def construct():
    """Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4] to minimize the normalized maximum of its autoconvolution."""
    start_time = time.time()
    
    # Initial guess: uniform
    best_f = np.ones(N)
    best_score = _compute_score(best_f)
    
    # Time-domain gradient descent with softplus parameters
    # f_i = log(1 + exp(c_i))
    c = np.full(N, np.log(np.e - 1))  # f_i = 1
    
    lr = 0.05
    best_c = c.copy()
    best_f = np.logaddexp(0, c)
    best_score = _compute_score(best_f)
    
    # Momentum
    velocity = np.zeros(N)
    beta = 0.9
    
    iterations = 0
    while time.time() - start_time < TIME_LIMIT - 15:
        # Compute gradient numerically using a batch
        batch_size = min(30, N)
        idx = np.random.choice(N, batch_size, replace=False)
        
        eps = 1e-4
        grad_batch = np.zeros(batch_size)
        for j, i in enumerate(idx):
            c_plus = c.copy()
            c_plus[i] += eps
            s_plus = _score_from_coeffs(c_plus)
            
            c_minus = c.copy()
            c_minus[i] -= eps
            s_minus = _score_from_coeffs(c_minus)
            
            grad_batch[j] = (s_plus - s_minus) / (2 * eps)
        
        # Update with momentum
        velocity[idx] = beta * velocity[idx] - lr * grad_batch
        c[idx] += velocity[idx]
        
        # Evaluate
        f_new = np.logaddexp(0, c)
        score_new = _compute_score(f_new)
        
        if score_new < best_score:
            best_score = score_new
            best_f = f_new.copy()
            best_c = c.copy()
            # Slightly increase learning rate
            lr = min(lr * 1.05, 0.5)
        else:
            # Revert and decrease learning rate
            c = best_c.copy()
            lr = max(lr * 0.95, 0.005)
        
        iterations += 1
    
    # Return the best solution
    best_f = np.maximum(best_f, 0)
    return [float(x) for x in best_f]
