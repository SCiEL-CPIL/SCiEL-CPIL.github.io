"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute the autoconvolution weights for the discrete step function
# The step function f(x) = sum_{i=0}^{N-1} h_i * 1_{[x_i, x_{i+1})}(x) on [-1/4, 1/4]
# The autoconvolution (f * f)(t) = integral f(t-s) f(s) ds
# For step functions, this can be computed via the overlap of intervals.
# However, the "third autocorrelation inequality" likely refers to a specific bound C3.
# Given the task description: "improve an autoconvolution upper bound" and "score is the upper bound on C3".
# And "lower is better".
#
# Let's assume the standard context: We are looking for a function f with ||f||_1 = 1 (or similar normalization)
# to minimize a specific autocorrelation sum or bound.
# A common inequality involves the sum of absolute values of autocorrelation coefficients.
# Let's look at the "signed step function" aspect.
#
# A typical objective for such problems (like minimizing the sidelobe levels or a specific moment)
# often involves the discrete autocorrelation R[k] = sum_i h_i h_{i+k}.
# The "upper bound on C3" might refer to a specific constant in an inequality like:
# C3 >= sum |R[k]| or similar.
#
# Let's assume the objective is to minimize the sum of squares of the autocorrelation sidelobes,
# or perhaps the maximum absolute sidelobe.
# However, the prompt says "upper bound on C3".
#
# Let's try a standard optimization: Minimize the "energy" of the sidelobes in the autocorrelation.
# Let h be the vector of heights.
# We need to normalize h. Let's assume ||h||_1 = 1 or ||h||_2 = 1.
# Let's assume ||h||_2 = 1 for simplicity, or perhaps the area is fixed.
# "400 equal intervals on [-1/4, 1/4]". Width of each interval w = (1/2)/400 = 1/800.
# Area = sum |h_i| * w. Let's fix Area = 1. So sum |h_i| = 800.
#
# The autocorrelation of the continuous function f is:
# (f*f)(t) = integral f(s) f(t-s) ds.
# For step functions, this is a piecewise linear function.
# The "C3" might be a specific integral or maximum value.
#
# Given the ambiguity, let's look at the seed: `return np.ones(N)`.
# If h = [1, 1, ..., 1], Area = 400 * (1/800) = 0.5.
# Maybe the normalization is different.
#
# Let's assume the objective is to minimize the L2 norm of the autocorrelation sidelobes,
# which is a common proxy for "sharpness" or "concentration".
# Discrete autocorrelation r[k] = sum_{i} h[i] h[i+k].
# Objective: sum_{k=1}^{N-1} r[k]^2.
# This is equivalent to minimizing the energy of the sidelobes.
#
# Let's implement this.
#
# Variables: h (N,)
# Constraints: sum |h_i| = C (fixed area). Let's pick C=1 for simplicity, or match seed?
# Seed is np.ones(N). If we use L2 normalization, seed is not unit norm.
# Let's assume the "upper bound" is scale-invariant or we should normalize.
#
# Let's try: Minimize sum_{k=1}^{N-1} (sum_i h_i h_{i+k})^2 subject to sum h_i^2 = 1.
# This is a non-convex optimization problem.
#
# We can use gradient descent or L-BFGS-B.
#
# Let's define the objective function and its gradient.
# r[k] = <h, shift_k(h)>
# F(h) = sum_{k=1}^{N-1} r[k]^2
#
# Gradient:
# dF/dh_j = 2 * sum_{k=1}^{N-1} r[k] * (d r[k] / d h_j)
# r[k] = sum_i h_i h_{i+k}
# d r[k] / d h_j = h_{j+k} + h_{j-k} (with boundary handling)
#
# So grad_j = 2 * sum_{k=1}^{N-1} r[k] (h_{j+k} + h_{j-k})
#
# We can compute r via FFT for speed.
#
# Let's implement this optimization.
# We need to handle the constraint sum h^2 = 1.
# We can use Lagrange multipliers or simply project back to the sphere.
#
# Since the problem asks for a "signed step function", h can be negative.
#
# Let's start with a random initialization or the seed, and optimize.
#
# Deadline guard: 120s. We will run a few iterations of L-BFGS-B.

def _objective_and_grad(h):
    """
    h: array of shape (N,)
    Returns: objective value, gradient
    Objective: sum of squares of autocorrelation sidelobes
    Constraint: ||h||_2 = 1 (handled by projection or penalty, here we assume normalized input)
    """
    # Compute autocorrelation via FFT
    n = len(h)
    # Zero-pad to 2*n to avoid circular convolution effects
    h_pad = np.zeros(2 * n)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    R = np.fft.irfft(H * np.conj(H))
    
    # R[0] is the energy (sum h^2). R[k] for k>0 are the sidelobes.
    # We want to minimize sum_{k=1}^{n-1} R[k]^2
    sidelobes = R[1:n]
    obj = np.sum(sidelobes ** 2)
    
    # Gradient
    # grad_j = 2 * sum_{k=1}^{n-1} R[k] * (h_{j+k} + h_{j-k})
    # This is a convolution of R with a "V" shape kernel.
    # Let's compute it efficiently.
    # term_j = sum_{k=1}^{n-1} R[k] h_{j+k} + sum_{k=1}^{n-1} R[k] h_{j-k}
    # The first sum is a convolution of R with h.
    # Let's use FFT for the gradient as well.
    # dF/dh = 2 * (R * h) + 2 * (R_rev * h) ?
    # Actually, the gradient of sum_{k} R[k]^2 with respect to h is 2 * (R * h) in the circular sense?
    # No, it's linear autocorrelation.
    # Let's derive it carefully.
    # F = sum_{k=1}^{n-1} (sum_i h_i h_{i+k})^2
    # Let A_k = sum_i h_i h_{i+k}
    # dF/dh_j = 2 sum_{k=1}^{n-1} A_k (h_{j+k} + h_{j-k})
    # This is equivalent to 2 * (A * h) where * is linear convolution, but restricted to valid indices?
    # Actually, if we define the vector g such that g_j = sum_{k=1}^{n-1} A_k h_{j+k} + A_k h_{j-k}
    # This is a convolution of A (padded) with a kernel [1, 0, ..., 0, 1, 0, ..., 0]?
    # Let's just compute it directly for N=400, it's O(N^2) which is 160,000 operations, very fast.
    
    grad = np.zeros(n)
    # Vectorized approach:
    # Create a matrix M where M[k, j] = 1 if j+k < n else 0, etc.
    # Or just loop over k.
    for k in range(1, n):
        # h_{j+k} for j=0 to n-k-1
        # h_{j-k} for j=k to n-1
        # grad[0:n-k] += A_k * h[k:n]
        # grad[k:n] += A_k * h[0:n-k]
        grad[:n-k] += sidelobes[k-1] * h[k:]
        grad[k:] += sidelobes[k-1] * h[:n-k]
    grad *= 2
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Initial guess: random signs, normalized
    # Or start with seed? Seed is all ones.
    # Let's try a random initialization with random signs.
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_obj = np.inf
    
    # Try a few random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Random initialization
        h = np.random.randn(N)
        # Add some structure? Maybe alternating signs?
        # Let's just use random.
        h = h / np.linalg.norm(h)
        
        # Optimize
        try:
            res = minimize(
                _objective_and_grad,
                h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            if res.success:
                h_opt = res.x
                # Normalize
                h_opt = h_opt / np.linalg.norm(h_opt)
                obj, _ = _objective_and_grad(h_opt)
                
                if obj < best_obj:
                    best_obj = obj
                    best_h = h_opt
        except Exception:
            pass
    
    # If we didn't find anything better, return the seed (or a normalized version)
    # The seed is np.ones(N).
    # Let's check the objective of the seed.
    h_seed = np.ones(N)
    h_seed = h_seed / np.linalg.norm(h_seed)
    obj_seed, _ = _objective_and_grad(h_seed)
    
    if best_obj < obj_seed:
        return best_h
    else:
        return h_seed

# EVOLVE-BLOCK-END
