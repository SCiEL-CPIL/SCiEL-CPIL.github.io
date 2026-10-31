"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 600

import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # 10s safety margin before 140s hard limit

    # We want to minimize the normalized maximum of the autoconvolution.
    # For a step function with heights a_i on N equal intervals of width h,
    # the function is f(x) = sum_i a_i * 1_{[x_i, x_{i+1})}(x).
    # The autoconvolution (f*f)(t) = integral f(t-s) f(s) ds.
    # For step functions, this is a piecewise linear function.
    # The maximum of f*f is what we want to minimize, normalized by (integral f)^2.
    # Since the problem is scale-invariant, we can fix sum(a_i) = 1 and minimize max(f*f).
    # Actually, the normalization is max(f*f) / (integral f)^2.
    # If we scale f by c, max(f*f) scales by c^2 and (integral f)^2 scales by c^2, so the ratio is invariant.
    # So we fix sum(a_i) = 1 and minimize max(f*f).

    # The autoconvolution of a step function:
    # f(x) = a_i for x in [i*h, (i+1)*h), i = 0, ..., N-1, where h = (1/2)/N = 1/(2N).
    # (f*f)(t) = integral f(t-s) f(s) ds
    # For t in [k*h, (k+1)*h), the value is sum over i,j such that (i-j)*h <= t < (i-j+1)*h... 
    # Actually, for discrete step functions, the autoconvolution at point t depends on how the intervals overlap.
    # More precisely, if f is piecewise constant with values a_i on interval I_i,
    # then (f*f)(t) = sum_{i,j} a_i a_j * length(I_i ∩ (t - I_j))
    # For equal-width intervals, this is a convolution of the sequence a with itself, sampled appropriately.
    
    # Actually, for step functions on equal intervals, the autoconvolution is:
    # (f*f)(t) = h * sum_{i,j} a_i a_j * w_{ij}(t)
    # where w_{ij}(t) is the length of intersection of [i*h, (i+1)*h] and [t - (j+1)*h, t - j*h].
    
    # The maximum of (f*f) occurs at some t. Due to the structure, the maximum will be at one of the "peaks" 
    # which correspond to the discrete convolution of a with itself, scaled by h.
    # Specifically, (f*f)(k*h) for integer k is h * sum_i a_i a_{i-k} (with appropriate boundary handling).
    # But actually, the maximum might occur between the sample points.
    
    # Let me think more carefully. For a step function f(x) = sum_i a_i 1_{[i*h, (i+1)*h)}(x),
    # the autoconvolution is:
    # (f*f)(t) = integral_{-infty}^{infty} f(t-s) f(s) ds
    # = sum_i sum_j a_i a_j * length([i*h, (i+1)*h) ∩ [t - (j+1)*h, t - j*h))
    
    # For t in [m*h, (m+1)*h) for some integer m, the intersection length is:
    # For each pair (i,j), the interval [i*h, (i+1)*h) intersects [t-(j+1)*h, t-j*h) 
    # in an interval of length that depends on t.
    
    # Actually, a cleaner way: (f*f)(t) is a piecewise linear function. 
    # On each interval [m*h, (m+1)*h), it is linear.
    # The maximum on each such interval is at one of the endpoints.
    # So the global maximum is at some t = m*h for integer m.
    
    # Therefore, max(f*f) = max_m (f*f)(m*h)
    # And (f*f)(m*h) = h * sum_i a_i a_{i-m} where a_k = 0 for k < 0 or k >= N.
    # Wait, let me verify: (f*f)(m*h) = sum_i sum_j a_i a_j * length([i*h, (i+1)*h) ∩ [m*h - (j+1)*h, m*h - j*h))
    # = sum_i sum_j a_i a_j * length([i*h, (i+1)*h) ∩ [(m-j-1)*h, (m-j)*h))
    # This is h if i = m-j, i.e., j = m-i, and 0 otherwise? No, that's not right.
    # The intersection of [i*h, (i+1)*h) and [(m-j-1)*h, (m-j)*h) is non-empty only if these intervals overlap.
    # [i*h, (i+1)*h) overlaps [(m-j-1)*h, (m-j)*h) iff i*h < (m-j)*h and (i+1)*h > (m-j-1)*h
    # iff i < m-j and i+1 > m-j-1 iff i < m-j and i >= m-j.
    # This is only possible if i = m-j. So the intersection is [i*h, (i+1)*h) ∩ [(m-j-1)*h, (m-j)*h) = [i*h, (i+1)*h) when i = m-j.
    # So length = h when j = m-i, and 0 otherwise.
    # Therefore (f*f)(m*h) = h * sum_i a_i a_{m-i} = h * (a * a)[m]
    # where (a*a) is the discrete cross-correlation (or convolution with index reversal).
    
    # So max(f*f) = h * max_m (a * a)[m]
    # And (integral f)^2 = (h * sum_i a_i)^2 = h^2 * (sum a_i)^2
    # So the normalized maximum is:
    # M = max(f*f) / (integral f)^2 = [h * max_m (a*a)[m]] / [h^2 * (sum a_i)^2] = max_m (a*a)[m] / [h * (sum a_i)^2]
    
    # Since we fix sum a_i = 1, M = max_m (a*a)[m] / h
    # We want to minimize max_m (a*a)[m] subject to a_i >= 0 and sum a_i = 1.
    
    # The discrete convolution (a*a)[m] = sum_i a_i a_{i-m} for m in range(-N+1, N).
    # Note: (a*a)[0] = sum_i a_i^2, which is minimized when a is uniform, giving 1/N.
    # But other lags might be larger.
    
    # We want to minimize max_m |sum_i a_i a_{i-m}|. Since a_i >= 0, the convolution is non-negative, so we minimize max_m (a*a)[m].
    
    # This is a convex optimization problem? The function max_m (a*a)[m] is convex in a (it's the max of linear functions).
    # The constraint a_i >= 0 and sum a_i = 1 is linear.
    # So we can use convex optimization.
    
    # Let's set up the problem:
    # Variables: a_0, ..., a_{N-1}
    # Objective: minimize t
    # Constraints: t >= (a*a)[m] for all m = -(N-1), ..., N-1
    #              a_i >= 0 for all i
    #              sum a_i = 1
    
    # Number of variables: N (for a) + 1 (for t) = N+1
    # Number of constraints: 2N-1 (for the convolution) + N (non-negativity) + 1 (sum) = 3N
    
    # This is a linear program! We can solve it with scipy.optimize.linprog.
    
    # Let's construct the LP.
    # Variables: x = [a_0, a_1, ..., a_{N-1}, t]
    # Objective: minimize [0, 0, ..., 0, 1] @ x
    # Constraints:
    # 1. For each lag m in range(-(N-1), N): t - sum_i a_i a_{i-m} >= 0
    #    But this is not linear in x! Because it involves a_i a_{i-m}.
    
    # Oops, the objective is not linear. The function max_m (a*a)[m] is not linear, it's the max of quadratic functions.
    # So it's a quadratic constraint.
    
    # Alternative: since the problem is scale-invariant and we want to minimize the peak of the autocorrelation,
    # a good approach is to use gradient-based optimization on the smooth approximation.
    
    # Let's use the smooth max surrogate: smooth_max(x, beta) = (1/beta) * log(sum exp(beta * x_i))
    # This approximates max(x) for large beta.
    
    # We minimize f(a) = smooth_max((a*a)[m] for m in range(-(N-1), N), beta)
    # subject to a_i >= 0, sum a_i = 1.
    
    # We can use L-BFGS-B with a log barrier or just project onto the simplex.
    
    # Actually, let's use a different approach: parameterize a_i = exp(z_i) / sum_j exp(z_j)
    # This automatically satisfies a_i > 0 and sum a_i = 1.
    # Then we minimize f(z) = smooth_max((a(z)*a(z))[m] for m, beta)
    
    # The gradient can be computed numerically or analytically. Let's use numerical gradient with finite differences,
    # but that might be slow. Let's use L-BFGS-B which uses numerical gradients by default.
    
    # Actually, L-BFGS-B with finite differences on 600 variables might be slow. Each function evaluation requires
    # computing the full convolution, which is O(N^2) = 360,000 operations. That's fine.
    # But finite differences require 601 evaluations per gradient step, which is 601 * 360,000 = 216 million operations per step.
    # That might be too slow.
    
    # Let's use a smarter approach: we can compute the gradient analytically.
    # Let C_m = sum_i a_i a_{i-m} for m in range(-(N-1), N).
    # Let s = max_m C_m (approximated by smooth_max).
    # d s / d a_k = (sum_m exp(beta * C_m) * d C_m / d a_k) / (beta * sum_m exp(beta * C_m))
    # d C_m / d a_k = a_{k-m} + a_{k+m} (with appropriate boundary handling, i.e., a_j = 0 for j < 0 or j >= N)
    
    # This is doable but complex. Let's try a simpler approach first: use projected gradient descent on the simplex.
    
    # Actually, let's try a different strategy: use the fact that the optimal solution for minimizing the peak 
    # of the autocorrelation subject to non-negativity and fixed L1 norm is likely to be a "bang-bang" solution 
    # or a smooth solution. Let's try a few heuristic solutions and then refine with local optimization.
    
    # Heuristic 1: uniform distribution. a_i = 1/N for all i.
    # Then C_m = sum_i (1/N)(1/N) * (N - |m|) = (N - |m|) / N^2 for |m| < N.
    # Max C_m = 1/N (at m=0).
    # So M = (1/N) / h = (1/N) / (1/(2N)) = 2.
    # This matches the seed program's score of 1.0 (since the seed returns [1.0]*N, and the evaluator 
    # normalizes by (integral f)^2, so the raw value is 2, and the score is seed/raw = 1/2? No, the score is 
    # the upper bound on C1, and the search score is seed/value. The seed value is 2 (for the constant function), 
    # so the search score for the seed is 1.0. If we get value < 2, the search score > 1.0.)
    
    # Heuristic 2: cosine bump. a_i = cos^2(pi * (i + 0.5) / N) normalized to sum to 1.
    # This is a smooth bump that might have a flatter autocorrelation peak.
    
    # Let's compute the autocorrelation for the cosine bump and see if it's better than uniform.
    
    # Actually, let's just use gradient-based optimization. I'll implement a simple projected gradient method
    # on the simplex, using the smooth max surrogate with a moderate beta.
    
    # But first, let's try a simple approach: use scipy.optimize.minimize with L-BFGS-B on the reparameterized 
    # variables z, where a_i = exp(z_i) / sum exp(z_j).
    
    # To speed things up, I'll compute the convolution using FFT.
    
    def compute_conv(a):
        """Compute the discrete autocorrelation of a."""
        # Pad a to length 2N-1 to compute the full convolution.
        a_padded = np.zeros(2 * N - 1)
        a_padded[:N] = a
        conv = np.fft.ifft(np.abs(np.fft.fft(a_padded, n=2 * N - 1)) ** 2).real
        # The convolution is centered, so we need to extract the relevant part.
        # The lag m corresponds to index (m + N - 1) in the convolution output.
        # Actually, np.convolve(a, a, mode='full') gives a vector of length 2N-1, where index k corresponds to lag k - (N-1).
        # So for lag m in range(-(N-1), N), the index is m + (N-1).
        return conv
    
    def smooth_max_conv(a, beta):
        """Compute the smooth max of the autocorrelation."""
        conv = compute_conv(a)
        # Apply smooth max
        scaled = beta * conv
        # For numerical stability
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        smooth_max_val = (max_scaled + np.log(np.sum(exp_scaled))) / beta
        return smooth_max_val, conv
    
    # Objective function for L-BFGS-B on z variables.
    def objective(z):
        a = np.exp(z - np.max(z))
        a = a / np.sum(a)
        val, _ = smooth_max_conv(a, beta)
        return val
    
    # We need to compute the gradient analytically for efficiency.
    # Let a_i = exp(z_i) / S, where S = sum_j exp(z_j).
    # d a_i / d z_k = a_i * (delta_{ik} - a_k)
    # Let C_m = sum_i a_i a_{i-m}
    # d C_m / d z_k = sum_i (d a_i / d z_k) a_{i-m} + sum_i a_i (d a_{i-m} / d z_k)
    # = sum_i a_i (delta_{ik} - a_k) a_{i-m} + sum_i a_i a_{i-m} (delta_{i-m,k} - a_k)
    # = a_k a_{k-m} - a_k sum_i a_i a_{i-m} + a_k a_{k+m} - a_k sum_i a_i a_{i-m}
    # = a_k a_{k-m} + a_k a_{k+m} - 2 a_k C_m
    # (with a_j = 0 for j < 0 or j >= N)
    
    # The gradient of the smooth max:
    # d s / d a_k = (sum_m exp(beta * C_m) * d C_m / d a_k) / (beta * sum_m exp(beta * C_m))
    # where d C_m / d a_k = a_{k-m} + a_{k+m}
    
    # Then d s / d z_k = sum_i d s / d a_i * d a_i / d z_k = d s / d a_k * (delta_{kk} - a_k) ... wait, this is not right.
    # d s / d z_k = sum_i (d s / d a_i) * (d a_i / d z_k)
    # = sum_i (d s / d a_i) * a_i * (delta_{ik} - a_k)
    # = (d s / d a_k) * a_k - a_k * sum_i (d s / d a_i) * a_i
    
    # This is getting complex. Let's just use numerical gradients with a small step size, 
    # and hope that L-BFGS-B can handle it within the time budget.
    
    # Actually, let's try a different approach: use a projected subgradient method on the original a variables,
    # with a simple step size schedule. This might be slower but more robust.
    
    # Or, let's use a genetic algorithm / random search with local refinement.
    
    # Given the time constraints, let's try a simpler approach: 
    # 1. Start with a few good initial guesses (uniform, cosine, random).
    # 2. Use L-BFGS-B with numerical gradients on the a variables, with constraints a_i >= 0 and sum a_i = 1.
    # 3. To handle the simplex constraint, we can use the reparameterization a_i = exp(z_i) / sum exp(z_j).
    
    # Let's implement this carefully.
    
    beta = 50.0  # Smooth max parameter
    
    def objective_z(z):
        a = np.exp(z - np.max(z))
        a = a / np.sum(a)
        val, _ = smooth_max_conv(a, beta)
        return val
    
    # Initial guesses
    initial_guesses = []
    
    # 1. Uniform
    initial_guesses.append(np.zeros(N))
    
    # 2. Cosine bump
    i = np.arange(N)
    cos_bump = np.cos(np.pi * (i + 0.5) / N) ** 2
    cos_bump = cos_bump / np.sum(cos_bump)
    initial_guesses.append(np.log(cos_bump))
    
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        r = np.random.exponential(1.0, size=N)
        r = r / np.sum(r)
        initial_guesses.append(np.log(r))
    
    best_val = float('inf')
    best_a = None
    
    for z0 in initial_guesses:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective_z, z0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                a_best = np.exp(res.x - np.max(res.x))
                a_best = a_best / np.sum(a_best)
                best_a = a_best
        except Exception:
            pass
    
    if best_a is None:
        # Fallback to uniform
        best_a = np.ones(N) / N
    
    # Return the step heights. Since the problem is scale-invariant, we can return a_best (sum=1).
    # But let's double-check: the evaluator computes the raw value from the returned heights.
    # If we return a_best with sum=1, then integral f = h * sum a_i = h, and max(f*f) = h * max(C_m).
    # So M = max(f*f) / (integral f)^2 = h * max(C_m) / h^2 = max(C_m) / h.
    # This is the value we minimized. Good.
    
    return [float(v) for v in best_a]
# EVOLVE-BLOCK-END
