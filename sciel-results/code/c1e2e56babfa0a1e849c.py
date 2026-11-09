"""New program using Scipy optimization for linear overlap minimization."""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # We use a larger N to allow for smoother transitions
    # which can reduce peak overlap in linear shifts.
    N = 50
    w = 2.0 / N
    
    # Initial guess: A triangular wave or ramp might have lower peak autocorrelation
    # than a random or constant vector in linear shifts.
    # Let's start with a smooth "bump" or just a ramp.
    x = np.linspace(0, 1, N)
    h_init = 0.5 + 0.4 * np.sin(2 * np.pi * x) # Oscillatory
    # Ensure sum is N/2
    # The sum of sin over a period is 0, so mean is 0.5. Sum is N/2.
    # Clip to [0, 1]
    h_init = np.clip(h_init, 0.0, 1.0)
    # Re-normalize to ensure sum is exactly N/2
    h_init = h_init / h_init.sum() * (N / 2.0)
    h_init = np.clip(h_init, 0.0, 1.0)
    
    def calc_linear_overlap(h_vec):
        # Compute max linear overlap
        # Overlap(delta) = sum_i h[i] * (1 - h[i - delta]) for valid i
        # We consider shifts delta from 0 to N-1
        # h is defined on indices 0..N-1
        # 1-h is defined on indices 0..N-1 (values 1-h[j])
        # Shifted 1-h by delta: g[k] = 1 - h[k - delta] if k-delta in [0, N-1] else 0?
        # The problem says "overlap of h with shifted 1-h".
        # Usually this means integral over the intersection of supports.
        # Support of h: [0, 2]. Support of shifted 1-h: [delta, delta+2].
        # Intersection length: 2 - delta for delta in [0, 2].
        # In discrete steps:
        max_overlap = 0.0
        for delta in range(N):
            # h indices: 0..N-1
            # 1-h shifted by delta: index j in 1-h corresponds to index j+delta in h's frame?
            # Let's align them.
            # h[i] at position i*w.
            # 1-h[j] at position (j+delta)*w.
            # Overlap is sum over i where both exist.
            # i*w in [0, 2] and (j+delta)*w in [0, 2] with i=w? No.
            # The functions are:
            # f(x) = h[i] for x in [i*w, (i+1)*w)
            # g(x) = 1 - h[j] for x in [(j+delta)*w, (j+delta+1)*w)
            # We want sum over x f(x)g(x).
            # This is equivalent to summing over pairs (i, j) such that the intervals overlap.
            # Since they are step functions, if we assume the shift is an integer number of steps (delta * w),
            # the overlap at shift delta is sum_i h[i] * (1 - h[i - delta]) for valid i, i-delta.
            # Wait, if we shift 1-h by delta steps to the right:
            # 1-h[j] is at position j+delta.
            # h[i] is at position i.
            # They overlap at position i if i = j + delta => j = i - delta.
            # So term is h[i] * (1 - h[i - delta]).
            # Valid i: 0 <= i < N and 0 <= i - delta < N.
            # So i ranges from max(0, delta) to min(N, N + delta).
            # Actually, since both are length N, the valid range for i is [delta, N-1] if delta > 0.
            # If delta = 0, i in [0, N-1].
            
            if delta == 0:
                terms = h_vec * (1 - h_vec)
            else:
                # h[i] for i in delta..N-1
                # h[i-delta] for i in delta..N-1
                h_part = h_vec[delta:]
                h_shifted_part = h_vec[:N-delta]
                terms = h_part * (1 - h_shifted_part)
            
            overlap_sum = np.sum(terms)
            if overlap_sum > max_overlap:
                max_overlap = overlap_sum
        
        return max_overlap * w

    # To make it differentiable, we use LogSumExp of the overlaps
    def objective(h_vec):
        overlaps = np.zeros(N)
        for delta in range(N):
            if delta == 0:
                terms = h_vec * (1 - h_vec)
            else:
                h_part = h_vec[delta:]
                h_shifted_part = h_vec[:N-delta]
                terms = h_part * (1 - h_shifted_part)
            overlaps[delta] = np.sum(terms) * w
        
        # LogSumExp to approximate max
        # scale by a temperature parameter for smoothness
        temp = 0.05
        m = np.max(overlaps)
        return m + temp * np.log(np.sum(np.exp((overlaps - m) / temp)))
        
    # Constraints: sum(h) = N/2, 0 <= h <= 1
    # We will use L-BFGS-B with bounds.
    # To handle the equality constraint sum(h)=N/2, we can project or use a constrained solver.
    # L-BFGS-B doesn't handle equality constraints directly well without transformation.
    # We can optimize over N-1 variables and fix the last one?
    # Or just penalize the sum in the objective?
    # Better: Use `scipy.optimize.minimize` with `method='SLSQP'` which handles equality constraints.
    
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0})
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    res = minimize(objective, h_init, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 100, 'ftol': 1e-10})
    
    h_opt = res.x
    # Final projection to ensure constraints are met strictly
    h_opt = np.clip(h_opt, 0.0, 1.0)
    h_opt = h_opt / h_opt.sum() * (N / 2.0)
    h_opt = np.clip(h_opt, 0.0, 1.0)
    
    return h_opt
# EVOLVE-BLOCK-END
