"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    # We want to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Fix ||h||_1 = 1. Maximize ||h||_2^2 / ||h*h||_inf.
    # Equivalent to minimizing ||h*h||_inf / ||h||_2^2.
    
    def objective(h):
        # h is a vector of length N
        # Compute autocorrelation
        # h*h is the convolution of h with itself
        # Since h is non-negative, the max is likely at the center or peaks
        # Full convolution length is 2*N - 1
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-10:
            return 1e10
        return max_conv / l2_sq

    def constraint_sum(h):
        return np.sum(h) - 1.0

    # Initial guesses
    initial_guesses = []
    
    # 1. Uniform
    initial_guesses.append(np.ones(N) / N)
    
    # 2. Sidon-like sparse
    sidon_indices = [0, 1, 3, 7, 15, 31, 48]
    h_sidon = np.zeros(N)
    for idx in sidon_indices:
        h_sidon[idx] = 1.0 / len(sidon_indices)
    initial_guesses.append(h_sidon)
    
    # 3. Random sparse (5 non-zeros)
    for _ in range(5):
        indices = np.random.choice(N, 5, replace=False)
        h_rand = np.zeros(N)
        h_rand[indices] = 0.2
        initial_guesses.append(h_rand)
        
    # 4. Random dense
    for _ in range(5):
        h_rand = np.random.rand(N)
        h_rand /= np.sum(h_rand)
        initial_guesses.append(h_rand)

    best_score = 0
    best_h = np.ones(N) / N

    for h0 in initial_guesses:
        # Use SLSQP for local optimization
        # Bounds: h_i >= 0
        bounds = [(0, 1) for _ in range(N)]
        cons = [{'type': 'eq', 'fun': constraint_sum}]
        
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success:
                # Calculate the actual ratio for the found h
                h_opt = res.x
                # Normalize to ensure L1 = 1 (numerical drift)
                l1 = np.sum(h_opt)
                if l1 > 0:
                    h_opt = h_opt / l1
                    conv = np.convolve(h_opt, h_opt, mode='full')
                    max_conv = np.max(conv)
                    l2_sq = np.sum(h_opt**2)
                    if max_conv > 0:
                        score = l2_sq / max_conv
                    else:
                        score = 0
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_h = h_opt
        except Exception:
            pass

    return best_h
# EVOLVE-BLOCK-END
