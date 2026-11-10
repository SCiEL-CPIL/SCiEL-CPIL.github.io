# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # Safety margin

    # Pre-compute grid coordinates
    y, x = np.mgrid[0:n, 0:n]
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Pre-compute projections for slopes
    proj_m1 = x - y
    proj_0 = x
    proj_1 = x + y
    proj_2 = x + 2 * y

    def compute_entropies(P):
        """Compute H_{-1}, H_0, H_1, H_2 for a given distribution P."""
        offset = n - 1
        idx_m1 = (proj_m1 + offset).astype(np.int32).ravel()
        weights = P.ravel()
        counts_m1 = np.bincount(idx_m1, weights=weights, minlength=2 * n - 1)
        probs_m1 = counts_m1[counts_m1 > 0]
        H_m1 = -np.sum(probs_m1 * np.log2(probs_m1)) if len(probs_m1) > 0 else 0.0

        idx_0 = x.astype(np.int32).ravel()
        counts_0 = np.bincount(idx_0, weights=weights, minlength=n)
        probs_0 = counts_0[counts_0 > 0]
        H_0 = -np.sum(probs_0 * np.log2(probs_0)) if len(probs_0) > 0 else 0.0

        idx_1 = (x + y).astype(np.int32).ravel()
        counts_1 = np.bincount(idx_1, weights=weights, minlength=2 * n - 1)
        probs_1 = counts_1[counts_1 > 0]
        H_1 = -np.sum(probs_1 * np.log2(probs_1)) if len(probs_1) > 0 else 0.0

        idx_2 = (x + 2 * y).astype(np.int32).ravel()
        counts_2 = np.bincount(idx_2, weights=weights, minlength=3 * n - 1)
        probs_2 = counts_2[counts_2 > 0]
        H_2 = -np.sum(probs_2 * np.log2(probs_2)) if len(probs_2) > 0 else 0.0

        return H_m1, H_0, H_1, H_2

    # Start with uniform distribution as baseline
    P_uniform = np.ones((n, n)) / (n * n)
    H_m1_u, H_0_u, H_1_u, H_2_u = compute_entropies(P_uniform)
    M_u = max(H_0_u, H_1_u, H_2_u)
    best_ratio = H_m1_u / M_u if M_u > 0 else 0.0
    best_P = P_uniform.copy()

    # We will optimize over the full 100-dimensional space using a smarter parameterization.
    # Instead of a Gaussian, we use a log-linear model: P[y,x] = exp(a[y,x]) / Z
    # where a[y,x] is a 10x10 matrix of parameters.
    # To make this tractable, we can use a low-rank factorization or just optimize directly.
    # Given the small size (100 params), direct optimization with multiple restarts is feasible.
    # However, the landscape might be complex. Let's try optimizing the log-weights directly.
    
    # Alternative: Use a mixture of Gaussians? Or just stick to the Gaussian but with better search?
    # The current program uses a single Gaussian. The stagnation signal suggests a structural change.
    # Let's try a "diagonal" distribution plus a "correlation" term, or simply a more flexible param.
    
    # Let's try optimizing the log-probabilities directly, constrained to sum to 1.
    # This is a 99-dimensional optimization (since sum is fixed).
    # We can use scipy.optimize.minimize with method='L-BFGS-B' or 'Nelder-Mead'.
    # To avoid boundary issues, we can optimize unbounded parameters and map them to probabilities via softmax.
    # P = exp(theta) / sum(exp(theta))
    # This ensures P > 0 and sum(P) = 1.
    
    def objective(theta):
        if time.time() > deadline - 10:
            return 1000.0
        
        # Softmax to get probabilities
        # theta is a 100-dim vector
        theta_max = np.max(theta)
        exp_theta = np.exp(theta - theta_max)
        P = exp_theta / np.sum(exp_theta)
        
        # Reshape to 10x10
        P_2d = P.reshape((n, n))
        
        # Calculate entropies
        H_m1, H_0, H_1, H_2 = compute_entropies(P_2d)
        
        M = max(H_0, H_1, H_2)
        if M <= 1e-9:
            return 1000.0
            
        ratio = H_m1 / M
        return -ratio

    # Initial guesses
    # 1. Uniform: theta = 0
    # 2. Gaussian-like: theta = -0.5 * Q
    # 3. Random
    
    restarts = 20
    for i in range(restarts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            # Uniform
            p0 = np.zeros(100)
        elif i == 1:
            # Diagonal bias: favor x=y
            p0 = np.zeros(100)
            for j in range(n):
                p0[j*n+j] = -2.0  # Lower weight for off-diagonal? No, we want to maximize H_{-1}.
                # H_{-1} is high when x-y is spread out. Uniform is max for H_{-1} if independent.
                # But M = max(H_0, H_1, H_2). 
                # For uniform, H_0 = log2(10) ~ 3.32, H_1 ~ 4.32, H_2 ~ 4.75. M = 4.75.
                # H_{-1} ~ 4.32. Ratio ~ 0.91.
                # To increase ratio, we want to increase H_{-1} relative to M.
                # H_{-1} is max when x-y is uniform. This happens when X and Y are independent uniform.
                # But M is also determined by marginals.
                # If we make X and Y dependent, we might reduce M while keeping H_{-1} high?
                # Or increase H_{-1} further? No, H_{-1} <= log2(19) ~ 4.25.
                # Wait, range of x-y is -9..9 (19 values). Max entropy is log2(19) ~ 4.247.
                # Uniform gives H_{-1} = log2(100) - I(X;Y)? No.
                # For uniform on 10x10, X-Y takes values -9..9 with probs 1/100, 2/100, ..., 10/100, ..., 1/100.
                # This is not uniform. The entropy is less than log2(19).
                # To get H_{-1} close to log2(19), we want the distribution of X-Y to be uniform.
                # This can be achieved if X and Y are independent uniform.
                # Let's check: If X,Y independent uniform, P(X-Y=k) = sum_x P(X=x)P(Y=x-k).
                # This is convolution of two uniform distributions, which is triangular, not uniform.
                # So H_{-1} for independent uniform is < log2(19).
                # Can we make X-Y uniform?
                # We need a joint distribution such that the marginal of X-Y is uniform.
                # This is possible. For example, let Y be uniform, and X = Y + K where K is uniform independent of Y?
                # If K is uniform on 0..9, and Y uniform on 0..9, then X is not necessarily in 0..9.
                # We are constrained to grid 0..9.
                # So we need a distribution on the grid such that X-Y is uniform.
                # Let's try to construct one.
                # If we pick P[x,y] such that for each d, sum_{x-y=d} P[x,y] = 1/19.
                # This is a system of linear equations.
                # There are 19 constraints and 100 variables.
                # We can solve for a feasible P.
                # To maximize H_{-1}, we want X-Y to be as uniform as possible.
                # To minimize M, we want X, X+Y, X+2Y to have low entropy.
                # This means we want X, X+Y, X+2Y to be concentrated.
                # This seems contradictory.
                
            pass
        else:
            # Random initialization
            p0 = np.random.randn(100) * 0.5
        
        # Use Nelder-Mead for robustness
        res = minimize(objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-7})
        
        if res.fun < -best_ratio:
            # Reconstruct P
            theta_max = np.max(res.x)
            exp_theta = np.exp(res.x - theta_max)
            P_candidate = exp_theta / np.sum(exp_theta)
            P_candidate = P_candidate.reshape((n, n))
            
            H_m1_c, H_0_c, H_1_c, H_2_c = compute_entropies(P_candidate)
            M_c = max(H_0_c, H_1_c, H_2_c)
            if M_c > 1e-9:
                ratio_c = H_m1_c / M_c
                if ratio_c > best_ratio:
                    best_ratio = ratio_c
                    best_P = P_candidate

    # Final validation and cleanup
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = P_uniform

    return best_P

# EVOLVE-BLOCK-END
