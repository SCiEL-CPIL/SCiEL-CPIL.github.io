import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    time_limit = 610.0  # Safety margin before 630s kill

    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
    V_m1 = X - Y
    V_0 = X
    V_1 = X + Y
    V_2 = X + 2*Y

    bin_idx_m1 = (V_m1 + 9).flatten()
    bin_idx_0 = V_0.flatten()
    bin_idx_1 = V_1.flatten()
    bin_idx_2 = V_2.flatten()

    num_bins_m1 = 19
    num_bins_0 = 10
    num_bins_1 = 19
    num_bins_2 = 28

    def compute_ratio_from_logP(logP_flat, beta=0.0):
        z = logP_flat - np.max(logP_flat)
        P_flat = np.exp(z)
        sum_P = np.sum(P_flat)
        if sum_P < 1e-300:
            return 0.0
        P_flat = P_flat / sum_P

        counts_m1 = np.bincount(bin_idx_m1, weights=P_flat, minlength=num_bins_m1)
        counts_0 = np.bincount(bin_idx_0, weights=P_flat, minlength=num_bins_0)
        counts_1 = np.bincount(bin_idx_1, weights=P_flat, minlength=num_bins_1)
        counts_2 = np.bincount(bin_idx_2, weights=P_flat, minlength=num_bins_2)

        def entropy(counts):
            c = counts[counts > 1e-15]
            if len(c) == 0:
                return 0.0
            return -np.sum(c * np.log2(c))

        H_m1 = entropy(counts_m1)
        H_0 = entropy(counts_0)
        H_1 = entropy(counts_1)
        H_2 = entropy(counts_2)

        if beta > 1e-6:
            # Soft-max via log-sum-exp
            # M = beta * log(sum(exp(H_i / beta)))
            H_vals = np.array([H_0, H_1, H_2])
            M = beta * logsumexp(H_vals / beta)
        else:
            M = max(H_0, H_1, H_2)
        
        if M < 1e-9:
            return 0.0
        return H_m1 / M

    def objective(logP_flat, beta=0.0):
        r = compute_ratio_from_logP(logP_flat, beta)
        return -r

    best_score = 0.0
    best_P = np.ones((10, 10)) / 100.0

    # Parametric ansatz: P(x,y) ∝ exp(-αx² - βy² - γxy)
    def logP_from_params(params):
        alpha, beta, gamma = params
        x_vals = np.arange(n).reshape(1, -1)  # (1, 10)
        y_vals = np.arange(n).reshape(-1, 1)  # (10, 1)
        logP = -alpha * x_vals**2 - beta * y_vals**2 - gamma * x_vals * y_vals
        return logP.flatten()

    # Generate initial points
    init_points = []
    
    # 1. Uniform
    init_points.append(np.zeros(100))

    # 2. Parametric grid search for good starts
    param_inits = []
    for alpha in np.linspace(-1.5, 1.5, 8):
        for beta in np.linspace(-1.5, 1.5, 8):
            for gamma in np.linspace(-2.0, 2.0, 10):
                param_inits.append((alpha, beta, gamma))
                
    # Evaluate parametric points with a smooth objective (beta=1.0) to find good basins
    scored_params = []
    for params in param_inits:
        if time.time() - start_time > time_limit - 100:
            break
        logP = logP_from_params(params)
        score = -objective(logP, beta=1.0)
        scored_params.append((score, params))

    scored_params.sort(key=lambda t: t[0], reverse=True)
    for score, params in scored_params[:15]:
        init_points.append(logP_from_params(params))

    # 3. Rank-1 structured points
    for la in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        for lb in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            logP = np.zeros(100)
            for y in range(10):
                for x in range(10):
                    logP[y*10+x] = la*x + lb*y
            init_points.append(logP)

    # 4. Diagonal concentration
    for c in range(-9, 10):
        logP = np.full(100, -20.0)
        for y in range(10):
            x = y + c
            if 0 <= x < 10:
                logP[y*10+x] = 0.0
        init_points.append(logP)

    # 5. Random points
    rng = np.random.default_rng(12345)
    for _ in range(30):
        init_points.append(rng.normal(0, 1.5, 100))

    # Annealing schedule for beta
    # Start with larger beta (smoother) to find global structure, then reduce
    betas = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.0]
    
    # We will run optimization for each beta, using the best result from the previous stage as a strong initialization
    current_best_logP = init_points[0] # Start with uniform
    
    # First, optimize the initial points with the first beta to find the best candidate
    # Then for subsequent betas, use that best candidate + jittered versions
    
    active_inits = init_points
    
    for i, beta in enumerate(betas):
        if time.time() - start_time > time_limit - 20:
            break
            
        # If this is not the first stage, we prioritize the best found so far
        # We create a new set of inits: the best current, plus some random perturbations of it
        if i > 0:
            # Keep the best few from the previous stage's results? 
            # Actually, we just track best_P. Let's use best_P's logP as a primary init.
            z_best = np.log(np.clip(best_P, 1e-300, None)).flatten()
            new_inits = [z_best]
            # Add jittered versions
            for _ in range(10):
                jitter = rng.normal(0, 0.1, 100)
                new_inits.append(z_best + jitter)
            # Also keep some original diverse inits if time permits, but usually focusing on the basin is better
            # Let's mix: 50% best, 50% diverse from original list
            active_inits = new_inits + init_points[:10]
        else:
            active_inits = init_points

        for init in active_inits:
            if time.time() - start_time > time_limit - 10:
                break
            try:
                # Use L-BFGS-B
                res = minimize(lambda x: objective(x, beta), init, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                # Evaluate the *true* objective (beta=0) for the result
                true_score = -objective(res.x, beta=0.0)
                
                if true_score > best_score:
                    best_score = true_score
                    z = res.x - np.max(res.x)
                    P_flat = np.exp(z)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
            except Exception:
                pass

    # Final polish with beta=0 (hard max) using the best found
    if time.time() - start_time < time_limit - 5:
        z = np.log(np.clip(best_P, 1e-300, None)).flatten()
        for trial in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            jitter = rng.normal(0, 0.02, 100)
            try:
                res = minimize(lambda x: objective(x, 0.0), z + jitter, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                score = -res.fun
                if score > best_score:
                    best_score = score
                    z2 = res.x - np.max(res.x)
                    P_flat = np.exp(z2)
                    P_flat = P_flat / np.sum(P_flat)
                    best_P = P_flat.reshape(10, 10)
                    z = z2
            except Exception:
                pass

    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / s

    return best_P
