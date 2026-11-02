import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Baseline: 2-element
    h2 = np.zeros(N)
    h2[0] = 2.0/3.0
    h2[1] = 1.0/3.0
    try_candidate(h2)
    
    # Baseline: uniform
    h_uni = np.ones(N) / N
    try_candidate(h_uni)
    
    # Gaussian mixture model approach
    # h(x) = max_i a_i * exp(-(x - c_i)^2 / (2 * sigma_i^2))
    # Parameters: for each Gaussian: (a_i, c_i, sigma_i)
    # Total params: 3 * K where K is number of Gaussians
    
    def gauss_mixture_to_heights(params, K):
        """Convert parameter vector to height vector."""
        h = np.zeros(N)
        for i in range(K):
            a = params[3*i]
            c = params[3*i + 1]
            sigma = max(0.01, params[3*i + 2])
            # x positions: equal intervals on [-1/4, 1/4]
            x = np.linspace(-0.25, 0.25, N)
            h = np.maximum(h, a * np.exp(-(x - c)**2 / (2 * sigma**2)))
        return h
    
    def negative_ratio_gauss(params, K):
        """Negative of the ratio for minimization."""
        h = gauss_mixture_to_heights(params, K)
        if np.sum(h) < 1e-15:
            return 1e10
        return -compute_score(h)
    
    # Try different numbers of Gaussians
    for K in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 10:
            break
        
        # Multiple random initializations
        for trial in range(15):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + K * 100 + trial)
            
            # Initialize parameters
            params = np.zeros(3 * K)
            for i in range(K):
                params[3*i] = np.random.uniform(0.5, 2.0)  # amplitude
                params[3*i + 1] = np.random.uniform(-0.2, 0.2)  # center
                params[3*i + 2] = np.random.uniform(0.02, 0.15)  # sigma
            
            # Use L-BFGS-B for constrained optimization
            result = minimize(
                negative_ratio_gauss,
                params,
                args=(K,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < -best_score:
                h_opt = gauss_mixture_to_heights(result.x, K)
                s = try_candidate(h_opt)
    
    # Also try some structured initializations refined with coordinate descent
    def coordinate_descent(h, max_iter=100):
        """Cyclic coordinate descent for maximizing the ratio."""
        h = h.copy()
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                best_local = compute_score(h)
                for delta in [-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]:
                    h[i] = max(0.0, old + delta)
                    s = compute_score(h)
                    if s > best_local:
                        best_local = s
                        improved = True
                if not improved:
                    h[i] = old
            if not improved:
                break
        return h
    
    # Try various structured initializations
    structures = []
    
    # Ramp functions
    for slope in [0.5, 1.0, 2.0]:
        h = np.linspace(0, slope, N)
        structures.append(h)
    
    # Symmetric bump
    center = N // 2
    for width in [5, 10, 15, 20]:
        h = np.zeros(N)
        for i in range(max(0, center-width), min(N, center+width+1)):
            h[i] = np.exp(-((i-center)**2) / (2*width**2))
        structures.append(h)
    
    # Periodic patterns
    for period in [2, 3, 4, 5]:
        h = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                h[i] = 1.0
        structures.append(h)
    
    # Power-law decay
    for alpha in [0.5, 1.0, 1.5]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i+1)**(-alpha)
        structures.append(h)
    
    # Refine each structure with coordinate descent
    for h_init in structures:
        if time.time() > deadline - 15:
            break
        h_init = np.asarray(h_init, dtype=float)
        if np.sum(h_init) < 1e-15:
            continue
        h_refined = coordinate_descent(h_init, max_iter=80)
        try_candidate(h_refined)
    
    # Final: try a more intensive coordinate descent on the best found
    if time.time() < deadline - 5:
        h_final = coordinate_descent(best_h, max_iter=300)
        try_candidate(h_final)
    
    return list(best_h)
