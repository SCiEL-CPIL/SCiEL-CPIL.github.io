import numpy as np
import time
import sys

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h||_inf * ||h*h||_inf)"""
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    l_inf = np.max(h)
    if l1 < 1e-15 or l_inf < 1e-15:
        return 0.0
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return (l2_sq) / (l1 * l_inf * max_conv)

def _optimize_coordinate(h, deadline):
    """
    Coordinate descent: for each coordinate i, find the optimal value
    by solving a 1D optimization problem.
    
    For fixed other coordinates, the ratio as a function of h[i] is:
    r(x) = (S2 + x^2) / ((S1 + x) * M(x) * C(x))
    
    where S2 = sum of h[j]^2 for j!=i, S1 = sum of h[j] for j!=i,
    M(x) = max(x, rest_max), C(x) = max of autoconvolution.
    
    This is complex, so we use a simpler approach: evaluate the ratio
    at a set of candidate values and pick the best.
    """
    best_h = h.copy()
    best_ratio = _compute_ratio(best_h)
    
    rest = h.copy()
    rest_max_excl = 0.0
    if np.max(h) > 0:
        # Find the index of the max to handle normalization properly
        # Actually, let's just try a range of values for each coordinate
        pass
    
    for i in range(N):
        if time.time() > deadline:
            break
        
        old_val = h[i]
        rest = h.copy()
        rest[i] = 0.0
        
        # Candidate values: 0, old_val, and a grid
        # Also try values scaled relative to the current max of rest
        rest_max = np.max(rest)
        candidates = [0.0, old_val]
        if rest_max > 0:
            for frac in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                candidates.append(rest_max * frac)
        
        best_val = old_val
        best_r = _compute_ratio(h)
        
        for c in candidates:
            if c < 0:
                continue
            trial = h.copy()
            trial[i] = c
            # Normalize to max 1
            if np.max(trial) > 1e-15:
                trial = trial / np.max(trial)
            r = _compute_ratio(trial)
            if r > best_r:
                best_r = r
                best_val = c
        
        if best_val != old_val:
            h[i] = best_val
            if np.max(h) > 1e-15:
                h = h / np.max(h)
            new_r = _compute_ratio(h)
            if new_r > best_ratio:
                best_ratio = new_r
                best_h = h.copy()
    
    return best_h, best_ratio

def _gaussian_mixture(h0, n_gaussians, deadline):
    """
    Parametric search: represent h as a mixture of shifted, scaled Gaussians.
    Optimize the parameters (centers, widths, amplitudes) using Nelder-Mead.
    """
    from scipy.optimize import minimize
    
    # Sample grid
    x = np.linspace(-0.25, 0.25, N)
    
    def make_h(params):
        h = np.zeros(N)
        for k in range(n_gaussians):
            amp = params[3*k]
            center = params[3*k+1]
            width = max(params[3*k+2], 0.01)
            h += amp * np.exp(-((x - center)**2) / (2 * width**2))
        h = np.maximum(h, 0)
        if np.max(h) > 1e-15:
            h = h / np.max(h)
        return h
    
    def objective(params):
        h = make_h(params)
        return -_compute_ratio(h)
    
    # Initialize parameters
    params0 = np.zeros(3 * n_gaussians)
    for k in range(n_gaussians):
        params0[3*k] = 1.0 / n_gaussians  # amplitude
        params0[3*k+1] = -0.25 + 0.5 * (k + 0.5) / n_gaussians  # center
        params0[3*k+2] = 0.05  # width
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        if time.time() > deadline:
            return h0
        h_opt = make_h(result.x)
        return h_opt
    except:
        return h0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Sidon set (B2 sequence)
    S = [0, 1, 4, 10, 18, 23, 30]
    h1 = np.zeros(N)
    for idx in S:
        h1[idx] = 1.0
    initializations.append(h1)
    
    # 2. Sparse random
    np.random.seed(42)
    h2 = np.zeros(N)
    for idx in np.random.choice(N, 7, replace=False):
        h2[idx] = 1.0
    initializations.append(h2)
    
    # 3. Gaussian bump
    x = np.linspace(-0.25, 0.25, N)
    h3 = np.exp(-((x - 0.0)**2) / (2 * 0.08**2))
    initializations.append(h3)
    
    # 4. Two bumps
    h4 = np.exp(-((x - (-0.1))**2) / (2 * 0.06**2)) + np.exp(-((x - 0.1)**2) / (2 * 0.06**2))
    initializations.append(h4)
    
    # 5. Uniform
    initializations.append(np.ones(N))
    
    # 6. Sparse comb
    h6 = np.zeros(N)
    h6[::7] = 1.0
    initializations.append(h6)
    
    best_h = None
    best_ratio = 0.0
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        h0 = h0.copy()
        if np.max(h0) > 0:
            h0 = h0 / np.max(h0)
        
        # Local search with random perturbations
        h = h0.copy()
        ratio = _compute_ratio(h)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
        
        # Coordinate descent iterations
        for _ in range(5):
            if time.time() > deadline:
                break
            h, ratio = _optimize_coordinate(h, deadline)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
        
        # Random perturbations to escape local minima
        for _ in range(200):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            h[i] = np.clip(h[i] + np.random.uniform(-0.2, 0.2), 0, 2)
            if np.max(h) > 1e-15:
                h = h / np.max(h)
            ratio = _compute_ratio(h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h.copy()
                # Do a few coordinate descent steps from here
                for _ in range(2):
                    if time.time() > deadline:
                        break
                    h, ratio = _optimize_coordinate(h, deadline)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_h = h.copy()
    
    # Try Gaussian mixture parametric search
    if time.time() < deadline - 30:
        for n_g in [2, 3, 4]:
            if time.time() > deadline - 10:
                break
            h_gauss = _gaussian_mixture(best_h, n_g, deadline - 10)
            if np.max(h_gauss) > 0:
                h_gauss = h_gauss / np.max(h_gauss)
            ratio = _compute_ratio(h_gauss)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_gauss.copy()
    
    # Final polish with coordinate descent
    if time.time() < deadline - 5:
        h_final, ratio_final = _optimize_coordinate(best_h, deadline - 2)
        if ratio_final > best_ratio:
            best_ratio = ratio_final
            best_h = h_final.copy()
    
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    if np.max(best_h) > 1e-15:
        best_h = best_h / np.max(best_h)
    
    return list(best_h)
