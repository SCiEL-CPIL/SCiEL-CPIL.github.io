import numpy as np
from scipy.optimize import minimize
from itertools import product

N = 50

# EVOLVE-BLOCK-START
def _autoconv_max(f):
    """Compute max of linear autoconvolution."""
    conv = np.convolve(f, f, mode='full')
    return np.max(conv)

def _score(f):
    """Compute the ratio ||f||_2^2 / (||f||_1 * max(f*f))."""
    f = np.asarray(f, dtype=float)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(f**2)
    amax = _autoconv_max(f)
    if amax < 1e-12:
        return 0.0
    return l2_sq / (l1 * amax)

def _normalize(f):
    """Normalize to L1=1, ensuring non-negativity."""
    f = np.maximum(f, 0)
    s = np.sum(f)
    if s < 1e-12:
        return np.ones(N) / N
    return f / s

def _simulated_annealing(f_init, n_iter=20000, T0=1.0, alpha=0.9999, beta=0.01):
    """
    Simulated Annealing on the simplex.
    f_init: initial vector (will be normalized)
    n_iter: number of steps
    T0: initial temperature
    alpha: cooling factor
    beta: step size parameter (fraction of mass to move)
    """
    f = _normalize(f_init)
    current_score = _score(f)
    best_f = f.copy()
    best_score = current_score
    
    T = T0
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        # Pick two indices to swap mass
        i_idx, j_idx = rng.choice(N, 2, replace=False)
        
        # Determine direction: move mass from i to j or j to i
        # We want to explore the simplex. 
        # Let's move a small fraction of the mass of the larger one to the smaller one? 
        # Or just random perturbation.
        # Perturbation: f_new = f + delta * (e_j - e_i)
        # We need f_new >= 0.
        
        # Let's pick a random epsilon
        # epsilon is a fraction of the total mass, or fraction of f[i]?
        # Using fraction of total mass (1/N) is a good scale.
        # Let's try moving a random fraction of the mass at i to j.
        
        if f[i_idx] < 1e-9:
            continue
            
        # Random step size
        # We want to allow large moves early, small moves late.
        # step_frac is between 0 and 1.
        step_frac = rng.uniform(0.1, 1.0)
        delta = f[i_idx] * step_frac
        
        # Try moving delta from i to j
        f_trial = f.copy()
        f_trial[i_idx] -= delta
        f_trial[j_idx] += delta
        
        # Check non-negativity (should be okay if delta <= f[i])
        if f_trial[i_idx] < 0:
            continue
            
        s_trial = _score(f_trial)
        
        # Metropolis criterion
        if s_trial > current_score:
            f = f_trial
            current_score = s_trial
            if s_trial > best_score:
                best_score = s_trial
                best_f = f.copy()
        else:
            # Accept with probability exp((s_trial - current_score) / T)
            # Note: scores are positive. 
            # Difference is negative.
            diff = s_trial - current_score
            prob = np.exp(diff / T)
            if rng.random() < prob:
                f = f_trial
                current_score = s_trial
                if s_trial > best_score:
                    best_score = s_trial
                    best_f = f.copy()
        
        T *= alpha
        if T < 1e-6:
            T = 1e-6
            
    return best_f, best_score

def _local_polish(f, max_iter=50):
    """
    Coordinate-wise line search to polish the result.
    """
    f = _normalize(f)
    best_score = _score(f)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            old_val = f[i]
            if old_val < 1e-9:
                continue
                
            # Try varying f[i] by scaling it, then renormalizing?
            # No, that changes all others. 
            # Better: Transfer mass from i to others or others to i?
            # Simple approach: Try setting f[i] to 0, or doubling it, etc.
            # Let's do a fine grid search on f[i] while keeping sum=1.
            # This is hard because changing f[i] requires adjusting others.
            
            # Alternative: Nelder-Mead on the full vector is too slow for 50D in tight loop?
            # Let's stick to a simpler local move: 
            # Pick a random j, move mass between i and j.
            # This is basically SA with T=0 (greedy).
            
            # Let's just do a few random greedy moves.
            pass 
            
        # If no improvement, break? 
        # Actually, let's just return f. The SA is the main driver.
        # We can add a simple "peel" operation: 
        # If a component is very small, set it to 0 and renormalize.
        # If a component is very large, maybe split it?
        
        # Let's try a simple gradient-free local search:
        # For each i, try moving a small amount to a random j.
        # If it improves, accept.
        
        any_improvement = False
        for i in range(N):
            if f[i] < 1e-6:
                continue
            # Try moving 10% of f[i] to a random j
            j = np.random.randint(N)
            if i == j: continue
            delta = f[i] * 0.1
            f_trial = f.copy()
            f_trial[i] -= delta
            f_trial[j] += delta
            s_trial = _score(f_trial)
            if s_trial > best_score + 1e-8:
                f = f_trial
                best_score = s_trial
                any_improvement = True
        if not any_improvement:
            break
            
    return f, best_score

def construct():
    np.random.seed(42)
    best_f = None
    best_score = 0.0
    
    # Initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussian centered
    x = np.arange(N)
    mu = 24.5
    sigma = 5.0
    g = np.exp(-(x-mu)**2 / (2*sigma**2))
    inits.append(g / np.sum(g))
    
    # 3. Triangular
    t = np.zeros(N)
    center = 24.5
    width = 10.0
    for i in range(N):
        d = abs(i - center)
        if d < width:
            t[i] = 1 - d/width
    inits.append(t / np.sum(t))
    
    # 4. Random Sparse
    for _ in range(5):
        r = np.zeros(N)
        idx = np.random.choice(N, 10, replace=False)
        r[idx] = np.random.rand(10)
        inits.append(r / np.sum(r))
        
    # 5. Random Dense
    for _ in range(5):
        r = np.random.rand(N)
        inits.append(r / np.sum(r))

    # Run SA on each
    for f_init in inits:
        f_sa, s_sa = _simulated_annealing(f_init, n_iter=5000, T0=0.1, alpha=0.9998)
        if s_sa > best_score:
            best_score = s_sa
            best_f = f_sa
            
        # Polish the SA result
        f_pol, s_pol = _local_polish(f_sa, max_iter=20)
        if s_pol > best_score:
            best_score = s_pol
            best_f = f_pol

    # Final safety: if best_f is None or score is 0, return uniform
    if best_f is None or best_score < 1e-6:
        best_f = np.ones(N) / N
        best_score = _score(best_f)
        
    return list(best_f)
# EVOLVE-BLOCK-END
