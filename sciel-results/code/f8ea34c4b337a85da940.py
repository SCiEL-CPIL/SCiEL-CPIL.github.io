"""Second autocorrelation inequality - maximize ratio of squared L2 norm to L1 norm times sup norm of autoconvolution."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = time.time() + 100.0  # safety margin before 120s limit

def autoconvolution(h):
    """Compute autoconvolution of h (linear convolution)."""
    return np.convolve(h, h, mode='full')

def smooth_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def objective_and_constraints(h, beta=50.0):
    """
    We want to maximize: ||h||_2^2 / max_k (h*h)_k
    Subject to: sum(h) = 1, h >= 0
    
    Using epigraph: maximize t s.t. t * max_k(h*h)_k <= ||h||_2^2
    But this is non-convex. Instead, directly optimize the ratio with smooth max.
    """
    h = np.maximum(h, 0)  # clip negatives during optimization
    s = np.sum(h)
    if s < 1e-10:
        return 0.0
    h_norm = h / s  # normalize to sum=1
    
    conv = autoconvolution(h_norm)
    conv_max = smooth_max(conv, beta)
    l2_sq = np.sum(h_norm ** 2)
    
    if conv_max < 1e-10:
        return 0.0
    return l2_sq / conv_max

def construct():
    best_score = 0.0
    best_h = None
    
    # Try several initializations
    n_restarts = 30
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > DEADLINE:
            break
        
        # Various initialization strategies
        if restart == 0:
            # Uniform
            h0 = np.ones(N) / N
        elif restart == 1:
            # Peak in center
            h0 = np.zeros(N)
            center = N // 2
            for i in range(N):
                h0[i] = np.exp(-((i - center) ** 2) / (2 * 5.0))
            h0 /= np.sum(h0)
        elif restart == 2:
            # Two peaks
            h0 = np.zeros(N)
            for i in range(N):
                h0[i] = np.exp(-((i - 10) ** 2) / 4.0) + np.exp(-((i - 40) ** 2) / 4.0)
            h0 /= np.sum(h0)
        elif restart == 3:
            # Alternating
            h0 = np.ones(N)
            h0[1::2] = 0.5
            h0 /= np.sum(h0)
        elif restart == 4:
            # Linear ramp
            h0 = np.linspace(0.1, 1.0, N)
            h0 /= np.sum(h0)
        elif restart == 5:
            # Cosine window
            h0 = np.cos(np.pi * (np.arange(N) - N/2 + 0.5) / N) ** 2
            h0 = np.maximum(h0, 0)
            h0 /= np.sum(h0)
        elif restart == 6:
            # Gaussian narrow
            center = N // 2
            h0 = np.exp(-((np.arange(N) - center) ** 2) / 2.0)
            h0 /= np.sum(h0)
        elif restart == 7:
            # Sparse: only a few non-zero
            h0 = np.zeros(N)
            idx = rng.choice(N, size=5, replace=False)
            h0[idx] = 1.0
            h0 /= np.sum(h0)
        elif restart == 8:
            # Random
            h0 = rng.dirichlet(np.ones(N))
        else:
            # Perturbed best or random
            if best_h is not None and rng.random() < 0.5:
                h0 = best_h + rng.normal(0, 0.05, N)
                h0 = np.maximum(h0, 0)
                s = np.sum(h0)
                if s > 1e-10:
                    h0 /= s
                else:
                    h0 = rng.dirichlet(np.ones(N))
            else:
                h0 = rng.dirichlet(np.ones(N) * 0.5)
        
        # Evaluate initial
        score0 = objective_and_constraints(h0, beta=50.0)
        
        # SLSQP optimization
        def neg_obj(h):
            return -objective_and_constraints(h, beta=50.0)
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0,
             'jac': lambda h: np.ones(N)}
        ]
        bounds = [(0.0, None)] * N
        
        try:
            res = minimize(neg_obj, h0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                h_opt = np.maximum(res.x, 0)
                s = np.sum(h_opt)
                if s > 1e-10:
                    h_opt /= s
                    score_opt = objective_and_constraints(h_opt, beta=50.0)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
        
        # Polish with higher beta (sharper max approximation)
        if best_h is not None:
            if time.time() > DEADLINE:
                break
            h_polish = best_h.copy()
            for beta in [100.0, 200.0, 500.0]:
                def neg_obj_p(h):
                    return -objective_and_constraints(h, beta=beta)
                try:
                    res_p = minimize(neg_obj_p, h_polish, method='SLSQP', bounds=bounds,
                                   constraints=constraints,
                                   options={'maxiter': 100, 'ftol': 1e-14, 'disp': False})
                    if res_p.success:
                        h_polish = np.maximum(res_p.x, 0)
                        s = np.sum(h_polish)
                        if s > 1e-10:
                            h_polish /= s
                except Exception:
                    pass
            
            score_polish = objective_and_constraints(h_polish, beta=500.0)
            if score_polish > best_score:
                best_score = score_polish
                best_h = h_polish.copy()
    
    # Final answer: return the heights (scale doesn't matter for the ratio, but we return normalized)
    if best_h is not None:
        return list(best_h)
    else:
        return [1.0] * N
