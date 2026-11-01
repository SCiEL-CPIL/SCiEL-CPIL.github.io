"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from itertools import combinations

# Pre-compute pair indices
_pair_i, _pair_j = [], []
for i in range(N):
    for j in range(i + 1, N):
        _pair_i.append(i)
        _pair_j.append(j)
_pair_i = np.array(_pair_i)
_pair_j = np.array(_pair_j)
NP = len(_pair_i)


def _pairwise_sq(X):
    """Compute squared pairwise distances for all pairs in _pair_i, _pair_j."""
    di = X[_pair_i]
    dj = X[_pair_j]
    diff = di - dj
    return np.sum(diff * diff, axis=1)


def _score(X):
    """Compute (max/min)^2 of pairwise distances."""
    d = _pairwise_sq(X)
    dmin = np.min(d)
    dmax = np.max(d)
    if dmin < 1e-15:
        return np.inf
    return (dmax / dmin) ** 2


def _objective_smooth(X, t=20.0):
    """
    Smooth surrogate for max/min ratio using log-sum-exp.
    We minimize log(dmax) - log(dmin) approximately via soft-max and soft-min.
    Actually, to minimize (dmax/dmin)^2, we can minimize 2*(log(dmax) - log(dmin)).
    Using log-sum-exp for max: log(dmax) ~ logsumexp(d / t) * t  -- no, that's for exp.
    
    Better: 
    log(dmax) = logsumexp(d) - log(NP)  [if we use base e and large t]
    Actually, logsumexp(d) approximates log(sum(exp(d))) which is ~ dmax + log(count_of_max).
    So log(dmax) ≈ logsumexp(d) - log(NP) is a good smooth upper bound/surrogate.
    
    For min: log(dmin) = -logsumexp(-d) + log(NP).
    
    So objective = 2 * ( (logsumexp(d) - log(NP)) - (-logsumexp(-d) + log(NP)) )
                 = 2 * ( logsumexp(d) + logsumexp(-d) - 2*log(NP) )
    
    This is smooth and differentiable.
    """
    d = _pairwise_sq(X)
    # Numerical stability for logsumexp
    m = np.max(d)
    logsumexp_d = m + np.log(np.sum(np.exp(d - m)))
    
    m_neg = np.max(-d)
    logsumexp_neg_d = m_neg + np.log(np.sum(np.exp(-d - m_neg)))
    
    # log(dmax) approx = logsumexp_d - log(NP)
    # log(dmin) approx = -logsumexp_neg_d + log(NP)
    # Ratio log = log(dmax) - log(dmin) = logsumexp_d + logsumexp_neg_d - 2*log(NP)
    # We want to minimize (dmax/dmin)^2 = exp(2 * log_ratio)
    # But minimizing log_ratio is equivalent and smoother.
    # Let's just return 2 * log_ratio for stability, or exp if we want the actual value.
    # The task score is (max/min)^2. Minimizing the log is a good proxy.
    # However, to align with the "score" definition, let's return the actual value if possible,
    # but the smooth surrogate is for the optimizer.
    
    log_ratio = logsumexp_d + logsumexp_neg_d - 2.0 * np.log(NP)
    # Return a value that correlates with the score. 
    # Since exp is monotonic, minimizing log_ratio minimizes score.
    return 2.0 * log_ratio


def _grad_objective(X, t=20.0):
    """
    Gradient of the smooth objective.
    f(X) = 2 * (logsumexp(d) + logsumexp(-d) - 2*log(NP))
    df/dd_i = 2 * (exp(d_i - m) / sum(exp(d - m)) - exp(-d_i - m_neg) / sum(exp(-d - m_neg)))
             = 2 * (p_max_i - p_min_i)
    where p_max_i is the softmax of d, and p_min_i is the softmax of -d (which is 1/softmax(d) normalized? No, softmax(-d)).
    
    Then dX_k = sum_{pairs involving k} df/dd_pair * 2 * (x_k - x_j)
    """
    d = _pairwise_sq(X)
    m = np.max(d)
    exp_d = np.exp(d - m)
    sum_exp_d = np.sum(exp_d)
    p_max = exp_d / sum_exp_d
    
    m_neg = np.max(-d)
    exp_neg_d = np.exp(-d - m_neg)
    sum_exp_neg_d = np.sum(exp_neg_d)
    p_min = exp_neg_d / sum_exp_neg_d
    
    # df_dd = 2 * (p_max - p_min)
    df_dd = 2.0 * (p_max - p_min)
    
    grad = np.zeros_like(X)
    # For each pair (i, j), contribution to grad[i] is df_dd * 2 * (x_i - x_j)
    # Contribution to grad[j] is df_dd * 2 * (x_j - x_i)
    
    # Vectorized accumulation
    # grad[i] += sum_{j > i} df_dd_ij * 2 * (x_i - x_j) + sum_{j < i} df_dd_ji * 2 * (x_i - x_j)
    # This is easier with loops for clarity given small N, but let's try vectorized.
    
    # Let's use a simple loop for gradient accumulation as N=14 is small.
    for k in range(NP):
        i = _pair_i[k]
        j = _pair_j[k]
        val = 2.0 * df_dd[k]
        diff = X[i] - X[j]
        grad[i] += val * diff
        grad[j] -= val * diff
        
    return grad


def _center(X):
    """Center the point cloud at the origin."""
    return X - np.mean(X, axis=0)


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial guess: Cube-like structure
    # 14 points. 3x3x3 cube has 27 points.
    # Let's place them on a grid and then optimize.
    
    best_X = None
    best_score = np.inf
    
    # Strategy 1: Random restarts with smooth optimization
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initialization
        X = rng.normal(0, 1, (N, D))
        
        # Optimize using L-BFGS-B with the smooth objective
        try:
            res = minimize(
                _objective_smooth,
                X,
                jac=_grad_objective,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            X_opt = res.x
            X_opt = _center(X_opt)
            
            score = _score(X_opt)
            if score < best_score:
                best_score = score
                best_X = X_opt.copy()
                
        except Exception:
            continue
            
    # Strategy 2: Polish the best result with direct coordinate optimization on the true objective
    # using a more precise local solver if time permits.
    if best_X is not None and time.time() < deadline - 10:
        # Use Nelder-Mead on the true (non-smooth) objective for final polish
        # Or use L-BFGS-B with subgradient? No, it's not differentiable.
        # Nelder-Mead is derivative-free and robust.
        try:
            res = minimize(
                _score,
                best_X,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            X_polished = res.x
            X_polished = _center(X_polished)
            
            score = _score(X_polished)
            if score < best_score:
                best_score = score
                best_X = X_polished.copy()
        except Exception:
            pass
            
    if best_X is None:
        # Fallback: Grid
        side = int(np.ceil(N ** (1 / D)))
        grid = []
        for x in range(side):
            for y in range(side):
                for z in range(side):
                    if len(grid) < N:
                        grid.append([x, y, z])
        best_X = np.array(grid, dtype=float)
        
    return best_X

# EVOLVE-BLOCK-END
