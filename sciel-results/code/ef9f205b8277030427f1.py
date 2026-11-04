import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    return l2_sq / max_val

def _score_neg(h):
    return -_score(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    best_score = 1.0
    best_h = np.ones(N) / N

    # 1. Try specific heuristic structures to get a good starting point
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for k in range(3, len(primes)+1):
        idx = primes[:k]
        h = np.zeros(N)
        h[idx] = 1.0 / k
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # 2. Random sparse search to find a diverse set of starting points
    np.random.seed(42)
    num_trials = 500
    for i in range(num_trials):
        if time.time() - start_time > deadline - 30:
            break
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0 / k
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        
        # Also try random weights
        w = np.random.rand(k)
        w = w / np.sum(w)
        h = np.zeros(N)
        h[idx] = w
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # 3. Local optimization using L-BFGS-B
    # We optimize h directly with bounds [0, inf) and a penalty for sum != 1 is not needed 
    # if we normalize inside the objective, but L-BFGS-B works better with fixed scale or 
    # we can just let it explore. Since the objective is scale invariant (for h>=0), 
    # the optimizer might drift. To prevent this, we can fix the sum by parameterizing 
    # h = softmax(x) or simply normalize h in the objective and hope the optimizer 
    # stays in a good region. 
    # A better approach: parameterize h = exp(x) / sum(exp(x)) to ensure positivity and sum=1.
    # But L-BFGS-B requires bounds. Let's use the direct h with bounds [0, 1] and 
    # normalize in the objective.
    
    def objective(h):
        if time.time() - start_time > deadline - 5:
            return 1e10  # Penalty to stop
        return -_score(h)

    bounds = [(0.0, 1.0)] * N
    
    # Run optimization from the best found so far
    try:
        res = minimize(objective, best_h, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'maxfun': 10000})
        if res.fun < -best_score:
            best_score = -res.fun
            best_h = res.x
    except Exception:
        pass

    # 4. Try optimization from a few other random good points
    np.random.seed(123)
    for i in range(5):
        if time.time() - start_time > deadline - 10:
            break
        k = np.random.randint(5, 12)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0 / k
        # Perturb
        h += np.random.rand(N) * 0.1
        h = np.maximum(h, 0)
        h /= np.sum(h)
        
        try:
            res = minimize(objective, h, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'maxfun': 5000})
            if res.fun < -best_score:
                best_score = -res.fun
                best_h = res.x
        except Exception:
            pass

    # Ensure non-negative and valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / np.sum(best_h)

    return best_h.tolist()
# EVOLVE-BLOCK-END
