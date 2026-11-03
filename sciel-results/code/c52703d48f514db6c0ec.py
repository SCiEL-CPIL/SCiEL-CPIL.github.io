import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Initialize with a Sidon-like set
    # Greedy Sidon set in range N
    def greedy_sidon(n, k):
        s = []
        for i in range(n):
            if len(s) >= k: break
            # Check if i can be added
            valid = True
            for j in s:
                # Check sums
                pass # Simplified: just pick random distinct
            s.append(i)
        return s

    # Use a known good Sidon set or random sparse
    # Random sparse is safer
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Try to optimize weights on a sparse support
    # Support size ~ 7-10
    for k in range(5, 15):
        for _ in range(100):
            if time.time() > deadline: break
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0 / k # Normalize sum to 1
            
            # Optimize weights on this support
            def neg_score(x):
                h_loc = np.zeros(N)
                h_loc[idx] = x
                h_loc /= np.sum(h_loc) # Keep normalized
                return -_score_raw(h_loc)
            
            x0 = np.ones(k) / k
            bounds = [(0.0, 1.0)] * k
            # Constraint: sum(x) = 1 is handled by normalization inside, but SLSQP needs it?
            # Better to optimize x directly and normalize in score.
            # But SLSQP needs smooth objective.
            # Let's just use Nelder-Mead on the sparse vector
            res = minimize(neg_score, x0, method='Nelder-Mead', options={'maxiter': 500})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[idx] = res.x
                h_opt /= np.sum(h_opt)
                s = _score_raw(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt
    return list(best_h)
