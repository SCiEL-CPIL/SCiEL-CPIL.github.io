import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
NUM_CONV = 2 * N - 1

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    np.random.seed(42)
    
    # Strategy: Focus on finding good SUPPORT SETS (which indices are nonzero)
    # For a given support of size k, the optimal weights can be found via
    # convex optimization. The key insight is that the ratio depends on
    # the autocorrelation structure, so we want supports where the
    # autocorrelation peaks are spread out.
    
    def optimize_weights_for_support(support, maxiter=100):
        """Given a set of active indices, optimize the weights."""
        k = len(support)
        if k < 2:
            return None
        
        # Use log-barrier + projected gradient in log-space
        # Minimize -log(ratio) = -log(l2_sq) + log(l1) + log(max_conv) + log(DELTA)
        # Subject to h >= 0 on support, sum(h) = 1
        
        # Simple approach: use SLSQP with equality constraint sum=1
        def obj(x):
            h_full = np.zeros(N)
            h_full[support] = x
            l1 = np.sum(h)
            if l1 <= 1e-15:
                return 1e10
            l2_sq = np.sum(h**2)
            conv = np.convolve(h_full, h_full, mode='full')
            mc = np.max(conv)
            if mc <= 1e-15:
                return 1e10
            return -(l2_sq / (l1 * DELTA * mc))
        
        def obj_grad(x):
            h_full = np.zeros(N)
            h_full[support] = x
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            conv = np.convolve(h_full, h_full, mode='full')
            mc = np.max(conv)
            # Numerical gradient
            grad = np.zeros(k)
            eps = 1e-7
            base = -obj(x)  # this is the actual score we want to maximize
            for i in range(k):
                xp = x.copy()
                xp[i] += eps
                hp = np.zeros(N)
                hp[support] = xp
                lp1 = np.sum(hp)
                if lp1 <= 1e-15:
                    grad[i] = 0
                    continue
                l2p = np.sum(hp**2)
                convp = np.convolve(hp, hp, mode='full')
                mcp = np.max(convp)
                if mcp <= 1e-15:
                    grad[i] = 0
                    continue
                sp = l2p / (lp1 * DELTA * mcp)
                grad[i] = (sp - base) / eps
            return -grad  # negative because we're minimizing -score
        
        # Initialize with uniform
        x0 = np.ones(k) / k
        bounds = [(0, 1)] * k
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1, 
                       'jac': lambda x: np.ones(k)}]
        
        try:
            res = opt.minimize(obj, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints, jac=obj_grad,
                              options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.success or res.fun < 1e9:
                h_full = np.zeros(N)
                h_full[support] = np.clip(res.x, 0, None)
                if np.sum(h_full) > 1e-15:
                    h_full /= np.sum(h_full)
                    return h_full
        except Exception:
            pass
        return None
    
    # Generate diverse support sets
    support_candidates = []
    
    # 1. Random subsets of various sizes
    for trial in range(80):
        k = np.random.randint(5, 25)
        support = np.random.choice(N, k, replace=False)
        support_candidates.append(support)
    
    # 2. Structured supports: arithmetic progressions (Sidon-like)
    for start in range(0, N, 3):
        for step in [2, 3, 4, 5, 7]:
            support = list(range(start, N, step))
            if len(support) >= 3:
                support_candidates.append(np.array(support))
    
    # 3. Supports with large gaps (to spread autocorrelation)
    for _ in range(30):
        k = np.random.randint(5, 15)
        # Choose indices with minimum gap
        min_gap = np.random.randint(2, 8)
        support = []
        for i in range(k):
            if i == 0:
                support.append(np.random.randint(0, max(1, N - k * min_gap)))
            else:
                prev = support[-1]
                lo = prev + min_gap
                hi = N - 1
                if lo >= hi:
                    break
                support.append(np.random.randint(lo, hi))
        if len(support) >= 3:
            support_candidates.append(np.array(support))
    
    # 4. Pairs and small sets
    for _ in range(20):
        k = np.random.randint(2, 5)
        support = np.random.choice(N, k, replace=False)
        support_candidates.append(support)
    
    # Optimize weights for each support
    for support in support_candidates:
        if time.time() > deadline - 30:
            break
        
        h_candidate = optimize_weights_for_support(support, maxiter=80)
        if h_candidate is not None:
            s = _score_raw(h_candidate)
            if s > best_score:
                best_score = s
                best_h = h_candidate.copy()
    
    # Local refinement: try moving weight between nearby indices
    def local_refine(h, max_iter=20):
        h = h.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            improved = False
            conv = np.convolve(h, h, mode='full')
            M_val = np.max(conv)
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            current_score = l2_sq / (l1 * DELTA * M_val)
            
            # Try all pairs of indices
            best_gain = 1e-8
            best_move = None
            epsilon = 1e-3
            
            for i in range(N):
                if h[i] < epsilon * 0.5:
                    continue
                for j in range(N):
                    if i == j:
                        continue
                    h_new = h.copy()
                    h_new[i] -= epsilon
                    h_new[j] += epsilon
                    if h_new[i] < 0:
                        continue
                    
                    l2_sq_new = np.sum(h_new**2)
                    conv_new = np.convolve(h_new, h_new, mode='full')
                    M_new = np.max(conv_new)
                    if M_new <= 1e-15:
                        continue
                    score_new = l2_sq_new / (l1 * DELTA * M_new)
                    
                    if score_new > current_score + best_gain:
                        best_gain = score_new - current_score
                        best_move = (i, j)
            
            if best_move:
                i, j = best_move
                h[i] -= epsilon
                h[j] += epsilon
                h = np.clip(h, 0, None)
                h /= np.sum(h)
                improved = True
            else:
                # Try smaller epsilon
                epsilon *= 0.5
                if epsilon < 1e-6:
                    break
        
        return h
    
    if time.time() < deadline - 15:
        h_refined = local_refine(best_h, max_iter=15)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined
    
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) > 0:
        best_h /= np.sum(best_h)
    else:
        best_h = np.ones(N) / N

    return list(best_h)
