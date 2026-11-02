"""Optimize 50 nonnegative step heights to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) with sum(h)=1."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    deadline = time.time() + 138.0  # Safety margin before 150s kill
    
    def compute_ratio(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        h = h / l1  # normalize to sum=1
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        conv_max = np.max(conv)
        if conv_max < 1e-15:
            return 0.0
        return l2_sq / conv_max
    
    def neg_log_ratio(h):
        r = compute_ratio(h)
        if r <= 1e-15:
            return 1e10
        return -np.log(r)
    
    def smooth_neg_log_ratio(h, beta=100.0):
        """Smooth surrogate using log-sum-exp for the max."""
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 1e10
        h = h / l1
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        # log-sum-exp for smooth max
        conv_max_val = np.max(conv)
        if conv_max_val < 1e-15:
            return 1e10
        shifted = conv / conv_max_val
        # log-sum-exp of shifted values
        m = np.max(shifted)
        lse = m + np.log(np.sum(np.exp(shifted - m)))
        # ratio = l2_sq / conv_max, log(ratio) = log(l2_sq) - log(conv_max)
        # Using smooth: log(ratio_smooth) = log(l2_sq) - log(conv_max) - lse + m
        # Actually simpler: we want to minimize -log(l2_sq) + log(conv_max_smooth)
        # conv_max_smooth = conv_max * exp(lse - m) = conv_max * exp(lse) since m=max
        # Hmm, let me just use the exact ratio for now and rely on SLSQP
        r = l2_sq / conv_max_val
        if r <= 1e-15:
            return 1e10
        return -np.log(r)
    
    # Known good Sidon sets (B2 sequences) within [0, 49]
    # Greedy Sidon: 0, 1, 3, 7, 12, 20, 30, 42
    # Let's try several support patterns
    
    def get_sidon_supports():
        supports = []
        # Classic greedy Sidon
        supports.append(np.array([0, 1, 3, 7, 12, 20, 30, 42]))
        # Another Sidon set
        supports.append(np.array([0, 1, 4, 10, 18, 29, 43]))
        # Larger Sidon-like sets
        supports.append(np.array([0, 1, 3, 7, 12, 20, 30, 42, 49]))
        # Spaced out
        supports.append(np.array([0, 2, 5, 9, 14, 20, 27, 35, 44]))
        # Very sparse
        supports.append(np.array([0, 1, 4, 10, 18, 29]))
        # Two-cluster
        supports.append(np.array([0, 1, 3, 7, 12, 20, 30, 42]))
        # Symmetric
        supports.append(np.array([0, 1, 4, 10, 18, 29, 43, 49]))
        return supports
    
    def optimize_on_support(support, h_init=None, max_iter=300):
        """Optimize amplitudes on a given support."""
        m = len(support)
        if m < 1:
            return None, 0.0
        
        # Parameterize: optimize log-amplitudes for positivity
        # h[support] = exp(a) / sum(exp(a))
        
        def neg_log_ratio_a(a):
            a = np.asarray(a)
            # Softmax for numerical stability
            a_max = np.max(a)
            weights = np.exp(a - a_max)
            weights = weights / np.sum(weights)
            h = np.zeros(N)
            h[support] = weights
            r = compute_ratio(h)
            if r <= 1e-15:
                return 1e10
            return -np.log(r)
        
        # Initial amplitudes: equal
        if h_init is None:
            a0 = np.zeros(m)
        else:
            h_supp = h_init[support]
            # Convert to log-amplitudes
            h_supp_pos = np.maximum(h_supp, 1e-10)
            a0 = np.log(h_supp_pos)
            a0 = a0 - np.max(a0)  # for numerical stability
        
        best_a = a0.copy()
        best_val = neg_log_ratio_a(a0)
        
        # Try SLSQP on the log-parameterization (no constraints needed)
        try:
            result = minimize(
                neg_log_ratio_a,
                a0,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_a = result.x
        except Exception:
            pass
        
        # Also try L-BFGS-B
        try:
            def grad_neg_log_ratio_a(a):
                eps = 1e-7
                g = np.zeros_like(a)
                for i in range(len(a)):
                    a_plus = a.copy()
                    a_plus[i] += eps
                    a_minus = a.copy()
                    a_minus[i] -= eps
                    g[i] = (neg_log_ratio_a(a_plus) - neg_log_ratio_a(a_minus)) / (2 * eps)
                return g
            
            result2 = minimize(
                neg_log_ratio_a,
                best_a,
                jac=grad_neg_log_ratio_a,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_a = result2.x
        except Exception:
            pass
        
        # Convert back to h
        a_max = np.max(best_a)
        weights = np.exp(best_a - a_max)
        weights = weights / np.sum(weights)
        h = np.zeros(N)
        h[support] = weights
        
        r = compute_ratio(h)
        return h, r
    
    def optimize_full(h0, max_iter=200):
        """Optimize the full 50-dim vector with sum constraint."""
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            result = minimize(
                neg_log_ratio,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )
            if result.success or result.fun < 1e9:
                h_opt = np.clip(result.x, 0, None)
                h_opt_sum = np.sum(h_opt)
                if h_opt_sum > 1e-15:
                    h_opt = h_opt / h_opt_sum
                r = compute_ratio(h_opt)
                return h_opt, r
        except Exception:
            pass
        return h0, compute_ratio(h0)
    
    best_h = np.ones(N) / N
    best_ratio = compute_ratio(best_h)
    
    # Phase 1: Optimize on various Sidon supports
    supports = get_sidon_supports()
    for support in supports:
        if time.time() > deadline - 30:
            break
        h_opt, r = optimize_on_support(support)
        if h_opt is not None and r > best_ratio:
            best_ratio = r
            best_h = h_opt.copy()
    
    # Phase 2: Full optimization from best result
    if time.time() < deadline - 20:
        h_opt, r = optimize_full(best_h.copy(), max_iter=200)
        if r > best_ratio:
            best_ratio = r
            best_h = h_opt.copy()
    
    # Phase 3: Local refinement with random perturbations
    if time.time() < deadline - 15:
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            h_pert = best_h + np.random.randn(N) * 0.005
            h_pert = np.clip(h_pert, 0, None)
            h_pert_sum = np.sum(h_pert)
            if h_pert_sum > 1e-15:
                h_pert = h_pert / h_pert_sum
                r = compute_ratio(h_pert)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_pert.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 10:
        h_opt, r = optimize_full(best_h.copy(), max_iter=500)
        if r > best_ratio:
            best_ratio = r
            best_h = h_opt.copy()
    
    # Ensure nonnegativity and normalization
    best_h = np.maximum(best_h, 0)
    s = np.sum(best_h)
    if s > 1e-15:
        best_h = best_h / s
    
    return list(best_h)
