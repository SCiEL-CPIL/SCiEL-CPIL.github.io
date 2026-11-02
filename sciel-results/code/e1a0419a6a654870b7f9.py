import numpy as np
import time
from scipy.optimize import minimize
import warnings

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def compute_conv_grads(h, peak_indices):
        """
        Compute the gradient of the sum of conv[peak_indices] w.r.t h.
        conv[k] = sum_{j} h[j] h[k-j]
        d conv[k] / d h[i] = h[k-i] + h[i-k] (if indices valid)
        """
        grad = np.zeros(N)
        for k in peak_indices:
            for i in range(N):
                val = 0.0
                idx1 = k - i
                if 0 <= idx1 < N:
                    val += h[idx1]
                idx2 = i - k
                if 0 <= idx2 < N:
                    val += h[idx2]
                grad[i] += val
        return grad

    def multiplicative_optimize(h0, max_iters=2000, eta_start=0.1, eta_end=0.001):
        """
        Optimize using multiplicative updates on the simplex.
        Minimizes the peak autoconvolution while maintaining non-negativity and sum=1.
        """
        h = np.maximum(h0, 0).copy()
        s = np.sum(h)
        if s < 1e-10:
            return h0, 0.0
        h = h / s
        
        best_h = h.copy()
        best_s = score(h)
        
        # We will use a decaying learning rate
        # and a "top-k" peak selection to smooth the gradient
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            if max_conv < 1e-15:
                break
            
            # Select top K peaks to define the "cost"
            # Using more peaks makes the landscape smoother and less jagged
            K = min(5, len(conv))
            # Find indices of top K values
            # np.argpartition is faster for top K
            if K < len(conv):
                top_k_indices = np.argpartition(conv, -K)[-K:]
            else:
                top_k_indices = np.arange(len(conv))
            
            # Normalize the peaks for the gradient?
            # We want to reduce h where it contributes to these peaks.
            # The objective is to minimize max_conv. 
            # Using a sum of top-k peaks is a surrogate.
            
            grad = compute_conv_grads(h, top_k_indices)
            
            # Adaptive eta
            # Start high, decay to low
            progress = it / max_iters
            eta = eta_start * (eta_end / eta_start) ** progress
            
            # Multiplicative update
            # h_i <- h_i * exp(-eta * grad_i)
            # To prevent numerical issues, we can clip the exponent
            exp_arg = -eta * grad
            # Normalize exp_arg to avoid overflow/underflow? 
            # Actually, since we renormalize, we can just scale it.
            # Let's center the update to maintain sum approx 1 before normalization
            exp_arg_centered = exp_arg - np.mean(exp_arg)
            
            update = np.exp(exp_arg_centered)
            h = h * update
            
            # Renormalize to simplex
            s = np.sum(h)
            if s < 1e-15:
                # If it died, restart with small noise
                h = np.ones(N) / N
                continue
            h = h / s
            
            s_val = score(h)
            if s_val > best_s:
                best_s = s_val
                best_h = h.copy()
            
            # Early stop if converged
            if it > 100 and abs(s_val - best_s) < 1e-10:
                break
                
        return best_h, best_s

    def local_refine(h0):
        """
        Refine the solution using SLSQP on the non-zero support.
        """
        h = np.maximum(h0, 0).copy()
        s = np.sum(h)
        if s < 1e-10:
            return h0, 0.0
        h = h / s
        
        # Identify active support
        support = np.where(h > 1e-6)[0]
        if len(support) == 0:
            return h, score(h)
        if len(support) == 1:
            h_opt = np.zeros(N)
            h_opt[support[0]] = 1.0
            return h_opt, score(h_opt)

        def neg_ratio(x):
            x = np.maximum(x, 0)
            s_x = np.sum(x)
            if s_x < 1e-10:
                return 1e10
            x = x / s_x
            return -score(x)
        
        # We optimize only on the support variables to speed up SLSQP
        x0 = h[support]
        
        # Bounds for the support variables
        bounds = [(0, 1)] * len(support)
        
        # Constraint: sum(x) = 1
        # But since we normalize inside neg_ratio, we can just use bounds.
        # However, SLSQP works better with explicit constraints if we don't normalize internally.
        # Let's stick to normalizing internally and using bounds [0, 1].
        
        try:
            # Use SLSQP
            result = minimize(neg_ratio, x0, method='SLSQP', 
                             bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-12})
            
            if result.success:
                x_opt = np.maximum(result.x, 0)
                s_opt = np.sum(x_opt)
                if s_opt > 1e-10:
                    x_opt = x_opt / s_opt
                    h_full = np.zeros(N)
                    h_full[support] = x_opt
                    s_val = score(h_full)
                    if s_val > score(h):
                        return h_full, s_val
        except Exception:
            pass
        
        return h, score(h)

    def find_sidon_set(k, seed=0):
        """Find a k-element Sidon-like set via greedy + local search."""
        rng = np.random.RandomState(seed)
        best_set = None
        best_mc = np.inf
        
        for attempt in range(20):
            S = set()
            for _ in range(k):
                candidates = [j for j in range(N) if j not in S]
                if not candidates:
                    break
                best_cand = None
                best_mc_cand = np.inf
                for j in candidates:
                    S_test = S | {j}
                    h_tmp = np.zeros(N)
                    h_tmp[list(S_test)] = 1.0
                    conv = np.convolve(h_tmp, h_tmp, mode='full')
                    mc = np.max(conv)
                    if mc < best_mc_cand:
                        best_mc_cand = mc
                        best_cand = j
                if best_cand is not None:
                    S.add(best_cand)
            
            S = sorted(S)
            h_tmp = np.zeros(N)
            h_tmp[S] = 1.0
            conv = np.convolve(h_tmp, h_tmp, mode='full')
            mc = np.max(conv)
            
            # Local search: swap elements
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(len(S)):
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:i] + [j] + S[i+1:])
                        h_tmp = np.zeros(N)
                        h_tmp[S_new] = 1.0
                        conv = np.convolve(h_tmp, h_tmp, mode='full')
                        mc_new = np.max(conv)
                        if mc_new < mc:
                            mc = mc_new
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
            
            if mc < best_mc:
                best_mc = mc
                best_set = S
        
        return best_set if best_set else sorted(rng.choice(N, k, replace=False).tolist())

    best_h = np.zeros(N)
    best_score = 0.0
    
    # Strategy 1: Optimize on known good sparse supports
    seed_sets = [
        [0, 1, 4, 10, 18, 23, 30],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 40, 47],
        [0, 2, 5, 11, 19, 24, 31],
        [0, 2, 5, 11, 19, 24, 31, 40],
        [0, 3, 7, 12, 18, 25, 33],
        [0, 4, 9, 15, 22, 30, 39],
    ]
    
    for S in seed_sets:
        if time.time() > deadline - 5.0:
            break
        h0 = np.zeros(N)
        h0[S] = 1.0 / len(S)
        
        # Multiplicative update
        h_opt, s = multiplicative_optimize(h0, max_iter=1000, eta_start=0.2, eta_end=0.01)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
            
        # Local refinement
        if time.time() < deadline - 10.0:
            h_ref, s_ref = local_refine(h_opt)
            if s_ref > best_score:
                best_score = s_ref
                best_h = h_ref.copy()

    # Strategy 2: Find new Sidon sets and optimize
    for k in range(4, 11):
        if time.time() > deadline - 5.0:
            break
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
            S = find_sidon_set(k, seed=seed)
            h0 = np.zeros(N)
            h0[S] = 1.0 / len(S)
            
            h_opt, s = multiplicative_optimize(h0, max_iter=800, eta_start=0.2, eta_end=0.01)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
                
            if time.time() < deadline - 10.0:
                h_ref, s_ref = local_refine(h_opt)
                if s_ref > best_score:
                    best_score = s_ref
                    best_h = h_ref.copy()

    # Strategy 3: Random restarts with multiplicative updates
    rng = np.random.RandomState(123)
    for iter in range(200):
        if time.time() > deadline - 5.0:
            break
        
        k = rng.randint(3, 10)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        h0 = np.zeros(N)
        h0[S] = 1.0 / k
        # Add noise
        h0 += rng.randn(N) * 0.02
        h0 = np.maximum(h0, 0)
        s_norm = np.sum(h0)
        if s_norm > 1e-10:
            h0 /= s_norm
            
        h_opt, s = multiplicative_optimize(h0, max_iter=500, eta_start=0.15, eta_end=0.01)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()

    # Final polish
    if time.time() < deadline - 5.0:
        h_ref, s_ref = local_refine(best_h)
        if s_ref > best_score:
            best_score = s_ref
            best_h = h_ref.copy()

    return list(best_h)
