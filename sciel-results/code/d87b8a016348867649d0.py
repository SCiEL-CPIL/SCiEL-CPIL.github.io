import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 138.0  # Leave 12s buffer for safety (evaluator kills at 150s)
    
    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def objective_lse_and_grad(h, beta, lambda_sum):
        """
        Objective: LSE(correlations) + penalty(sum - N/2)^2
        Returns (value, gradient)
        """
        g = 1.0 - h
        c_all = np.concatenate([np.correlate(h, g, mode='full')[:N-1], 
                                 np.correlate(h, g, mode='full')[N:]])
        
        # LogSumExp for numerical stability
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500
        exps = np.exp(beta * (c_all - max_c))
        sum_exps = np.sum(exps)
        lse = max_c + np.log(sum_exps) / beta
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        
        value = lse + sum_pen
        
        # Analytic gradient
        # dLSE/dh_i = (1/sum_exps) * sum_k exps[k] * dC_k/dh_i
        # C_k = sum_j h[j] * (1 - h[j+k]) for positive shift k
        # dC_k/dh_i = (1 - h[i+k]) if i+k < N, plus -h[i-k] if i-k >= 0
        # For negative shifts, similar logic
        grad = np.zeros(N)
        
        # For each shift k, compute dC_k/dh
        # Shifts are from -(N-1) to -1 and 1 to N-1
        # c_all[0:N-1] corresponds to shifts N-1, N-2, ..., 1 (positive)
        # c_all[N-1:2N-2] corresponds to shifts -1, -2, ..., -(N-1) (negative)
        
        # Positive shifts: c_all[j] for j in 0..N-2 corresponds to shift k = N-1-j
        # C_k = sum_{j=0}^{N-k-1} h[j] * (1-h[j+k])
        # dC_k/dh[i] = (1-h[i+k]) if i+k < N, plus -h[i-k] if i-k >= 0
        
        for j in range(N-1):
            k = N-1-j  # shift value
            weight = exps[j] / sum_exps
            # dC_k/dh[i] for i where i+k < N: (1-h[i+k])
            for i in range(N-k):
                grad[i] += weight * (1 - h[i+k])
            # dC_k/dh[i] for i where i-k >= 0: -h[i-k]
            for i in range(k, N):
                grad[i] += weight * (-h[i-k])
        
        # Negative shifts: c_all[N-1+j] for j in 0..N-2 corresponds to shift k = -(j+1)
        # C_{-k} = sum_{j=k}^{N-1} h[j] * (1-h[j-k])
        # dC_{-k}/dh[i] = (1-h[i-k]) if i-k >= 0, plus -h[i+k] if i+k < N
        for j in range(N-1):
            k = j+1  # shift magnitude
            weight = exps[N-1+j] / sum_exps
            # dC_{-k}/dh[i] for i where i-k >= 0: (1-h[i-k])
            for i in range(k, N):
                grad[i] += weight * (1 - h[i-k])
            # dC_{-k}/dh[i] for i where i+k < N: -h[i+k]
            for i in range(N-k):
                grad[i] += weight * (-h[i+k])
        
        # Scale by beta for LSE gradient
        grad *= beta
        
        # Add gradient of sum penalty: d/dh_i [lambda_sum * (sum_h - N/2)^2] = 2*lambda_sum*(sum_h-N/2)
        grad += 2 * lambda_sum * (sum_h - N/2)
        
        return value, grad

    def objective_epigraph(h, t, lambda_sum):
        """
        Epigraph formulation: minimize t + penalty(sum - N/2)^2
        subject to C_k <= t for all k.
        We use a smooth approximation: sum_k max(0, C_k - t)^2 + t + penalty
        """
        g = 1.0 - h
        c_all = np.concatenate([np.correlate(h, g, mode='full')[:N-1], 
                                 np.correlate(h, g, mode='full')[N:]])
        
        # Smooth max: use soft plus for each violation
        violations = np.maximum(0.0, c_all - t)
        violation_penalty = np.sum(violations**2)
        
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - N/2)**2
        
        return t + violation_penalty + sum_pen

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2.
        Uses water-filling approach.
        """
        h = np.array(h, dtype=float).copy()
        target_sum = N / 2.0
        
        # Water-filling: find threshold tau such that sum(clip(h - tau, 0, 1)) = target_sum
        # Binary search on tau
        lo = -1.0
        hi = 2.0
        for _ in range(60):
            mid = (lo + hi) / 2
            h_adjusted = np.clip(h - mid, 0.0, 1.0)
            s = np.sum(h_adjusted)
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        
        tau = (lo + hi) / 2
        h_proj = np.clip(h - tau, 0.0, 1.0)
        
        # Fine-tune: distribute remaining error among non-bound elements
        diff = target_sum - np.sum(h_proj)
        if abs(diff) > 1e-14:
            mask = (h_proj > 1e-10) & (h_proj < 1 - 1e-10)
            if np.any(mask):
                h_proj[mask] += diff / np.sum(mask)
                h_proj = np.clip(h_proj, 0.0, 1.0)
        
        return h_proj

    def optimize_lse(h0, beta, lambda_sum, max_iter, ftol=1e-9, gtol=1e-9):
        """Optimize using LSE surrogate with analytic gradient."""
        bounds = [(0, 1) for _ in range(N)]
        
        def obj_and_grad(h):
            return objective_lse_and_grad(h, beta, lambda_sum)

        try:
            res = scipy.optimize.minimize(
                obj_and_grad, 
                h0, 
                jac=True,
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        # Project to feasible set
        h_opt = project_to_feasible(h_opt)
        
        return h_opt

    def optimize_epigraph(h0, t0, lambda_sum, max_iter, ftol=1e-9, gtol=1e-9):
        """Optimize using epigraph formulation with auxiliary variable t."""
        # Variables: [h_0, ..., h_{N-1}, t]
        # Bounds: h_i in [0,1], t in [0, N/2]
        bounds = [(0, 1) for _ in range(N)] + [(0, N/2)]
        
        def obj(x):
            h = x[:N]
            t = x[N]
            return objective_epigraph(h, t, lambda_sum)

        x0 = np.concatenate([h0, [t0]])
        
        try:
            res = scipy.optimize.minimize(
                obj, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            x_opt = res.x
            h_opt = x_opt[:N]
        except Exception:
            h_opt = h0.copy()
        
        # Project to feasible set
        h_opt = project_to_feasible(h_opt)
        
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)

    rng = np.random.default_rng(42)
    
    # List of restart configurations: (h0, method, params, max_iter)
    # method: 'lse' or 'epigraph'
    restarts = []
    
    # 1. Uniform start with LSE
    restarts.append((np.full(N, 0.5), 'lse', (5.0, 100.0), 200))
    
    # 2-5. Random perturbations with LSE
    for i in range(4):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 5.0 + i * 5.0
        lambda_sum = 100.0 + i * 50.0
        
        restarts.append((h_rand, 'lse', (beta, lambda_sum), 200))
        
    # 6-9. Bimodal initializations with LSE
    for i in range(4):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.9, 0.1)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 15.0 + i * 5.0
        lambda_sum = 200.0 + i * 50.0
        
        restarts.append((h_rand, 'lse', (beta, lambda_sum), 200))
    
    # 10-13. Epigraph formulations with different t initializations
    for i in range(4):
        h_rand = rng.uniform(0.2, 0.8, size=N)
        h_rand = h_rand / np.sum(h_rand) * (N/2)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        t0 = score_h(h_rand) + 0.5  # Start with t slightly above current max
        lambda_sum = 200.0 + i * 50.0
        
        restarts.append((h_rand, 'epigraph', (t0, lambda_sum), 200))

    for idx, (h0, method, params, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        if method == 'lse':
            beta, lambda_sum = params
            h_opt = optimize_lse(h0, beta, lambda_sum, max_iter)
        else:  # epigraph
            t0, lambda_sum = params
            h_opt = optimize_epigraph(h0, t0, lambda_sum, max_iter)
        
        current_score = score_h(h_opt)
        
        if not np.isclose(np.sum(h_opt), N/2, atol=1e-8):
            continue
        if not (np.all(h_opt >= -1e-12) and np.all(h_opt <= 1 + 1e-12)):
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
    
    # Final polish pass with tighter tolerances
    if time.time() - start_time < time_limit - 10:
        # Use the best solution found so far as starting point
        h_polish = best_h.copy()
        
        # Try several high-beta polish iterations with LSE
        for beta_polish in [30.0, 50.0, 80.0]:
            if time.time() - start_time > time_limit - 5:
                break
            
            h_opt = optimize_lse(h_polish, beta_polish, 500.0, 500, ftol=1e-11, gtol=1e-11)
            
            current_score = score_h(h_opt)
            
            if np.isclose(np.sum(h_opt), N/2, atol=1e-10) and \
               np.all(h_opt >= -1e-12) and np.all(h_opt <= 1 + 1e-12):
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
                    h_polish = h_opt.copy()
        
        # Try epigraph polish
        if time.time() - start_time < time_limit - 5:
            t0 = best_score + 0.1
            h_opt = optimize_epigraph(h_polish, t0, 1000.0, 500, ftol=1e-11, gtol=1e-11)
            
            current_score = score_h(h_opt)
            
            if np.isclose(np.sum(h_opt), N/2, atol=1e-10) and \
               np.all(h_opt >= -1e-12) and np.all(h_opt <= 1 + 1e-12):
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
    
    return best_h
