"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Objective: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Since h >= 0, ||h||_1 = sum(h)
    # By scale invariance, fix ||h||_2 = 1, then maximize 1 / (||h||_1 * ||h*h||_inf)
    # Equivalent to minimizing ||h||_1 * ||h*h||_inf subject to ||h||_2 = 1, h >= 0
    
    def autoconv(h):
        # Linear convolution (autoconvolution)
        return np.convolve(h, h, mode='full')
    
    def objective(h):
        """Return the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf). Higher is better."""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 0.0
        ac = autoconv(h)
        ac_max = np.max(np.abs(ac))
        if ac_max < 1e-30:
            return 0.0
        return norm2_sq / (norm1 * ac_max)
    
    def minimize_obj(h):
        """Minimize ||h||_1 * ||h*h||_inf with ||h||_2=1, h>=0. Returns (value, h_normalized)."""
        h = np.maximum(h, 0.0)
        norm2 = np.sqrt(np.sum(h * h))
        if norm2 < 1e-30:
            return 1e30, h
        h = h / norm2
        norm1 = np.sum(h)
        ac = autoconv(h)
        ac_max = np.max(np.abs(ac))
        return norm1 * ac_max, h
    
    # Try multiple initializations and use local optimization
    best_h = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Gaussian centered
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-5 * x**2))
    
    # 3. Sinc-like
    inits.append(np.maximum(np.sinc(x / 2.0), 0.0))
    
    # 4. Cosine window
    inits.append(np.cos(np.pi * x / 2.0) ** 2)
    
    # 5. Power-law decay
    inits.append(np.maximum(1.0 - np.abs(x), 0.0) ** 2)
    
    # 6. Sparse center
    h_sparse = np.zeros(N)
    h_sparse[N//2 - 2:N//2 + 3] = 1.0
    inits.append(h_sparse)
    
    # 7. Sparse edges
    h_edges = np.zeros(N)
    h_edges[:5] = 1.0
    h_edges[-5:] = 1.0
    inits.append(h_edges)
    
    # 8. Bimodal
    h_bi = np.exp(-20 * (x - 0.5)**2) + np.exp(-20 * (x + 0.5)**2)
    inits.append(h_bi)
    
    # 9. Random with seed
    rng = np.random.RandomState(42)
    for seed in range(20):
        h_rand = rng.random(N)
        inits.append(h_rand)
    
    # 10. Ramp
    inits.append(np.linspace(0.1, 1.0, N))
    
    # 11. Inverse ramp
    inits.append(np.linspace(1.0, 0.1, N))
    
    # 12. Triangular
    inits.append(np.maximum(1.0 - np.abs(x), 0.0))
    
    # For each initialization, try a simple gradient-free local search
    # using coordinate descent / Nelder-Mead style
    
    for h0 in inits:
        if time.time() > deadline:
            break
        
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0 * h0) < 1e-30:
            continue
        
        # Normalize to ||h||_2 = 1
        h = h0.copy()
        norm2 = np.sqrt(np.sum(h * h))
        h = h / norm2
        
        # Nelder-Mead style optimization to minimize ||h||_1 * ||h*h||_inf
        # with projection onto h >= 0 and ||h||_2 = 1
        
        def project(h):
            h = np.maximum(h, 0.0)
            norm2 = np.sqrt(np.sum(h * h))
            if norm2 < 1e-30:
                return np.ones(N) / np.sqrt(N)
            return h / norm2
        
        def obj_val(h):
            h = project(h)
            norm1 = np.sum(h)
            ac = autoconv(h)
            ac_max = np.max(np.abs(ac))
            return norm1 * ac_max
        
        # Simple coordinate descent
        h_curr = h.copy()
        curr_val = obj_val(h_curr)
        
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step in step_sizes:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(N):
                    # Try increasing
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = project(h_trial)
                    trial_val = obj_val(h_trial)
                    if trial_val < curr_val - 1e-12:
                        h_curr = h_trial
                        curr_val = trial_val
                        improved = True
                        continue
                    # Try decreasing
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = project(h_trial)
                    trial_val = obj_val(h_trial)
                    if trial_val < curr_val - 1e-12:
                        h_curr = h_trial
                        curr_val = trial_val
                        improved = True
        
        # Compute score
        score = objective(h_curr)
        if score > best_score:
            best_score = score
            best_h = h_curr.copy()
    
    # Also try a more sophisticated approach: use scipy if available
    try:
        from scipy.optimize import minimize
        
        def minimize_scipy(h0):
            h0 = np.maximum(h0, 0.0)
            norm2 = np.sqrt(np.sum(h0 * h0))
            if norm2 < 1e-30:
                return None
            h0 = h0 / norm2
            
            # Use log-barrier for nonnegativity
            # Minimize log(sum(h)) + log(max(autoconv(h))) subject to sum(h^2)=1, h>=0
            # Simpler: just minimize the product directly with bounds
            
            def obj_and_grad(h):
                h = np.maximum(h, 0.0)
                norm2 = np.sqrt(np.sum(h * h))
                if norm2 < 1e-30:
                    return 1e30, np.zeros(N)
                h = h / norm2
                norm1 = np.sum(h)
                ac = autoconv(h)
                ac_max = np.max(np.abs(ac))
                val = norm1 * ac_max
                # Gradient is tricky with the max; use finite differences
                return val, None
            
            # Use Nelder-Mead with bounds via projection
            res = minimize(
                lambda x: obj_and_grad(x)[0],
                h0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            h_opt = project(res.x)
            return h_opt, objective(h_opt)
        
        # Try scipy on a few inits
        for h0 in inits[:10]:
            if time.time() > deadline:
                break
            result = minimize_scipy(h0)
            if result is not None:
                h_opt, score = result
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
    
    except ImportError:
        pass
    
    # Final polish: very fine coordinate descent on the best solution
    if best_h is not None and time.time() < deadline - 10:
        h_curr = best_h.copy()
        curr_val = objective(h_curr)
        
        for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(N):
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    norm2 = np.sqrt(np.sum(h_trial * h_trial))
                    if norm2 > 1e-30:
                        h_trial = h_trial / norm2
                    trial_score = objective(h_trial)
                    if trial_score > curr_val + 1e-12:
                        h_curr = h_trial
                        curr_val = trial_score
                        improved = True
                        continue
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    norm2 = np.sqrt(np.sum(h_trial * h_trial))
                    if norm2 > 1e-30:
                        h_trial = h_trial / norm2
                    trial_score = objective(h_trial)
                    if trial_score > curr_val + 1e-12:
                        h_curr = h_trial
                        curr_val = trial_score
                        improved = True
        
        if curr_val > best_score:
            best_score = curr_val
            best_h = h_curr.copy()
    
    # Return the best heights found
    if best_h is None:
        return [1.0] * N
    
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
