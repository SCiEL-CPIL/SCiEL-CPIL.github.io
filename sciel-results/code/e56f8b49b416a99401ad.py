"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Combined strategy:
1. Parametric Gaussian mixture model (2-3 components) to reduce search space
2. Epigraph formulation via SLSQP for structured optimization
3. Block-coordinate descent on symmetric pairs around dominant peak
4. Sparse initialization with diverse patterns
5. Local refinement with peak moving/scaling
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

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
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured
    np.random.seed(42)
    for trial in range(40):
        k = np.random.randint(2, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(25):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Structured sparse: pairs symmetric around center
    for k in range(2, 10):
        for offset in range(1, 8):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)
    
    # 7. Alternating patterns
    for start in range(N):
        for gap in [2, 3, 4, 5, 7, 10]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)
    
    # --- Phase 2: Parametric Gaussian Mixture Optimization ---
    # Represent h as sum of 2-3 shifted, scaled Gaussian basis functions
    # Parameters: [c1, w1, a1, c2, w2, a2, (c3, w3, a3)]
    # This reduces 50-dim search to 6-9 dim continuous search
    
    def gmm_to_h(params, n_components):
        """Convert GMM parameters to height vector"""
        h = np.zeros(N)
        for i in range(n_components):
            c = params[3*i]
            w = params[3*i+1]
            a = params[3*i+2]
            if w > 0.01:
                h += a * np.exp(-0.5 * ((x_idx - c) / w)**2)
        h = np.maximum(0, h)
        total = np.sum(h)
        if total > 1e-15:
            h /= total
        else:
            h = np.ones(N) / N
        return h
    
    def gmm_score(params, n_components):
        h = gmm_to_h(params, n_components)
        return _score_raw(h)
    
    # Optimize GMM parameters using Nelder-Mead
    gmm_inits = []
    
    # Try 2-component GMMs
    for c1 in np.linspace(5, N-5, 5):
        for w1 in [1, 2, 3, 5]:
            for c2 in np.linspace(5, N-5, 5):
                if abs(c2 - c1) < 3:
                    continue
                for w2 in [1, 2, 3, 5]:
                    params = np.array([c1, w1, 1.0, c2, w2, 1.0])
                    gmm_inits.append((params, 2))
    
    # Try 3-component GMMs
    for c1 in np.linspace(5, N-5, 3):
        for w1 in [1, 2, 4]:
            for c2 in np.linspace(5, N-5, 3):
                if abs(c2 - c1) < 3:
                    continue
                for w2 in [1, 2, 4]:
                    for c3 in np.linspace(5, N-5, 3):
                        if abs(c3 - c1) < 3 or abs(c3 - c2) < 3:
                            continue
                        for w3 in [1, 2, 4]:
                            params = np.array([c1, w1, 1.0, c2, w2, 1.0, c3, w3, 1.0])
                            gmm_inits.append((params, 3))
    
    # Optimize top GMM candidates
    gmm_candidates = []
    for params, n_comp in gmm_inits[:50]:
        s = gmm_score(params, n_comp)
        gmm_candidates.append((s, params, n_comp))
    
    gmm_candidates.sort(key=lambda x: -x[0])
    
    for s_init, params_init, n_comp in gmm_candidates[:15]:
        if time.time() > deadline - 50:
            break
        
        def neg_score(params):
            return -gmm_score(params, n_comp)
        
        bounds = []
        for i in range(n_comp):
            bounds.append((1, N-1))  # center
            bounds.append((0.5, N))  # width
            bounds.append((0.1, 10))  # amplitude
        
        try:
            res = opt.minimize(
                neg_score,
                params_init,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            h_candidate = gmm_to_h(res.x, n_comp)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # --- Phase 3: Epigraph Formulation Optimization ---
    num_conv = 2 * N - 1
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0
        
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
        
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        for j in range(N):
            k_start = j
            k_end = min(j + N, num_conv)
            if k_start < num_conv and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] = -2.0 * h[indices_h]
        return jac
        
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    def run_epigraph(h_init, maxiter=200, ftol=1e-10):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None
    
    # Run epigraph from multiple inits (limit to keep within time budget)
    max_epigraph_runs = min(len(inits), 25)
    for i, h_init in enumerate(inits[:max_epigraph_runs]):
        if time.time() > deadline - 40:
            break
        h_candidate = run_epigraph(h_init, maxiter=200, ftol=1e-10)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 4: Block-Coordinate Descent Refinement ---
    def block_coordinate_refine(h_start, max_iter=30):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 10:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            
            # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                    
                hi = h[i]
                hj = h[j]
                
                t_min = -hi
                t_max = hj
                
                if t_max - t_min < 1e-15:
                    continue
                    
                best_t = 0.0
                best_local_score = _score_raw(h)
                
                for t in np.linspace(t_min * 0.8, t_max * 0.8, 15):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-15):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_local_score:
                        best_local_score = s_test
                        best_t = t
                
                if best_t != 0.0:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    h /= np.sum(h)
                    improved = True
        
        return h
    
    # Apply block coordinate refinement to best solution
    if time.time() < deadline - 5:
        h_refined = block_coordinate_refine(best_h, max_iter=50)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined
    
    # --- Phase 5: Final Polish with Tighter Tolerances ---
    if time.time() < deadline - 2:
        h_polished = run_epigraph(best_h, maxiter=500, ftol=1e-12)
        if h_polished is not None:
            s_polished = _score_raw(h_polished)
            if s_polished > best_score:
                best_score = s_polished
                best_h = h_polished
    
    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return list(best_h)
