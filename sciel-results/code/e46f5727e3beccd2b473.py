import numpy as np
import time
import scipy.optimize as so

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    
    # Deadline guard
    deadline = time.time() + 620.0
    
    def _union_len_1d(pos, c):
        """Compute 1D union length of intervals [pos[i] + c*i/N, pos[i] + c*i/N + delta] for i=0..N-1"""
        lefts = pos + c * np.arange(N) / N
        rights = lefts + delta
        vals = np.concatenate([lefts, rights])
        types = np.concatenate([np.zeros(N, dtype=np.int8), np.ones(N, dtype=np.int8)])
        order = np.lexsort((types, vals))
        vals_s = vals[order]
        types_s = types[order]
        
        total = 0.0
        count = 0
        start = 0.0
        for v, t in zip(vals_s, types_s):
            if t == 0:
                if count == 0:
                    start = v
                count += 1
            else:
                count -= 1
                if count == 0:
                    total += v - start
        return total
    
    def _find_critical(pos):
        """Find all c in (0,1) where ordering of endpoints changes"""
        crit = set()
        for i in range(N):
            for j in range(i + 1, N):
                denom = i - j
                if abs(denom) < 1e-15:
                    continue
                c1 = N * (pos[j] - pos[i]) / (i - j)
                c2 = N * (pos[j] + delta - pos[i]) / (i - j)
                c3 = N * (pos[j] - pos[i] - delta) / (i - j)
                for c_val in [c1, c2, c3]:
                    if 0.0 < c_val < 1.0:
                        crit.add(round(c_val, 14))
        return sorted(crit)
    
    def _exact_volume(X, Y):
        """Compute exact union volume using piecewise-linear integration"""
        crit_x = _find_critical(X)
        crit_y = _find_critical(Y)
        all_crit = sorted(set([0.0] + crit_x + crit_y + [1.0]))
        
        vol = 0.0
        for k in range(len(all_crit) - 1):
            c0 = all_crit[k]
            c1 = all_crit[k + 1]
            h = c1 - c0
            if h < 1e-15:
                continue
            
            Lx0 = _union_len_1d(X, c0)
            Lx1 = _union_len_1d(X, c1)
            Ly0 = _union_len_1d(Y, c0)
            Ly1 = _union_len_1d(Y, c1)
            
            dx = Lx1 - Lx0
            dy = Ly1 - Ly0
            term = Lx0 * Ly0 + 0.5 * (Lx0 * dy + Ly0 * dx) + (1.0 / 3.0) * dx * dy
            vol += h * term
        
        return vol
    
    def _objective(params):
        X = params[:N]
        Y = params[N:]
        return _exact_volume(X, Y)
    
    def _build_positions(X, Y):
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = X[i_flat]
        base_y = Y[j_flat]
        return np.stack([base_x, base_y], axis=1)
    
    # Generate diverse initial candidates
    rng_init = np.random.default_rng(42)
    candidates = []
    
    # Keich-like construction
    pos_bits = int(np.log2(N))
    j_terms = np.arange(1, pos_bits + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps_bits = np.array(list(bin(idx)[2:]), dtype=int)
        eps_bits = np.pad(eps_bits, (pos_bits - len(eps_bits), 0), 'constant')
        keich[idx] = np.sum((1 - j_terms) * eps_bits * (2.0 ** -j_terms) / pos_bits)
    candidates.append(np.concatenate([keich, keich]))
    
    # Uniform spread
    uniform = np.linspace(0, 0.5, N)
    candidates.append(np.concatenate([uniform, uniform]))
    
    # Converging patterns
    center = 0.5
    for spread in [0.1, 0.2, 0.3, 0.4]:
        conv = center + spread * (np.arange(N) - (N-1)/2.0) / (N-1)
        candidates.append(np.concatenate([conv, conv]))
    
    # Asymmetric patterns
    for spread in [0.15, 0.25, 0.35]:
        conv_x = 0.5 + spread * (np.arange(N) - (N-1)/2.0) / (N-1)
        conv_y = 0.5 - spread * (np.arange(N) - (N-1)/2.0) / (N-1)
        candidates.append(np.concatenate([conv_x, conv_y]))
    
    # Random starts
    for _ in range(50):
        rx = rng_init.uniform(-0.2, 0.7, N)
        ry = rng_init.uniform(-0.2, 0.7, N)
        candidates.append(np.concatenate([rx, ry]))
    
    # Evaluate all candidates
    best_v = float('inf')
    best_p = None
    for p0 in candidates:
        try:
            v = _objective(p0)
            if v < best_v:
                best_v = v
                best_p = p0.copy()
        except Exception:
            pass
    
    # Phase 1: Nelder-Mead with multiple restarts
    nm_restarts = 12
    for restart in range(nm_restarts):
        if time.time() > deadline - 200:
            break
        
        if restart == 0:
            p_current = best_p.copy()
        else:
            scale = 0.08 / (1 + restart * 0.4)
            perturbation = rng_init.normal(0, scale, 2 * N)
            p_current = best_p + perturbation
        
        try:
            res = so.minimize(_objective, p_current, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9,
                                       'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
        
        # Quick Powell polish after each NM
        if time.time() < deadline - 180:
            try:
                res2 = so.minimize(_objective, best_p, method='Powell',
                                   options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10})
                if res2.fun < best_v:
                    best_v = res2.fun
                    best_p = res2.x.copy()
            except Exception:
                pass
    
    # Phase 2: Powell with tight tolerances
    if time.time() < deadline - 150:
        try:
            res_powell = so.minimize(_objective, best_p, method='Powell',
                                     options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-12})
            if res_powell.fun < best_v:
                best_v = res_powell.fun
                best_p = res_powell.x.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B for gradient-based refinement
    if time.time() < deadline - 120:
        try:
            res_lbfgs = so.minimize(_objective, best_p, method='L-BFGS-B',
                                    options={'maxiter': 8000, 'ftol': 1e-14, 'gtol': 1e-12,
                                             'maxls': 150, 'maxfun': 200000})
            if res_lbfgs.fun < best_v:
                best_v = res_lbfgs.fun
                best_p = res_lbfgs.x.copy()
        except Exception:
            pass
    
    # Phase 4: Another Powell pass with very tight tolerances
    if time.time() < deadline - 80:
        try:
            res_powell2 = so.minimize(_objective, best_p, method='Powell',
                                      options={'maxiter': 5000, 'xtol': 1e-13, 'ftol': 1e-13})
            if res_powell2.fun < best_v:
                best_v = res_powell2.fun
                best_p = res_powell2.x.copy()
        except Exception:
            pass
    
    # Phase 5: Coordinate descent with adaptive step sizes
    if time.time() < deadline - 40:
        p_cd = best_p.copy()
        v_cd = _objective(p_cd)
        for _cd_pass in range(200):
            if time.time() > deadline - 40:
                break
            improved = False
            for dim in range(2 * N):
                # Adaptive step: try logarithmically spaced steps
                for step in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                    for direction in [-1, 1]:
                        p_try = p_cd.copy()
                        p_try[dim] += direction * step
                        try:
                            v_try = _objective(p_try)
                            if v_try < v_cd - 1e-15:
                                v_cd = v_try
                                p_cd = p_try
                                improved = True
                        except Exception:
                            pass
            if not improved:
                break
        if v_cd < best_v:
            best_v = v_cd
            best_p = p_cd
    
    # Phase 6: Final L-BFGS-B polish
    if time.time() < deadline - 20:
        try:
            res_final = so.minimize(_objective, best_p, method='L-BFGS-B',
                                    options={'maxiter': 15000, 'ftol': 1e-15, 'gtol': 1e-13,
                                             'maxls': 200, 'maxfun': 300000})
            if res_final.fun < best_v:
                best_v = res_final.fun
                best_p = res_final.x.copy()
        except Exception:
            pass
    
    # Phase 7: Final micro-polish with coordinate descent
    if time.time() < deadline - 5:
        p_cd = best_p.copy()
        v_cd = _objective(p_cd)
        for _cd_pass in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for dim in range(2 * N):
                for step in [1e-9, 1e-10, 1e-11, 1e-12]:
                    for direction in [-1, 1]:
                        p_try = p_cd.copy()
                        p_try[dim] += direction * step
                        try:
                            v_try = _objective(p_try)
                            if v_try < v_cd - 1e-16:
                                v_cd = v_try
                                p_cd = p_try
                                improved = True
                        except Exception:
                            pass
            if not improved:
                break
        if v_cd < best_v:
            best_v = v_cd
            best_p = p_cd
    
    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    positions = _build_positions(X_opt, Y_opt)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        positions = _build_positions(keich, keich)
    
    # Ensure box area > 1e-9
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (np.arange(64) // 8 + 1) / 8)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (np.arange(64) % 8 + 1) / 8)
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        positions = _build_positions(keich, keich)
    
    return positions
