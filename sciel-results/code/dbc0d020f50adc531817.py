import numpy as np
import scipy.optimize as so
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    deadline = time.time() + 590.0  # Safety margin
    
    def compute_1d_volume_sq(X):
        """
        Computes the integral of L(c)^2 over c in [0, 1],
        where L(c) is the length of the union of intervals
        [X[i] + c*i/N, X[i] + c*i/N + 1/N] for i in 0..N-1.
        """
        # Endpoints: (A, B, is_right) where value is A + B*c
        eps = []
        for i in range(N):
            eps.append((X[i], i/N, 0))
            eps.append((X[i] + delta, i/N, 1))
        
        # Find critical points where the order of endpoints changes
        crit = {0.0, 1.0}
        for i in range(len(eps)):
            A1, B1, _ = eps[i]
            for j in range(i+1, len(eps)):
                A2, B2, _ = eps[j]
                if abs(B1 - B2) > 1e-15:
                    c = (A2 - A1) / (B1 - B2)
                    if 0.0 < c < 1.0:
                        crit.add(c)
        
        all_crit = sorted(crit)
        
        def union_length(c):
            pts = []
            for A, B, is_r in eps:
                pts.append((A + B*c, is_r))
            pts.sort()
            total = 0.0
            count = 0
            start_pos = 0.0
            for val, is_r in pts:
                if is_r == 0:
                    if count == 0:
                        start_pos = val
                    count += 1
                else:
                    count -= 1
                    if count == 0:
                        total += val - start_pos
            return total
        
        total_vol_sq = 0.0
        for k in range(len(all_crit) - 1):
            c0 = all_crit[k]
            c1 = all_crit[k+1]
            h = c1 - c0
            if h < 1e-12:
                continue
            
            L0 = union_length(c0)
            L1 = union_length(c1)
            
            # L(c) is linear on [c0, c1]: L(c) = L0 + (L1-L0)/h * (c-c0)
            # We want integral of L(c)^2 dc
            # Let u = (c-c0)/h, dc = h du
            # Integral from 0 to 1 of (L0 + u*(L1-L0))^2 * h du
            # = h * [ L0^2 + L0*(L1-L0) + (1/3)*(L1-L0)^2 ]
            # = h * [ L0^2 + L0*L1 - L0^2 + (1/3)(L1^2 - 2*L0*L1 + L0^2) ]
            # = h * [ L0*L1 + (1/3)L1^2 - (2/3)L0*L1 + (1/3)L0^2 ]
            # = h * [ (1/3)L0^2 + (1/3)L0*L1 + (1/3)L1^2 ]
            # = (h/3) * (L0^2 + L0*L1 + L1^2)
            
            term = (h / 3.0) * (L0*L0 + L0*L1 + L1*L1)
            total_vol_sq += term
            
        return total_vol_sq

    def objective(params):
        return compute_1d_volume_sq(params)

    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Normalize keich to be centered or just use as is? 
    # The absolute position doesn't matter for volume, only relative.
    # But optimization works better with bounded variables.
    # Let's center it.
    keich = keich - np.mean(keich)
    
    p0 = keich.copy()
    best_p = p0.copy()
    best_v = objective(p0)
    
    rng = np.random.default_rng(42)
    
    # Multi-start Nelder-Mead
    # We assume X = Y, so we only optimize 8 variables.
    # The score is proportional to 1/V. If we minimize V_1d_sq, 
    # the actual 3D volume V is V_1d_sq (since V = int Lx Ly = int L L = int L^2).
    
    # Generate diverse starts
    starts = [p0.copy()]
    
    # Random starts in a reasonable range
    # The optimal spread is likely around [-0.5, 0.5] or similar
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        p_try = rng.uniform(-0.5, 0.5, size=N)
        # Add some structure: sorted starts
        p_sorted = np.sort(p_try)
        starts.append(p_sorted)
        # Unsorted starts
        starts.append(p_try)
        
    # Halton-like starts
    def halton_sequence(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
        
    for i in range(1, 20):
        if time.time() > deadline - 10:
            break
        p_h = np.array([halton_sequence(i, base) for base in range(2, 2+N)])
        p_h = p_h - 0.5 # Center in [-0.5, 0.5]
        starts.append(p_h)

    for p_start in starts:
        if time.time() > deadline - 5:
            break
        
        try:
            # Nelder-Mead is good for low-dim non-smooth
            res = so.minimize(objective, p_start, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
                
            # Optional: Polish with Powell if time permits
            if time.time() < deadline - 2:
                res2 = so.minimize(objective, res.x, method='Powell',
                                   options={'maxiter': 50, 'xtol': 1e-10, 'ftol': 1e-12})
                if res2.fun < best_v:
                    best_v = res2.fun
                    best_p = res2.x.copy()
                    
        except Exception:
            pass

    # Final high-precision polish on the best found
    if time.time() < deadline - 1:
        try:
            res = so.minimize(objective, best_p, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    X_opt = best_p
    Y_opt = best_p # Symmetry assumption
    
    # Construct final positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
