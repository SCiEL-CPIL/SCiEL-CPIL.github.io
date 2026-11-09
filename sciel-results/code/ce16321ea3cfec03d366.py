import numpy as np
from scipy.optimize import minimize
import time

def compute_volume(X, Y):
    N = 8
    w = 1.0 / N
    
    def get_criticals(P):
        crits = [0.0, 1.0]
        for i in range(N):
            for k in range(N):
                if i == k: continue
                # L_i = L_k
                if i != k:
                    c = (P[k] - P[i]) * N / (i - k)
                    if 0 <= c <= 1: crits.append(c)
                # L_i = R_k
                if i != k:
                    c = (P[k] - P[i] + w) * N / (i - k)
                    if 0 <= c <= 1: crits.append(c)
                # R_i = L_k
                if i != k:
                    c = (P[k] - P[i] - w) * N / (i - k)
                    if 0 <= c <= 1: crits.append(c)
                # R_i = R_k
                if i != k:
                    c = (P[k] - P[i]) * N / (i - k) # same as L_i=L_k
                    # Actually R_i = R_k is same as L_i = L_k
                    # Let's just use the 4 combinations of offsets 0 and w
                # More systematically:
                # Endpoints are P_i + i/N c + delta, where delta in {0, w}
                # Equality: P_i + i/N c + d1 = P_k + k/N c + d2
                # (i-k)/N c = P_k - P_i + d2 - d1
                # c = N (P_k - P_i + d2 - d1) / (i - k)
                for d1 in [0, w]:
                    for d2 in [0, w]:
                        if i == k: continue
                        c = N * (P[k] - P[i] + d2 - d1) / (i - k)
                        if -1e-9 <= c <= 1 + 1e-9:
                            crits.append(np.clip(c, 0, 1))
        return np.unique(np.sort(crits))
    
    def length_at_c(P, c):
        intervals = []
        for i in range(N):
            start = P[i] + (i / N) * c
            end = start + w
            intervals.append((start, end))
        intervals.sort()
        merged = []
        for start, end in intervals:
            if not merged:
                merged.append([start, end])
            else:
                if start <= merged[-1][1] + 1e-12:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
        return sum(end - start for start, end in merged)
    
    crits_x = get_criticals(X)
    crits_y = get_criticals(Y)
    all_crits = np.unique(np.sort(np.concatenate([crits_x, crits_y])))
    
    total_vol = 0.0
    for idx in range(len(all_crits) - 1):
        c_a = all_crits[idx]
        c_b = all_crits[idx + 1]
        if c_b - c_a < 1e-12: continue
        c_mid = (c_a + c_b) / 2
        
        Lx_a = length_at_c(X, c_a)
        Lx_b = length_at_c(X, c_b)
        Lx_m = length_at_c(X, c_mid)
        
        Ly_a = length_at_c(Y, c_a)
        Ly_b = length_at_c(Y, c_b)
        Ly_m = length_at_c(Y, c_mid)
        
        # Integrate Lx(c) Ly(c) from c_a to c_b
        # Lx, Ly are linear, so product is quadratic.
        # Simpson's rule is exact for quadratics.
        f_a = Lx_a * Ly_a
        f_m = Lx_m * Ly_m
        f_b = Lx_b * Ly_b
        total_vol += (c_b - c_a) / 6.0 * (f_a + 4 * f_m + f_b)
        
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    
    N = 8
    # Keich construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    X0 = keich.copy()
    Y0 = keich.copy()
    
    def objective(params):
        X = params[:N]
        Y = params[N:]
        return compute_volume(X, Y)
        
    x0 = np.concatenate([X0, Y0])
    
    # Optimization
    best_x = x0
    best_val = objective(x0)
    
    # Use Powell or Nelder-Mead
    # Powell is good for this kind of problem
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 500, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception as e:
        pass
        
    if time.time() > deadline - 60:
        pass
    else:
        # Try Nelder-Mead from best_x
        try:
            res2 = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 500, 'disp': False, 'xatol': 1e-6, 'fatol': 1e-8})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception as e:
            pass
            
    X_opt = best_x[:N]
    Y_opt = best_x[N:]
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
