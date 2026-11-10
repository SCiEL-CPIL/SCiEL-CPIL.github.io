# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s kill
    
    N = 8
    
    def _get_breakpoints(X, N):
        """
        Calculate all c in (0, 1) where the order of endpoints changes.
        Endpoints for interval i are: L_i(c) = X[i] + (i/N)*c, R_i(c) = X[i] + (i/N)*c + 1/N
        Breakpoints occur when L_i(c) = L_j(c), L_i(c) = R_j(c), R_i(c) = L_j(c), or R_i(c) = R_j(c).
        Since R_i - L_i = 1/N is constant, R_i = R_j iff L_i = L_j.
        So we only need L_i = L_j and L_i = R_j (which implies L_i = L_j + 1/N).
        """
        breakpoints = set([0.0, 1.0])
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = (i - j) / N
                if denom == 0:
                    continue
                # L_i(c) = L_j(c) => X[i] + (i/N)c = X[j] + (j/N)c => (i-j)/N c = X[j] - X[i]
                c1 = (X[j] - X[i]) / denom
                if 0.0 < c1 < 1.0:
                    breakpoints.add(c1)
                
                # L_i(c) = R_j(c) => X[i] + (i/N)c = X[j] + (j/N)c + 1/N => (i-j)/N c = X[j] + 1/N - X[i]
                c2 = (X[j] + 1.0/N - X[i]) / denom
                if 0.0 < c2 < 1.0:
                    breakpoints.add(c2)
        
        return sorted(breakpoints)

    def _union_length(X, c, N):
        """
        Compute the total length of the union of intervals at height c.
        X: array of shape (N,)
        """
        Ls = X + c * np.arange(N) / N
        Rs = Ls + 1.0 / N
        order = np.argsort(Ls)
        Ls_sorted = Ls[order]
        Rs_sorted = Rs[order]
        
        total_len = 0.0
        current_end = -np.inf
        for l, r in zip(Ls_sorted, Rs_sorted):
            if l > current_end:
                total_len += (r - l)
                current_end = r
            else:
                if r > current_end:
                    total_len += (r - current_end)
                    current_end = r
        return total_len

    def _compute_cost(X, N=8):
        """
        Compute integral of (Union Length)^2 over c in [0, 1].
        This is the volume of the union if x and y are the same (symmetric case).
        """
        if not np.all(np.isfinite(X)):
            return 1e10
        
        breakpoints = _get_breakpoints(X, N)
        
        total_cost = 0.0
        
        for k in range(len(breakpoints) - 1):
            c_start = breakpoints[k]
            c_end = breakpoints[k + 1]
            if c_end - c_start < 1e-12:
                continue
            
            L_start = _union_length(X, c_start, N)
            L_end = _union_length(X, c_end, N)
            
            A = (L_end - L_start) / (c_end - c_start)
            B = L_start - A * c_start
            
            def F(c):
                return (A**2 * c**3) / 3.0 + (A * B * c**2) + (B**2 * c)
            
            total_cost += F(c_end) - F(c_start)
        
        return total_cost

    # Keich initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_x = keich.copy()
    best_cost = _compute_cost(keich, N)
    
    # Optimization with Nelder-Mead
    try:
        res = minimize(_compute_cost, best_x, args=(N,), method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < best_cost:
            best_cost = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        pass
    else:
        # Try random restarts to escape local minima
        rng = np.random.default_rng(42)
        for restart in range(40):
            if time.time() > deadline - 10:
                break
            x0 = rng.uniform(-0.5, 1.5, size=N)
            try:
                res = minimize(_compute_cost, x0, args=(N,), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_x = res.x
            except Exception:
                pass
    
    # High-precision polish pass with very tight tolerances
    if time.time() < deadline - 5:
        try:
            res = minimize(_compute_cost, best_x, args=(N,), method='Nelder-Mead',
                           options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Fallback check
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
