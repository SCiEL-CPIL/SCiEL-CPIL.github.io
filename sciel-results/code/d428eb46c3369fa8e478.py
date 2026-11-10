import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Leave 40s margin for safety

    N = 8
    w = 1.0 / N

    def compute_cost(z):
        """
        Compute the exact integral of L(z, c)^2 from 0 to 1,
        where L(z, c) is the length of the union of intervals
        [z_i + c*i/N, z_i + c*i/N + 1/N] for i=0..N-1.
        """
        if not np.all(np.isfinite(z)):
            return 1e10

        # Find all critical points c in [0, 1] where the order of endpoints changes.
        # Endpoints are L_i(c) = z_i + c*i/N and R_i(c) = z_i + c*i/N + 1/N.
        # Critical points occur when L_i(c) = L_j(c), L_i(c) = R_j(c), R_i(c) = L_j(c), R_i(c) = R_j(c).
        # L_i = L_j => c(i-j)/N = z_j - z_i => c = N(z_j - z_i)/(i-j)
        # L_i = R_j => c(i-j)/N = z_j - z_i + 1/N => c = N(z_j - z_i + 1/N)/(i-j)
        # R_i = L_j => c(i-j)/N = z_j - z_i - 1/N => c = N(z_j - z_i - 1/N)/(i-j)
        # R_i = R_j => c(i-j)/N = z_j - z_i => c = N(z_j - z_i)/(i-j)
        
        criticals = [0.0, 1.0]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = i - j
                # L_i = L_j and R_i = R_j
                c1 = N * (z[j] - z[i]) / denom
                if 0.0 < c1 < 1.0:
                    criticals.append(c1)
                # L_i = R_j
                c2 = N * (z[j] - z[i] + w) / denom
                if 0.0 < c2 < 1.0:
                    criticals.append(c2)
                # R_i = L_j
                c3 = N * (z[j] - z[i] - w) / denom
                if 0.0 < c3 < 1.0:
                    criticals.append(c3)
        
        criticals = sorted(set(criticals))
        
        total_cost = 0.0
        
        def get_union_length(c):
            starts = z + c * np.arange(N) / N
            ends = starts + w
            # Sort intervals by start
            order = np.argsort(starts)
            s_sorted = starts[order]
            e_sorted = ends[order]
            
            length = 0.0
            current_end = -np.inf
            for s, e in zip(s_sorted, e_sorted):
                if s > current_end:
                    length += (e - s)
                    current_end = e
                else:
                    if e > current_end:
                        length += (e - current_end)
                        current_end = e
            return length

        for k in range(len(criticals) - 1):
            c_start = criticals[k]
            c_end = criticals[k + 1]
            if c_end - c_start < 1e-12:
                continue
            
            L_start = get_union_length(c_start)
            L_end = get_union_length(c_end)
            
            # L(c) is linear in c within this interval.
            # L(c) = A*c + B
            A = (L_end - L_start) / (c_end - c_start)
            B = L_start - A * c_start
            
            # Integrate (A*c + B)^2 from c_start to c_end
            # = [A^2 c^3/3 + A B c^2 + B^2 c]
            F_start = (A**2 * c_start**3) / 3.0 + (A * B * c_start**2) + (B**2 * c_start)
            F_end = (A**2 * c_end**3) / 3.0 + (A * B * c_end**2) + (B**2 * c_end)
            
            total_cost += F_end - F_start
            
        return total_cost

    # Keich initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    best_z = keich.copy()
    best_cost = compute_cost(keich)

    # Optimization using Nelder-Mead
    # We will try a few restarts to escape local minima
    
    inits = [keich]
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        # Random perturbations around the best solution or random starts
        if best_cost < 1e-2:
            z0 = best_z + rng.normal(0, 0.01, N)
        else:
            z0 = rng.uniform(-0.5, 0.5, N)
        inits.append(z0)
        
    for z0 in inits:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(compute_cost, z0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_z = res.x
        except Exception:
            pass

    # Construct the 64x2 array
    # Tube k corresponds to i = k // N, j = k % N
    # x_k = z_i, y_k = z_j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_z[i_flat]
    base_y = best_z[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        # Fallback
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions
