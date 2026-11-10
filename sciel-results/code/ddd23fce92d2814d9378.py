import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    num_c_samples = 2000
    c_vals = np.linspace(0, 1, num_c_samples)
    step_c = 1.0 / (num_c_samples - 1)
    
    def get_union_lengths(x, c_vals):
        lengths = np.zeros(len(c_vals))
        for idx, c in enumerate(c_vals):
            starts = x + c * np.arange(N) / N
            ends = starts + 1.0 / N
            idx_sort = np.argsort(starts)
            s_sorted = starts[idx_sort]
            e_sorted = ends[idx_sort]
            
            total = 0.0
            cs = s_sorted[0]
            ce = e_sorted[0]
            for k in range(1, N):
                if s_sorted[k] > ce:
                    total += ce - cs
                    cs = s_sorted[k]
                    ce = e_sorted[k]
                else:
                    if e_sorted[k] > ce:
                        ce = e_sorted[k]
            total += ce - cs
            lengths[idx] = total
        return lengths

    def integrate(lengths):
        return step_c * (0.5 * lengths[0] + np.sum(lengths[1:-1]) + 0.5 * lengths[-1])

    # Initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x_curr = keich - keich[0]
    y_curr = keich - keich[0]
    
    len_x_curr = get_union_lengths(x_curr, c_vals)
    len_y_curr = get_union_lengths(y_curr, c_vals)
    
    for _ in range(30):
        # Optimize x, y is fixed
        def obj_x(z):
            x = np.concatenate(([0.0], z))
            len_x = get_union_lengths(x, c_vals)
            return integrate(len_x * len_y_curr)
            
        res_x = minimize(obj_x, x_curr[1:], method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
        x_curr = np.concatenate(([0.0], res_x.x))
        len_x_curr = get_union_lengths(x_curr, c_vals)
        
        # Optimize y, x is fixed
        def obj_y(z):
            y = np.concatenate(([0.0], z))
            len_y = get_union_lengths(y, c_vals)
            return integrate(len_x_curr * len_y)
            
        res_y = minimize(obj_y, y_curr[1:], method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
        y_curr = np.concatenate(([0.0], res_y.x))
        len_y_curr = get_union_lengths(y_curr, c_vals)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_curr[i_flat]
    base_y = y_curr[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
