import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600  # 600 seconds budget

    # 1D Union Length Calculation
    def union_length(positions):
        """
        Calculate the total length of the union of intervals [p_i, p_i + 1/N].
        positions: array of shape (N,)
        """
        # Create intervals
        starts = np.sort(positions)
        ends = starts + 1.0 / N
        
        # Merge intervals
        merged_starts = [starts[0]]
        merged_ends = [ends[0]]
        
        for i in range(1, N):
            if starts[i] <= merged_ends[-1]:
                # Overlapping or touching
                merged_ends[-1] = max(merged_ends[-1], ends[i])
            else:
                # Disjoint
                merged_starts.append(starts[i])
                merged_ends.append(ends[i])
                
        # Calculate total length
        return np.sum(np.array(merged_ends) - np.array(merged_starts))

    # Objective function for 1D optimization
    def objective_1d(pos):
        return union_length(pos)

    # Keich binary construction for initial guess
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    # Optimize 1D positions for x
    # We can use multiple restarts or just one good optimization
    # Given the small dimension (8), Nelder-Mead should be fast and effective.
    
    best_x = keich.copy()
    best_y = keich.copy()
    
    # Try to improve using local search
    # We might run a few different optimization attempts with slightly different initializations or settings
    # But starting from Keich is a strong start.
    
    if time.time() < deadline - 10:
        try:
            res_x = minimize(objective_1d, keich, method='Nelder-Mead', 
                             options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
            if res_x.success and res_x.fun < union_length(keich):
                best_x = res_x.x
        except Exception:
            pass

    if time.time() < deadline - 10:
        try:
            # We can try a slightly perturbed start for y to see if we find a different local minimum,
            # although the problem is symmetric, so it likely won't matter.
            # But let's just use the same optimization for y.
            res_y = minimize(objective_1d, keich, method='Nelder-Mead', 
                             options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
            if res_y.success and res_y.fun < union_length(keich):
                best_y = res_y.x
        except Exception:
            pass

    # Construct the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
