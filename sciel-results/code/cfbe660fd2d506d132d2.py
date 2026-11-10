# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    Maximizes H_{-1} / max(H_0, H_1, H_2).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds safety margin
    
    # Precompute projection indices
    # P is indexed as P[y, x]
    # For a given slope r, the value is v = x + r*y
    # We need to sum probabilities over all (x, y) such that x + r*y == v
    
    # Slopes:
    # r = -1: v = x - y. Range: -9 to 9. (19 values)
    # r = 0:  v = x.    Range: 0 to 9.  (10 values)
    # r = 1:  v = x + y. Range: 0 to 18. (19 values)
    # r = 2:  v = x + 2y. Range: 0 to 27. (28 values)
    
    # Let's define the "bins" for each slope.
    # For a fixed slope r, the bin index for cell (x, y) is:
    # idx = x + int(r * y) if r is integer.
    # Since all r are integers (-1, 0, 1, 2), we can just use integer arithmetic.
    
    # r = -1: bin = x - y. Offset by 9 to make non-negative? Or just use dictionary.
    # Using dictionaries or arrays is faster.
    
    # Let's create lookup tables for bin indices for each slope.
    # shape (10, 10) -> bin_index
    
    # r = -1
    bin_r_minus1 = np.zeros((n, n), dtype=int)
    for y in range(n):
        for x in range(n):
            bin_r_minus1[y, x] = x - y + 9  # shift to 0..18
    
    # r = 0
    bin_r_0 = np.zeros((n, n), dtype=int)
    for y in range(n):
        for x in range(n):
            bin_r_0[y, x] = x # 0..9
    
    # r = 1
    bin_r_1 = np.zeros((n, n), dtype=int)
    for y in range(n):
        for x in range(n):
            bin_r_1[y, x] = x + y # 0..18
    
    # r = 2
    bin_r_2 = np.zeros((n, n), dtype=int)
    for y in range(n):
        for x in range(n):
            bin_r_2[y, x] = x + 2*y # 0..27
    
    num_bins = {
        -1: 19,
        0: 10,
        1: 19,
        2: 28
    }
    
    def compute_entropies(P_flat):
        """
        P_flat: 1D array of length 100, representing P[y, x] row-major.
        Returns (H_minus1, H_max, ratio)
        """
        P = P_flat.reshape((n, n))
        
        # Clip and normalize (optimization step should keep it valid, but safe)
        P = np.clip(P, 1e-12, None)
        s = P.sum()
        if s < 1e-9:
            return 0, 1, 0
        P = P / s
        
        # Compute projected distributions
        # H_{-1}
        proj_minus1 = np.zeros(num_bins[-1])
        np.add.at(proj_minus1, bin_r_minus1.ravel(), P.ravel())
        H_minus1 = -np.sum(proj_minus1[proj_minus1 > 1e-15] * np.log2(proj_minus1[proj_minus1 > 1e-15]))
        
        # H_0
        proj_0 = np.zeros(num_bins[0])
        np.add.at(proj_0, bin_r_0.ravel(), P.ravel())
        H_0 = -np.sum(proj_0[proj_0 > 1e-15] * np.log2(proj_0[proj_0 > 1e-15]))
        
        # H_1
        proj_1 = np.zeros(num_bins[1])
        np.add.at(proj_1, bin_r_1.ravel(), P.ravel())
        H_1 = -np.sum(proj_1[proj_1 > 1e-15] * np.log2(proj_1[proj_1 > 1e-15]))
        
        # H_2
        proj_2 = np.zeros(num_bins[2])
        np.add.at(proj_2, bin_r_2.ravel(), P.ravel())
        H_2 = -np.sum(proj_2[proj_2 > 1e-15] * np.log2(proj_2[proj_2 > 1e-15]))
        
        H_max = max(H_0, H_1, H_2)
        
        if H_max < 1e-9:
            return H_minus1, H_max, 0
        
        ratio = H_minus1 / H_max
        return H_minus1, H_max, ratio

    def objective(params):
        """
        Minimize negative ratio.
        params: 99 parameters (sum to 1 is enforced by softmax or similar? 
        Better to use simplex constraint or just normalize inside).
        Using log-transform helps keep positivity.
        Let params be log-probabilities? No, simpler to just normalize inside objective.
        However, gradient of normalization is annoying.
        Let's use a transformation: P_i = exp(z_i) / sum(exp(z_j)).
        This ensures P_i > 0 and sum P_i = 1.
        """
        z = params
        # Softmax for stability
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        P = exp_z / np.sum(exp_z)
        
        H_minus1, H_max, ratio = compute_entropies(P)
        
        if not np.isfinite(ratio):
            return 1e10
        return -ratio

    # Initial guess: Uniform
    # z_i = log(1/100) for all i
    z0 = np.full(100, np.log(1/100.0))
    
    best_score = 0
    best_params = z0.copy()
    
    # Try a few random starts and optimizations
    n_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
            
        if i == 0:
            z_init = z0
        else:
            # Random perturbation
            z_init = z0 + rng.normal(0, 0.5, 100)
            
        try:
            res = minimize(objective, z_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            score = -res.fun
            
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
            
    # Refine with Nelder-Mead on the top candidate? Or just stick with L-BFGS-B.
    # Let's do a final check.
    
    # Convert best_params to P
    z_max = np.max(best_params)
    exp_z = np.exp(best_params - z_max)
    P = exp_z / np.sum(exp_z)
    
    # Ensure validity
    P = np.clip(P, 0, None)
    P = P / P.sum()
    
    return P

# EVOLVE-BLOCK-END
