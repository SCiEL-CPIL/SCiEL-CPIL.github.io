"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # Decomposition: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U, V, W, rank):
        """Verify if the decomposition is exact with integer/half-integer entries."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check if entries are integer or half-integer
        # Multiply by 2, check if integer
        if not np.allclose(U_arr * 2, np.round(U_arr * 2)): return False
        if not np.allclose(V_arr * 2, np.round(V_arr * 2)): return False
        if not np.allclose(W_arr * 2, np.round(W_arr * 2)): return False
        
        # Reconstruct tensor
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Using einsum for speed
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check exact match
        if not np.allclose(T_recon, target, atol=1e-9):
            return False
        return True

    def try_rank(r, seed=0):
        """Try to find a rank-r decomposition via optimization and snapping."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        # Scale to roughly match the magnitude of the solution
        # Schoolbook has 1s. Let's start with small random values.
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        def loss(params):
            if time.time() > deadline:
                return 1e10 # Timeout penalty
            
            U_flat = params[:nm * r]
            V_flat = params[nm * r : nm * r + mp * r]
            W_flat = params[nm * r + mp * r :]
            
            U_p = U_flat.reshape((nm), r)
            V_p = V_flat.reshape((mp), r)
            W_p = W_flat.reshape((pn), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - target) ** 2)
            return err

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 1000})
        except Exception:
            return None
            
        if res.fun < 1e-6:
            # Extract solution
            U_sol = res.x[:nm * r].reshape((nm), r)
            V_sol = res.x[nm * r : nm * r + mp * r].reshape((mp), r)
            W_sol = res.x[nm * r + mp * r :].reshape((pn), r)
            
            # Snap to nearest half-integer
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            if verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), r):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
            # If direct snap fails, try a local discrete search around the snapped values
            # This is a heuristic: perturb some entries by +/- 0.5 and check
            # Given the time constraints, we only do this if we have plenty of time left
            if time.time() < deadline - 20:
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                # Simple local search: try changing one element at a time? Too slow.
                # Instead, try different rounding modes or small perturbations
                # For now, we rely on the optimizer finding a solution close enough to snap.
                # If not, we return None.
        
        return None

    # Fallback schoolbook
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank
    # Lower bound is max(nm, mp, pn) = 25.
    # Known bounds for M_{2,5,5} might be higher. Let's try 30 to 49.
    # We iterate from lower to higher, but optimization is hard.
    # It might be better to try a few specific ranks with multiple seeds.
    
    ranks_to_try = [35, 40, 45, 48, 49]
    seeds = [42, 123, 999, 7, 13]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        for seed in seeds:
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed)
            if result is not None:
                best_result = result
                best_rank = r
                break
        if best_rank < best_rank: # This check is redundant if we break, but safe
            break

    return best_result
# EVOLVE-BLOCK-END
