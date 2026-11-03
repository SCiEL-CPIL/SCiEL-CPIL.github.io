"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook rank
    schoolbook_rank = nm * mp * pn  # 42
    
    # Fallback to schoolbook
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_rank = schoolbook_rank
    best_result = None
    
    # Try ranks from 41 down to 35
    for rank in range(41, 34, -1):
        if time.time() > deadline - 20:
            break
        
        # Multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            seed = trial * 1000 + rank
            np.random.seed(seed)
            
            # Initialize with structured random
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS iterations
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Fix U, V, solve for W
                for c in range(pn):
                    # Compute U^T V for this c
                    UV = np.einsum('ar,br->ab', U, V)  # (nm, mp)
                    # Actually need to be more careful
                    # T[:, :, c] should be approximated by sum_r U[:,r] * V[:,r] * W[c,r]
                    # So for fixed c: T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
                    # This is a linear system in W[c,:]
                    
                    # Build matrix: for each r, we have U[:,r] outer V[:,r]
                    # We want: sum_r W[c,r] * (U[:,r] outer V[:,r]) = T[:,:,c]
                    
                    # Let M be a matrix where M[:, r] = (U[:,r] outer V[:,r]).flatten()
                    # Then M @ W[c,:] = T[:,:,c].flatten()
                    
                    M = np.zeros((nm * mp, rank))
                    for r in range(rank):
                        M[:, r] = (U[:, r] * V[:, r, np.newaxis]).flatten()
                    
                    T_c = T[:, :, c].flatten()
                    sol, _, _, _ = np.linalg.lstsq(M, T_c, rcond=None)
                    W[c, :] = sol
                
                # Fix W, V, solve for U
                for a in range(nm):
                    M = np.zeros((mp * pn, rank))
                    for r in range(rank):
                        M[:, r] = (V[:, r] * W[:, r, np.newaxis]).flatten()
                    
                    T_a = T[a, :, :].flatten()
                    sol, _, _, _ = np.linalg.lstsq(M, T_a, rcond=None)
                    U[a, :] = sol
                
                # Fix U, W, solve for V
                for b in range(mp):
                    M = np.zeros((nm * pn, rank))
                    for r in range(rank):
                        M[:, r] = (U[:, r] * W[:, r, np.newaxis]).flatten()
                    
                    T_b = T[:, b, :].flatten()
                    sol, _, _, _ = np.linalg.lstsq(M, T_b, rcond=None)
                    V[b, :] = sol
            
            # Check if the decomposition is valid
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.abs(T_recon - T).max()
            
            if err < 1e-8:
                if rank < best_rank:
                    best_rank = rank
                    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(nm)]
                    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(mp)]
                    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(pn)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found valid for this rank, try lower rank
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
