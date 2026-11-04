import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    trivial_rank = n * m * p  # 24
    
    # Build the trivial solution as fallback
    U_trivial = np.zeros((nm, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((mp, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((pn, trivial_rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
    
    best_rank = trivial_rank
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    
    # Helper: check if decomposition is valid (exact)
    def check_valid(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Helper: snap to nearest half-integer and recheck
    def snap_to_half_integer(U, V, W):
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s
    
    # Try to find lower rank decompositions
    # Strategy: random search with various initializations
    # Try ranks from lower bound up to trivial_rank - 1
    
    lower_bound = max(nm, mp, pn)  # 12
    
    # Precompute index arrays for faster evaluation
    # For each (i,j,k), we need U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    # Sum over r should equal 1
    
    # Try different rank values
    for target_rank in range(lower_bound, trivial_rank):
        if time.time() > deadline - 10.0:
            break
        
        # Multiple attempts per rank
        max_attempts = 2000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization with small values
            # Use a mix of strategies
            if attempt % 3 == 0:
                # Small random integers
                U = np.random.randint(-2, 3, size=(nm, target_rank)) * 0.5
                V = np.random.randint(-2, 3, size=(mp, target_rank)) * 0.5
                W = np.random.randint(-2, 3, size=(pn, target_rank)) * 0.5
            elif attempt % 3 == 1:
                # Gaussian
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
            else:
                # Sparse random
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    idx_u = np.random.randint(0, nm)
                    idx_v = np.random.randint(0, mp)
                    idx_w = np.random.randint(0, pn)
                    U[idx_u, r] = np.random.choice([-1, 1]) * 0.5
                    V[idx_v, r] = np.random.choice([-1, 1]) * 0.5
                    W[idx_w, r] = np.random.choice([-1, 1]) * 0.5
            
            # Quick check
            if check_valid(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
            
            # Try snapping to half-integers
            U_s, V_s, W_s = snap_to_half_integer(U, V, W)
            if check_valid(U_s, V_s, W_s, target_rank):
                best_rank = target_rank
                best_U = U_s
                best_V = V_s
                best_W = W_s
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better solution, use it; otherwise use trivial
    if best_rank < trivial_rank:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out = U_trivial
        V_out = V_trivial
        W_out = W_trivial
    
    # Convert to lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
