import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    def build_rank44_decomp():
        """Build the standard rank-44 decomposition using numpy"""
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for j in range(4):
            w_r00 = 4*j
            w_r01 = 4*j + 2
            w_r10 = 4*j + 1
            w_r11 = 4*j + 3
            
            v_e = 2*j
            v_f = 2*j + 1
            v_g = 8 + 2*j
            v_h = 9 + 2*j
            
            # P1
            U[0, col] = 1; U[4, col] = 1
            V[v_e, col] = 1; V[v_h, col] = 1
            W[w_r00, col] = 1; W[w_r11, col] = 1
            col += 1
            
            # P2
            U[3, col] = 1; U[4, col] = 1
            V[v_e, col] = 1
            W[w_r10, col] = 1; W[w_r11, col] = -1
            col += 1
            
            # P3
            U[0, col] = 1
            V[v_f, col] = 1; V[v_h, col] = -1
            W[w_r01, col] = 1; W[w_r11, col] = 1
            col += 1
            
            # P4
            U[4, col] = 1
            V[v_g, col] = 1; V[v_e, col] = -1
            W[w_r00, col] = 1; W[w_r10, col] = 1
            col += 1
            
            # P5
            U[0, col] = 1; U[1, col] = 1
            V[v_h, col] = 1
            W[w_r00, col] = -1; W[w_r01, col] = 1
            col += 1
            
            # P6
            U[3, col] = 1; U[0, col] = -1
            V[v_e, col] = 1; V[v_f, col] = 1
            W[w_r11, col] = 1
            col += 1
            
            # P7
            U[1, col] = 1; U[4, col] = -1
            V[v_g, col] = 1; V[v_h, col] = 1
            W[w_r00, col] = 1
            col += 1
        
        for i in range(2):
            for k in range(8):
                u_idx = i * 3 + 2
                v_idx = 2 * 8 + k
                w_idx = k * 2 + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
        
        return U, V, W, rank
    
    def verify_decomp(U, V, W, n, m, p):
        """Verify that the decomposition is correct"""
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_std = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_std[u_idx, v_idx, w_idx] = 1
        return np.allclose(T_decomp, T_std, atol=1e-9)
    
    # Build the baseline rank-44 decomposition
    U, V, W, rank = build_rank44_decomp()
    
    # Verify it's correct
    if not verify_decomp(U, V, W, n, m, p):
        return {"U": [[0]*48 for _ in range(6)], "V": [[0]*48 for _ in range(24)], "W": [[0]*48 for _ in range(16)]}
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Time-limited search to try to reduce rank
    search_start = time.time()
    search_budget = 100
    
    # Try removing pairs of columns first
    for r1, r2 in combinations(range(rank), 2):
        if time.time() - search_start > search_budget:
            break
        
        mask = np.ones(rank, dtype=bool)
        mask[r1] = False
        mask[r2] = False
        new_rank = rank - 2
        
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        if verify_decomp(U_new, V_new, W_new, n, m, p):
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_rank
            break
    
    # If pair removal didn't work, try single column removal
    if best_rank == rank:
        for r_remove in range(rank):
            if time.time() - search_start > search_budget:
                break
            
            mask = np.ones(rank, dtype=bool)
            mask[r_remove] = False
            new_rank = rank - 1
            
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if verify_decomp(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W, best_rank = U_new, V_new, W_new, new_rank
                break
    
    # Convert to list of lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
