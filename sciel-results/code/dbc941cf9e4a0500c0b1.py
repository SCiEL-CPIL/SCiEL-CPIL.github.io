"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_strassen_21():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        
    return U, V, W

def _verify_full(U, V, W, n, m, p, rank):
    """
    Strict verification of the tensor decomposition.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # Check diagonal: should be 1
                s_diag = 0
                for r in range(rank):
                    s_diag += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                if not np.isclose(s_diag, 1.0):
                    return False
                
                # Check off-diagonal: should be 0
                for i_p in range(n):
                    for j_p in range(p):
                        if i_p == i and j_p == j:
                            continue
                        s_off = 0
                        for r in range(rank):
                            s_off += U[i*m+k][r] * V[k*p+j][r] * W[j_p*n+i_p][r]
                        if not np.isclose(s_off, 0.0):
                            return False
    return True

def _try_remove_product_full(U, V, W, n, m, p, deadline):
    rank = 21
    nU = n * m
    nV = m * p
    nW = p * n
    
    # Build the full tensor constraint matrix
    # We want to find c such that sum_{r != removed} c_r T_r = T_removed
    # T_r is a vector of length n*m*p (24)
    # T_r[i*m+k, k*p+j, j*n+i] ? No, T_r is indexed by (i,k,j)
    # The tensor entry is at (i,k,j).
    # We flatten (i,k,j) to a single index.
    
    num_eq = n * m * p
    M = np.zeros((num_eq, rank))
    T = np.zeros(num_eq)
    
    # Fill M with the tensor vectors for each term r
    for r in range(rank):
        eq_idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    M[eq_idx, r] = val
                    eq_idx += 1
    
    # The target tensor T is 1 for all (i,k,j)
    # Wait, the target tensor is T[i,k,j] = 1.
    # So T vector is all 1s.
    T[:] = 1.0
    
    for removed in range(rank):
        if time.time() > deadline:
            break
        
        idx = [r for r in range(rank) if r != removed]
        M_sub = M[:, idx]
        
        # We want M_sub @ c = T_removed
        # But T_removed is the vector of the removed term.
        # T_removed = M[:, removed]
        b = M[:, removed]
        
        try:
            c, residuals, rank_svd, sv = np.linalg.lstsq(M_sub, b, rcond=None)
        except:
            continue
        
        residual = M_sub @ c - b
        if np.max(np.abs(residual)) < 1e-9:
            # Construct new matrices
            new_rank = rank - 1
            new_U = [[0] * new_rank for _ in range(nU)]
            new_V = [[0] * new_rank for _ in range(nV)]
            new_W = [[0] * new_rank for _ in range(nW)]
            
            new_idx = 0
            for r in range(rank):
                if r == removed:
                    continue
                coeff = c[new_idx]
                for a in range(nU):
                    new_U[a][new_idx] = U[a][r]
                for b_idx in range(nV):
                    new_V[b_idx][new_idx] = V[b_idx][r]
                for c_idx in range(nW):
                    new_W[c_idx][new_idx] = W[c_idx][r] * coeff
                new_idx += 1
            
            if _verify_full(new_U, new_V, new_W, n, m, p, new_rank):
                return new_U, new_V, new_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U, V, W = _build_strassen_21()
    
    # Try to find rank-20 using full tensor constraints
    result = _try_remove_product_full(U, V, W, n, m, p, deadline)
    if result is not None:
        new_U, new_V, new_W = result
        if _verify_full(new_U, new_V, new_W, n, m, p, 20):
            return {"U": new_U, "V": new_V, "W": new_W}
    
    # Fallback to rank-21
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
