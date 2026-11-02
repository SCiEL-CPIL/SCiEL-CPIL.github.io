import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9

    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Strategy: Exploit block structure
    # Split middle dim m=3 into m1=1 (am=0) and m2=2 (am=1,2)
    # A = [A1 | A2], B = [B1; B2]
    # C = A1*B1 + A2*B2
    # A1*B1: (2x1)*(1x9) -> rank 2
    # A2*B2: (2x2)*(2x9) -> use Strassen on column pairs
    
    # Indexing:
    # U index: i = ai*m + am = ai*3 + am, ai in {0,1}, am in {0,1,2}
    # V index: j = am*p + bm = am*9 + bm, am in {0,1,2}, bm in {0..8}
    # W index: k = bm*n + ai = bm*2 + ai, bm in {0..8}, ai in {0,1}
    
    max_rank = 40
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    col = 0
    
    # Part 1: am=0, (2x1)*(1x9), rank 2
    # U index for (ai, am=0): ai*3 + 0
    # V index for (am=0, bm): 0*9 + bm = bm
    # W index for (bm, ai): bm*2 + ai
    
    # Term 0: ai=0
    U[0*3+0, col] = 1
    for bm in range(9):
        V[0*9+bm, col] = 1
        W[bm*2+0, col] = 1
    col += 1
    
    # Term 1: ai=1
    U[1*3+0, col] = 1
    for bm in range(9):
        V[0*9+bm, col] = 1
        W[bm*2+1, col] = 1
    col += 1
    
    # Part 2: am=1,2, (2x2)*(2x9)
    # U indices: (ai, am) where am in {1,2}
    #   (0,1): 0*3+1=1, (0,2): 0*3+2=2, (1,1): 1*3+1=4, (1,2): 1*3+2=5
    u_a = 1  # (ai=0, am=1)
    u_b = 2  # (ai=0, am=2)
    u_c = 4  # (ai=1, am=1)
    u_d = 5  # (ai=1, am=2)
    
    # V indices: (am, bm) where am in {1,2}
    #   (1,bm): 1*9+bm = 9+bm, (2,bm): 2*9+bm = 18+bm
    # W indices: (bm, ai)
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 9 + k0   # (am=1, bm=k0)
        v_f = 9 + k1   # (am=1, bm=k1)
        v_g = 18 + k0  # (am=2, bm=k0)
        v_h = 18 + k1  # (am=2, bm=k1)
        
        w_00 = k0 * 2 + 0
        w_01 = k1 * 2 + 0
        w_10 = k0 * 2 + 1
        w_11 = k1 * 2 + 1
        
        # Strassen 7 terms for 2x2*2x2
        # M1 = (a+d)(e+h)
        U[u_a, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1
        W[w_00, col] += 1
        W[w_01, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[u_c, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M3 = a(f-h)
        U[u_a, col] += 1
        V[v_f, col] += 1
        V[v_h, col] -= 1
        W[w_00, col] += 1
        W[w_01, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[u_d, col] += 1
        V[v_g, col] += 1
        V[v_e, col] -= 1
        W[w_10, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[u_a, col] += 1
        U[u_b, col] += 1
        V[v_h, col] += 1
        W[w_01, col] += 1
        W[w_11, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[u_c, col] += 1
        U[u_a, col] -= 1
        V[v_e, col] += 1
        V[v_f, col] += 1
        W[w_10, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[u_b, col] += 1
        U[u_d, col] -= 1
        V[v_g, col] += 1
        V[v_h, col] += 1
        W[w_11, col] += 1
        col += 1
    
    # Single column bm=8
    k8 = 8
    v_e = 9 + k8   # (am=1, bm=8)
    v_g = 18 + k8  # (am=2, bm=8)
    w_08 = k8 * 2 + 0
    w_18 = k8 * 2 + 1
    
    # 2x2*2x1: rank 4 (schoolbook)
    U[u_a, col] += 1
    V[v_e, col] += 1
    W[w_08, col] += 1
    col += 1
    
    U[u_b, col] += 1
    V[v_g, col] += 1
    W[w_08, col] += 1
    col += 1
    
    U[u_c, col] += 1
    V[v_e, col] += 1
    W[w_18, col] += 1
    col += 1
    
    U[u_d, col] += 1
    V[v_g, col] += 1
    W[w_18, col] += 1
    col += 1
    
    # Total: 2 + 28 + 4 = 34 terms
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    current_rank = col
    
    # Verify
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(decomp, T, atol=1e-10):
        # Fallback to naive
        rank_naive = 54
        U = np.zeros((n*m, rank_naive))
        V = np.zeros((m*p, rank_naive))
        W = np.zeros((p*n, rank_naive))
        for am in range(3):
            for bm in range(9):
                for ai in range(2):
                    idx = am*18 + bm*2 + ai
                    U[ai*3+am, idx] = 1
                    V[am*9+bm, idx] = 1
                    W[bm*2+ai, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    # Greedy column removal
    while True:
        if time.time() > deadline - 20:
            break
        
        improved = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 20:
                break
            
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            
            decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            residual = np.sum((decomp_new - T)**2)
            
            if residual < 1e-8:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break
        
        if not improved:
            break
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
