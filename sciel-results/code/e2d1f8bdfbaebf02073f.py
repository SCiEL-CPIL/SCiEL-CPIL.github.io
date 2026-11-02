import numpy as np

def construct():
    n, m, p = 2, 2, 11
    # Rank calculation: 5 * 7 (Strassen) + 4 (Schoolbook) = 39
    rank = 39
    
    # U: (n*m, rank) = (4, 39)
    # V: (m*p, rank) = (22, 39)
    # W: (p*n, rank) = (22, 39)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col_idx = 0
    
    # Indices mapping helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen decomposition for 2x2 x 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Contributions to C:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We process 5 pairs of columns
    # Columns: 0..10 (0-indexed)
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    # Single: 10
    
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    for ka, kb in pairs:
        # For each of the 7 Strassen terms
        # Define A coeffs, B coeffs, C coeffs for each term
        
        # M1
        # A: 11=1, 22=1
        # B: 1,ka=1, 2,kb=1
        # C: 1,ka=1, 2,kb=1
        U[u_idx(0,0)][col_idx] = 1
        U[u_idx(1,1)][col_idx] = 1
        V[v_idx(0,ka)][col_idx] = 1
        V[v_idx(1,kb)][col_idx] = 1
        W[w_idx(ka,0)][col_idx] = 1
        W[w_idx(kb,1)][col_idx] = 1
        col_idx += 1
        
        # M2
        # A: 21=1, 22=1
        # B: 1,ka=1
        # C: 2,ka=1
        U[u_idx(1,0)][col_idx] = 1
        U[u_idx(1,1)][col_idx] = 1
        V[v_idx(0,ka)][col_idx] = 1
        W[w_idx(ka,1)][col_idx] = 1
        col_idx += 1
        
        # M3
        # A: 11=1
        # B: 1,kb=1, 2,kb=-1
        # C: 1,kb=1
        U[u_idx(0,0)][col_idx] = 1
        V[v_idx(0,kb)][col_idx] = 1
        V[v_idx(1,kb)][col_idx] = -1
        W[w_idx(kb,0)][col_idx] = 1
        col_idx += 1
        
        # M4
        # A: 22=1
        # B: 2,ka=1, 1,ka=-1
        # C: 1,ka=1, 2,ka=1
        U[u_idx(1,1)][col_idx] = 1
        V[v_idx(1,ka)][col_idx] = 1
        V[v_idx(0,ka)][col_idx] = -1
        W[w_idx(ka,0)][col_idx] = 1
        W[w_idx(ka,1)][col_idx] = 1
        col_idx += 1
        
        # M5
        # A: 11=1, 12=1
        # B: 2,kb=1
        # C: 1,ka=-1, 1,kb=1
        U[u_idx(0,0)][col_idx] = 1
        U[u_idx(0,1)][col_idx] = 1
        V[v_idx(1,kb)][col_idx] = 1
        W[w_idx(ka,0)][col_idx] = -1
        W[w_idx(kb,0)][col_idx] = 1
        col_idx += 1
        
        # M6
        # A: 21=1, 11=-1
        # B: 1,ka=1, 1,kb=1
        # C: 2,kb=1
        U[u_idx(1,0)][col_idx] = 1
        U[u_idx(0,0)][col_idx] = -1
        V[v_idx(0,ka)][col_idx] = 1
        V[v_idx(0,kb)][col_idx] = 1
        W[w_idx(kb,1)][col_idx] = 1
        col_idx += 1
        
        # M7
        # A: 12=1, 22=-1
        # B: 2,ka=1, 2,kb=1
        # C: 1,ka=1, 2,kb=-1
        U[u_idx(0,1)][col_idx] = 1
        U[u_idx(1,1)][col_idx] = -1
        V[v_idx(1,ka)][col_idx] = 1
        V[v_idx(1,kb)][col_idx] = 1
        W[w_idx(ka,0)][col_idx] = 1
        W[w_idx(kb,1)][col_idx] = -1
        col_idx += 1
        
    # Single column 10
    ks = 10
    # A11 B1,ks -> C1,ks
    U[u_idx(0,0)][col_idx] = 1
    V[v_idx(0,ks)][col_idx] = 1
    W[w_idx(ks,0)][col_idx] = 1
    col_idx += 1
    
    # A12 B2,ks -> C1,ks
    U[u_idx(0,1)][col_idx] = 1
    V[v_idx(1,ks)][col_idx] = 1
    W[w_idx(ks,0)][col_idx] = 1
    col_idx += 1
    
    # A21 B1,ks -> C2,ks
    U[u_idx(1,0)][col_idx] = 1
    V[v_idx(0,ks)][col_idx] = 1
    W[w_idx(ks,1)][col_idx] = 1
    col_idx += 1
    
    # A22 B2,ks -> C2,ks
    U[u_idx(1,1)][col_idx] = 1
    V[v_idx(1,ks)][col_idx] = 1
    W[w_idx(ks,1)][col_idx] = 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
