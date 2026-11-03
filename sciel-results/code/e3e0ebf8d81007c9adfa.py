"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of columns (vectors)
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Block 1: Strassen for j in {0,1}, k in {0,1}
    # Indices for A: i in {0,1}, j in {0,1} -> 4 entries
    # Indices for B: j in {0,1}, k in {0,1} -> 4 entries
    # Indices for C: k in {0,1}, i in {0,1} -> 4 entries
    
    # Strassen vectors for the 2x2x2 subproblem
    # u, v, w are length 4 vectors corresponding to the 4 entries of the submatrices
    # Submatrix A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Submatrix B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Submatrix C indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    u_strassen = [
        [1, 0, 1, 0], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 0, 1, 0], # M5: A00+A10
        [-1, 0, 0, 1],# M6: A11-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    v_strassen = [
        [1, 0, 1, 0], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 0, 1, -1],# M3: B10-B11
        [0, 1, -1, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 0, 1, 0], # M6: B00+B10
        [0, 1, 0, 1]  # M7: B11+B01
    ]
    
    w_strassen = [
        [1, 0, 0, 1], # C00+=1, C11+=1
        [0, 1, 0, -1],# C10+=1, C11-=1
        [0, 0, 1, 1], # C01+=1, C11+=1
        [1, 1, 0, 0], # C00+=1, C10+=1
        [-1, 0, 1, 0],# C00-=1, C01+=1
        [0, 0, 0, 1], # C11+=1
        [-1, 0, 1, 0] # C00-=1, C01+=1
    ]
    
    for idx in range(7):
        u_sub = u_strassen[idx]
        v_sub = v_strassen[idx]
        w_sub = w_strassen[idx]
        
        u_full = [0] * (n * m)
        v_full = [0] * (m * p)
        w_full = [0] * (p * n)
        
        # Map sub-indices to full indices
        # A sub (i, j) where i,j in {0,1}
        # Full A index: i * m + j
        for i in range(2):
            for j in range(2):
                sub_idx = i * 2 + j
                full_idx = i * m + j
                u_full[full_idx] = u_sub[sub_idx]
                
        # B sub (j, k) where j,k in {0,1}
        # Full B index: j * p + k
        for j in range(2):
            for k in range(2):
                sub_idx = j * 2 + k
                full_idx = j * p + k
                v_full[full_idx] = v_sub[sub_idx]
                
        # C sub (k, i) where k,i in {0,1}
        # Full C index: k * n + i
        for k in range(2):
            for i in range(2):
                sub_idx = k * 2 + i
                full_idx = k * n + i
                w_full[full_idx] = w_sub[sub_idx]
                
        U_cols.append(u_full)
        V_cols.append(v_full)
        W_cols.append(w_full)
        
    # Block 2: Schoolbook for j in {0,1}, k in {2,3,4,5}
    for j in range(2):
        for k in range(2, p):
            for i in range(n):
                add_schoolbook(i, j, k)
                
    # Block 3: Schoolbook for j in {2,3,4}, k in {0,1}
    for j in range(2, m):
        for k in range(2):
            for i in range(n):
                add_schoolbook(i, j, k)
                
    # Block 4: Schoolbook for j in {2,3,4}, k in {2,3,4,5}
    for j in range(2, m):
        for k in range(2, p):
            for i in range(n):
                add_schoolbook(i, j, k)
                
    # Transpose to get U (n*m x rank), V (m*p x rank), W (p*n x rank)
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r in range(n * m):
            U[r][c] = U_cols[c][r]
        for r in range(m * p):
            V[r][c] = V_cols[c][r]
        for r in range(p * n):
            W[r][c] = W_cols[c][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
