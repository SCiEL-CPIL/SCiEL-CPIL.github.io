import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices
    # I: 0,1 -> I1; 2,3 -> I2
    # J: 0,1 -> J1; 2,3 -> J2; 4 -> J3
    # K: 0,1 -> K1; 2,3 -> K2; 4,5 -> K3
    
    # We will build U, V, W as lists of lists (or numpy arrays)
    # Start with empty lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i_start, j_start, k_start):
        # Adds a 2x2x2 Strassen block
        # i_start, j_start, k_start are indices in global I, J, K
        # Block size 2x2x2
        # Strassen intermediates M1..M7
        
        # M1 = (A11+A22)(B11+B22)
        # A part: A11, A22. B part: B11, B22.
        # U: A11, A22. V: B11, B22.
        # This is not direct. Strassen computes C from A and B.
        # We need to express C_ik = sum M_c * (A terms) * (B terms).
        
        # It is easier to generate the U, V, W for the 7 terms directly.
        # Each term M_c is a product of linear combinations of A and B.
        # We can expand each M_c into 4 scalar products? No.
        # Strassen is a tensor decomposition of rank 7.
        # We can hardcode the 7 terms.
        
        # Term 1: (A11+A22)(B11+B22)
        # This expands to A11B11 + A11B22 + A22B11 + A22B22.
        # But in the decomposition, it's ONE term in the sum for C.
        # Wait, the rank is 7, meaning we have 7 columns in U, V, W.
        # Column c corresponds to M_c.
        # U_c is the vector of coefficients for A in M_c.
        # V_c is the vector of coefficients for B in M_c.
        # W_c is the vector of coefficients for C in M_c.
        
        # M1 = (A11+A22)(B11+B22)
        # U1: A11=1, A22=1, others 0.
        # V1: B11=1, B22=1, others 0.
        # W1: C11=1, C22=1, C12=-1, C21=-1? No.
        # C11 = M1 - M4 + M5
        # C12 = M3 + M7
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6
        
        # So for M1:
        # W1: C11=1, C22=1.
        
        # We need to map local indices (0,1) to global indices.
        
        pass

    # This approach is complex to code generically.
    # Given time, I will output the 108 rank solution.
    
    # Let's just use the 108 rank structure.
    # I will implement the 108 rank solution.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a column
    def add_col(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1

    # Strassen 2x2x2
    # Local indices 0,1 for i,j,k
    # Global i = i0 + li, j = j0 + lj, k = k0 + lk
    
    def add_strassen(i0, j0, k0):
        # 7 terms
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C11=1.
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        # A is 2x2. A[li, lj]. Index li*m + lj? No, A is n x m.
        # A[i, j].
        u[i0*m + j0] = 1
        u[(i0+1)*m + (j0+1)] = 1
        v[j0*p + k0] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = 1
        w[(k0+1)*n + (i0+1)] = 1
        add_col(u, v, w)
        
        # M2: (A10+A11)B00 -> C10+=1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m + j0] = 1
        u[(i0+1)*m + (j0+1)] = 1
        v[j0*p + k0] = 1
        w[k0*n + (i0+1)] = 1
        add_col(u, v, w)
        
        # M3: A00(B01-B11) -> C01+=1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m + j0] = 1
        v[j0*p + (k0+1)] = 1
        v[(j0+1)*p + (k0+1)] = -1
        w[(k0+1)*n + i0] = 1
        add_col(u, v, w)
        
        # M4: A11(B10-B00) -> C00-=1, C10+=1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m + (j0+1)] = 1
        v[(j0+1)*p + k0] = 1
        v[j0*p + k0] = -1
        w[k0*n + i0] = -1
        w[k0*n + (i0+1)] = 1
        add_col(u, v, w)
        
        # M5: (A00+A01)B11 -> C00+=1, C11+=1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m + j0] = 1
        u[i0*m + (j0+1)] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[k0*n + i0] = 1
        w[(k0+1)*n + (i0+1)] = 1
        add_col(u, v, w)
        
        # M6: (A10-A00)(B00+B01) -> C11+=1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m + j0] = 1
        u[i0*m + j0] = -1
        v[j0*p + k0] = 1
        v[j0*p + (k0+1)] = 1
        w[(k0+1)*n + (i0+1)] = 1
        add_col(u, v, w)
        
        # M7: (A01-A11)(B10+B11) -> C01+=1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m + (j0+1)] = 1
        u[(i0+1)*m + (j0+1)] = -1
        v[(j0+1)*p + k0] = 1
        v[(j0+1)*p + (k0+1)] = 1
        w[(k0+1)*n + i0] = 1
        add_col(u, v, w)

    def add_rank4(i0, j0, k0):
        # 2x1x2 block. J size 1.
        # A is 2x1. B is 1x2.
        # C is 2x2.
        # 4 terms.
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m + j0] = 1
        v[j0*p + k0] = 1
        w[k0*n + i0] = 1
        add_col(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i0*m + j0] = 1
        v[j0*p + (k0+1)] = 1
        w[(k0+1)*n + i0] = 1
        add_col(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m + j0] = 1
        v[j0*p + k0] = 1
        w[k0*n + (i0+1)] = 1
        add_col(u, v, w)
        
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i0+1)*m + j0] = 1
        v[j0*p + (k0+1)] = 1
        w[(k0+1)*n + (i0+1)] = 1
        add_col(u, v, w)

    # Partitions
    # I: 0,1 (I1), 2,3 (I2)
    # J: 0,1 (J1), 2,3 (J2), 4 (J3)
    # K: 0,1 (K1), 2,3 (K2), 4,5 (K3)
    
    I_splits = [(0, 2), (2, 2)] # start, size
    J_splits = [(0, 2), (2, 2), (4, 1)]
    K_splits = [(0, 2), (2, 2), (4, 2)]
    
    for i_start, i_size in I_splits:
        for k_start, k_size in K_splits:
            # For each output block, sum over J
            for j_start, j_size in J_splits:
                if j_size == 2:
                    add_strassen(i_start, j_start, k_start)
                elif j_size == 1:
                    add_rank4(i_start, j_start, k_start)
                    
    # Convert to lists of lists for output
    U_out = [[0]*rank for _ in range(n*m)]
    V_out = [[0]*rank for _ in range(m*p)]
    W_out = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for i in range(n*m):
            U_out[i][c] = U[c][i]
        for j in range(m*p):
            V_out[j][c] = V[c][j]
        for k in range(p*n):
            W_out[k][c] = W[c][k]
            
    return {"U": U_out, "V": V_out, "W": W_out}
