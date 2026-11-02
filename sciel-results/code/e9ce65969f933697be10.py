import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = 2, 5, 6
    U_cols = []
    V_cols = []
    W_cols = []

    # Helper to add a tensor term: sum over c of u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    # We will build U, V, W as lists of rows (each row is a list of rank entries)
    # But it's easier to accumulate as lists of columns (vectors) and then transpose

    # We'll use a recursive or iterative block decomposition strategy
    # Split m=5 into 2+3.
    # Case A: i in [0,1], j in [0,1], k in [0,5] -> 2x2 by 2x6
    # Case B: i in [0,1], j in [2,4], k in [0,5] -> 2x3 by 3x6

    # Function to add a rank-1 term (or small block term)
    def add_term(u_row, v_row, w_row, val=1):
        # u_row is index in U (i*m+j), v_row in V (j*p+k), w_row in W (k*n+i)
        U_cols.append([0]*1 + [0]*0) # placeholder, will handle later
        # Actually, let's just append to lists of scalars for each row later.
        # Better: keep a list of (u_idx, v_idx, w_idx, coeff)
        pass

    # Let's use a different approach: collect (u_idx, v_idx, w_idx, coeff) tuples
    terms = []

    def add(u_idx, v_idx, w_idx, coeff=1.0):
        terms.append((u_idx, v_idx, w_idx, coeff))

    # --- Case A: 2x2 by 2x6 ---
    # Split k into 3+3.
    # A1: 2x2 by 2x3 (k=0,1,2)
    # A2: 2x2 by 2x3 (k=3,4,5)

    # For 2x2 by 2x3, we can use Strassen for the first 2x2 part and schoolbook for the last 1 column.
    # Strassen for 2x2 by 2x2 (k=0,1 relative to start):
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # Let's map global indices.
    # i=0,1; j=0,1; k=0..5

    # Block A1: k=0,1,2.
    # Sub-block A1a: 2x2 by 2x2 (k=0,1). Use Strassen.
    # Local indices: i_l=0,1; j_l=0,1; k_l=0,1.
    # Global i = i_l, j = j_l, k = k_l.
    # Strassen terms for 2x2x2:
    # 1: (i=0,j=0, coeff=1 for A) * (i=0,j=0 for B? No, Strassen mixes indices)
    # Let's stick to the explicit tensor decomposition of Strassen.
    # Strassen rank is 7.
    # The 7 basis tensors for 2x2x2 multiplication:
    # T1: A11(B11+B22) -> U[0,1], V[0,1], W[0,1], W[1,0]? No.
    # Let's just use the known 7 rank-1 tensors for 2x2x2.
    # From literature:
    # 1. U: [1,0,0,1], V: [1,0,0,1], W: [1,0,0,1]  (A11*B11 + A22*B22? No, that's not right)
    #
    # Let's derive the 7 terms for 2x2x2 manually or use a known table.
    # C = AB
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    #
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let's expand these into rank-1 terms (u, v, w) where u is 2x2 vector (flattened), v is 2x2, w is 2x2 (flattened as w[k*n+i]).
    # Note: W index is k*n+i. For n=2, w[0,0] is idx 0, w[1,0] is idx 2, w[0,1] is idx 1, w[1,1] is idx 3.
    # Wait, W[k*n+i].
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    #
    # U index: i*m+j. m=2.
    # i=0,j=0 -> 0
    # i=0,j=1 -> 1
    # i=1,j=0 -> 2
    # i=1,j=1 -> 3
    #
    # V index: j*p+k. p=2.
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 2
    # j=1,k=1 -> 3
    #
    # Term 1: M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11 -> U[0]=1, U[3]=1.
    # B part: B00 + B11 -> V[0]=1, V[3]=1.
    # C part: M1 contributes to C00 (+1), C11 (+1).
    # C00: k=0,i=0 -> W[0]=1.
    # C11: k=1,i=1 -> W[3]=1.
    # So Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1].
    
    # Term 2: M2 = (A10+A11)B00
    # A: A10+A11 -> U[2]=1, U[3]=1.
    # B: B00 -> V[0]=1.
    # C: M2 contributes to C10 (+1).
    # C10: k=0,i=1 -> W[1]=1.
    # Term 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,0].

    # Term 3: M3 = A00(B10-B11)
    # A: A00 -> U[0]=1.
    # B: B10-B11 -> V[2]=1, V[3]=-1.
    # C: M3 contributes to C01 (+1), C11 (+1).
    # C01: k=1,i=0 -> W[2]=1.
    # C11: k=1,i=1 -> W[3]=1.
    # Term 3: U=[1,0,0,0], V=[0,0,1,-1], W=[0,0,1,1].

    # Term 4: M4 = A11(B01-B00)
    # A: A11 -> U[3]=1.
    # B: B01-B00 -> V[1]=1, V[0]=-1.
    # C: M4 contributes to C00 (-1), C10 (+1).
    # C00: W[0]=-1.
    # C10: W[1]=1.
    # Term 4: U=[0,0,0,1], V=[-1,1,0,0], W=[-1,1,0,0].

    # Term 5: M5 = (A00+A01)B11
    # A: A00+A01 -> U[0]=1, U[1]=1.
    # B: B11 -> V[3]=1.
    # C: M5 contributes to C00 (-1), C01 (+1).
    # C00: W[0]=-1.
    # C01: W[2]=1.
    # Term 5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,0,1,0].

    # Term 6: M6 = (A10-A00)(B00+B01)
    # A: A10-A00 -> U[2]=1, U[0]=-1.
    # B: B00+B01 -> V[0]=1, V[1]=1.
    # C: M6 contributes to C11 (+1).
    # C11: W[3]=1.
    # Term 6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1].

    # Term 7: M7 = (A01-A11)(B10+B11)
    # A: A01-A11 -> U[1]=1, U[3]=-1.
    # B: B10+B11 -> V[2]=1, V[3]=1.
    # C: M7 contributes to C00 (+1).
    # C00: W[0]=1.
    # Term 7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0].

    def add_strassen_block(i_off, j_off, k_off):
        # Adds the 7 terms for a 2x2x2 block starting at (i_off, j_off, k_off)
        # U indices: (i_off+di)*m + (j_off+dj)
        # V indices: (j_off+dj)*p + (k_off+dk)
        # W indices: (k_off+dk)*n + (i_off+di)
        
        # Local 0,1 -> Global +off
        # Term 1
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+0)] = 1
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+0)*p + (k_off+0)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+0)*n + (i_off+0)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_from_vecs(u, v, w, 1.0)

        # Term 2
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m + (j_off+0)] = 1
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+0)*p + (k_off+0)] = 1
        w[(k_off+0)*n + (i_off+1)] = 1
        add_from_vecs(u, v, w, 1.0)

        # Term 3
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+0)] = 1
        v[(j_off+1)*p + (k_off+0)] = 1
        v[(j_off+1)*p + (k_off+1)] = -1
        w[(k_off+1)*n + (i_off+0)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_from_vecs(u, v, w, 1.0)

        # Term 4
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m + (j_off+1)] = 1
        v[(j_off+0)*p + (k_off+0)] = -1
        v[(j_off+0)*p + (k_off+1)] = 1
        w[(k_off+0)*n + (i_off+0)] = -1
        w[(k_off+0)*n + (i_off+1)] = 1
        add_from_vecs(u, v, w, 1.0)

        # Term 5
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+0)] = 1
        u[(i_off+0)*m + (j_off+1)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+0)*n + (i_off+0)] = -1
        w[(k_off+1)*n + (i_off+0)] = 1
        add_from_vecs(u, v, w, 1.0)

        # Term 6
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+0)] = -1
        u[(i_off+1)*m + (j_off+0)] = 1
        v[(j_off+0)*p + (k_off+0)] = 1
        v[(j_off+0)*p + (k_off+1)] = 1
        w[(k_off+1)*n + (i_off+1)] = 1
        add_from_vecs(u, v, w, 1.0)

        # Term 7
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+0)*m + (j_off+1)] = 1
        u[(i_off+1)*m + (j_off+1)] = -1
        v[(j_off+1)*p + (k_off+0)] = 1
        v[(j_off+1)*p + (k_off+1)] = 1
        w[(k_off+0)*n + (i_off+0)] = 1
        add_from_vecs(u, v, w, 1.0)

    def add_from_vecs(u, v, w, coeff):
        # Add a single rank-1 term defined by full vectors
        terms.append((u, v, w, coeff))

    # Now, build the full decomposition.
    # We will accumulate columns.
    # First, collect all (u, v, w, c) tuples.
    terms = []
    
    # Redefine add to store tuples
    def add_term(u, v, w, c):
        terms.append((u, v, w, c))

    # Case A: i in 0,1; j in 0,1; k in 0..5
    # A1: k=0,1,2.
    # A1a: 2x2x2 at (0,0,0). Strassen.
    add_strassen_block(0, 0, 0)
    # A1b: 2x2x1 at (0,0,2). Schoolbook rank 4.
    # i=0,1; j=0,1; k=2.
    # 4 terms: (i,j,k)
    for i in [0,1]:
        for j in [0,1]:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[i*m + j] = 1
            v[j*p + 2] = 1
            w[2*n + i] = 1
            add_term(u, v, w, 1.0)
            
    # A2: k=3,4,5.
    # A2a: 2x2x2 at (0,0,3). Strassen.
    add_strassen_block(0, 0, 3)
    # A2b: 2x2x1 at (0,0,5). Schoolbook rank 4.
    for i in [0,1]:
        for j in [0,1]:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[i*m + j] = 1
            v[j*p + 5] = 1
            w[5*n + i] = 1
            add_term(u, v, w, 1.0)

    # Case B: i in 0,1; j in 2,3,4; k in 0..5
    # Split k into 3+3.
    # B1: k=0,1,2. 2x3x3.
    # Split j into 2+1.
    # B1a: j=2,3; k=0,1,2. 2x2x3.
    #   B1a-i: 2x2x2 at (0,2,0). Strassen.
    add_strassen_block(0, 2, 0)
    #   B1a-ii: 2x2x1 at (0,2,2). Schoolbook rank 4.
    for i in [0,1]:
        for j in [2,3]:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[i*m + j] = 1
            v[j*p + 2] = 1
            w[2*n + i] = 1
            add_term(u, v, w, 1.0)
    # B1b: j=4; k=0,1,2. 2x1x3. Rank 2*1*3 = 6.
    for i in [0,1]:
        for k in [0,1,2]:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[i*m + 4] = 1
            v[4*p + k] = 1
            w[k*n + i] = 1
            add_term(u, v, w, 1.0)

    # B2: k=3,4,5. 2x3x3.
    # B2a: j=2,3; k=3,4,5. 2x2x3.
    #   B2a-i: 2x2x2 at (0,2,3). Strassen.
    add_strassen_block(0, 2, 3)
    #   B2a-ii: 2x2x1 at (0,2,5). Schoolbook rank 4.
    for i in [0,1]:
        for j in [2,3]:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[i*m + j] = 1
            v[j*p + 5] = 1
            w[5*n + i] = 1
            add_term(u, v, w, 1.0)
    # B2b: j=4; k=3,4,5. 2x1x3. Rank 6.
    for i in [0,1]:
        for k in [3,4,5]:
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[i*m + 4] = 1
            v[4*p + k] = 1
            w[k*n + i] = 1
            add_term(u, v, w, 1.0)

    # Now we have a list of terms. Each term is (u, v, w, c).
    # We need to combine them into matrices U, V, W of shape (dim, rank).
    # We can just treat each term as a column.
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for col_idx, (u, v, w, c) in enumerate(terms):
        for i in range(n*m):
            if u[i] != 0:
                U[i][col_idx] = u[i] * c
        for j in range(m*p):
            if v[j] != 0:
                V[j][col_idx] = v[j] * c
        for k in range(p*n):
            if w[k] != 0:
                W[k][col_idx] = w[k] * c

    return {"U": U, "V": V, "W": W}
