import numpy as np

def construct():
    n, m, p = 2, 5, 5
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # val: coefficient (1, -1, etc)
    # r: rank index
    def add_term(u_idx, v_idx, w_idx, val, r):
        U[u_idx][r] += val
        V[v_idx][r] += val
        W[w_idx][r] += val
        
    # We need to map logical (i, j, k) to indices
    # i in 0..1, j in 0..4, k in 0..4
    # U index: i*5 + j
    # V index: j*5 + k
    # W index: k*2 + i
    
    r = 0
    
    # Function to add trivial 2x1x1 (A_i, B_k) -> C_ik
    # This is just 1 term? No, 2x1x1 rank is 2.
    # C_11 = A_1 B_1, C_21 = A_2 B_1.
    # Term 1: A_1 B_1 -> C_11. U[0*5+j]=1, V[j*5+k]=1, W[k*2+0]=1.
    # Term 2: A_2 B_1 -> C_21. U[1*5+j]=1, V[j*5+k]=1, W[k*2+1]=1.
    def add_trivial_2x1x1(j, k, r_start):
        # i=0
        add_term(0*5 + j, j*5 + k, k*2 + 0, 1, r_start)
        # i=1
        add_term(1*5 + j, j*5 + k, k*2 + 1, 1, r_start+1)
        return r_start + 2

    # Function to add trivial 2x2x1 (A_i, B_j) -> C_i
    # A is 2x2, B is 2x1.
    # C_1 = A_11 B_1 + A_12 B_2
    # C_2 = A_21 B_1 + A_22 B_2
    # Rank 4.
    def add_trivial_2x2x1(j1, j2, k, r_start):
        # i=0, j=j1
        add_term(0*5 + j1, j1*5 + k, k*2 + 0, 1, r_start)
        # i=0, j=j2
        add_term(0*5 + j2, j2*5 + k, k*2 + 0, 1, r_start+1)
        # i=1, j=j1
        add_term(1*5 + j1, j1*5 + k, k*2 + 1, 1, r_start+2)
        # i=1, j=j2
        add_term(1*5 + j2, j2*5 + k, k*2 + 1, 1, r_start+3)
        return r_start + 4

    # Function to add trivial 2x1x2 (A_i, B_k) -> C_ik
    # A is 2x1, B is 1x2.
    # C_11 = A_1 B_1, C_12 = A_1 B_2
    # C_21 = A_2 B_1, C_22 = A_2 B_2
    # Rank 4.
    def add_trivial_2x1x2(j, k1, k2, r_start):
        # i=0, k=k1
        add_term(0*5 + j, j*5 + k1, k1*2 + 0, 1, r_start)
        # i=0, k=k2
        add_term(0*5 + j, j*5 + k2, k2*2 + 0, 1, r_start+1)
        # i=1, k=k1
        add_term(1*5 + j, j*5 + k1, k1*2 + 1, 1, r_start+2)
        # i=1, k=k2
        add_term(1*5 + j, j*5 + k2, k2*2 + 1, 1, r_start+3)
        return r_start + 4

    # Function to add Strassen 2x2x2
    # A is 2x2, B is 2x2.
    # j1, j2 are rows of B / cols of A.
    # k1, k2 are cols of B / cols of C.
    def add_strassen_2x2x2(j1, j2, k1, k2, r_start):
        # Standard Strassen for C = AB
        # C_11 = M1 - M4 + M6 - M7
        # C_12 = M3 + M5
        # C_21 = M2 + M4
        # C_22 = M1 - M2 + M3 + M7
        # M1 = (A_11 + A_22)(B_11 + B_22)
        # M2 = (A_21 + A_22)B_11
        # M3 = A_11(B_12 - B_22)
        # M4 = A_22(B_21 - B_11)
        # M5 = (A_11 + A_12)B_22
        # M6 = (A_21 - A_11)(B_11 + B_12)
        # M7 = (A_12 - A_22)(B_21 + B_22)
        
        # We need to map A_ii to U indices and B_jk to V indices.
        # A_11 -> U[0*5+j1], A_12 -> U[0*5+j2]
        # A_21 -> U[1*5+j1], A_22 -> U[1*5+j2]
        # B_11 -> V[j1*5+k1], B_12 -> V[j1*5+k2]
        # B_21 -> V[j2*5+k1], B_22 -> V[j2*5+k2]
        
        # M1
        r = r_start
        add_term(0*5+j1, j1*5+k1, k1*2+0, 1, r)
        add_term(0*5+j1, j1*5+k2, k2*2+0, 1, r)
        add_term(1*5+j1, j2*5+k1, k1*2+1, 1, r)
        add_term(1*5+j1, j2*5+k2, k2*2+1, 1, r)
        # Wait, M1 is (A_11+A_22)(B_11+B_22).
        # This expands to 4 terms in the tensor?
        # No, the rank-1 decomposition uses the products of sums.
        # The term for M1 in the decomposition is:
        # U part: A_11 + A_22. So U[0*5+j1]=1, U[1*5+j2]=1.
        # V part: B_11 + B_22. So V[j1*5+k1]=1, V[j2*5+k2]=1.
        # W part: C_11 + C_22? No.
        # The W part must reconstruct the output.
        # C_11 gets +M1, C_22 gets +M1.
        # So W[k1*2+0]=1, W[k2*2+1]=1.
        
        # Let's re-verify the Strassen reconstruction.
        # C_11 = M1 - M4 + M6 - M7
        # C_12 = M3 + M5
        # C_21 = M2 + M4
        # C_22 = M1 - M2 + M3 + M7
        
        # M1:
        # U: A_11 (0,j1) + A_22 (1,j2)
        # V: B_11 (j1,k1) + B_22 (j2,k2)
        # W: C_11 (k1,0) + C_22 (k2,1)
        add_term(0*5+j1, j1*5+k1, k1*2+0, 1, r_start)
        add_term(0*5+j1, j1*5+k2, k2*2+0, 1, r_start)
        add_term(1*5+j2, j2*5+k1, k1*2+1, 1, r_start)
        add_term(1*5+j2, j2*5+k2, k2*2+1, 1, r_start)
        # This is not a single rank-1 term. It's 4 rank-1 terms?
        # No, the "term" in the decomposition is a single rank-1 tensor.
        # The rank-1 tensor is $u \otimes v \otimes w$.
        # If $u = e_{0,j1} + e_{1,j2}$, $v = e_{j1,k1} + e_{j2,k2}$, $w = e_{k1,0} + e_{k2,1}$.
        # Then $u \otimes v \otimes w$ has 4 non-zero entries in the expanded form?
        # No, it is a single vector in the space of tensors.
        # In our matrix representation U, V, W, a single rank-1 term corresponds to:
        # One column in U, one in V, one in W.
        # The column in U can have multiple non-zero entries.
        # So, for M1:
        # U_col: U[0*5+j1]=1, U[1*5+j2]=1.
        # V_col: V[j1*5+k1]=1, V[j2*5+k2]=1.
        # W_col: W[k1*2+0]=1, W[k2*2+1]=1.
        # This is ONE rank index.
        
        r = r_start
        U[0*5+j1][r] = 1
        U[1*5+j2][r] = 1
        V[j1*5+k1][r] = 1
        V[j2*5+k2][r] = 1
        W[k1*2+0][r] = 1
        W[k2*2+1][r] = 1
        r += 1
        
        # M2 = (A_21 + A_22)B_11
        # U: A_21 (1,j1) + A_22 (1,j2)
        # V: B_11 (j1,k1)
        # W: C_21 (k1,1)
        U[1*5+j1][r] = 1
        U[1*5+j2][r] = 1
        V[j1*5+k1][r] = 1
        W[k1*2+1][r] = 1
        r += 1
        
        # M3 = A_11(B_12 - B_22)
        # U: A_11 (0,j1)
        # V: B_12 (j1,k2) - B_22 (j2,k2)
        # W: C_12 (k2,0)
        U[0*5+j1][r] = 1
        V[j1*5+k2][r] = 1
        V[j2*5+k2][r] = -1
        W[k2*2+0][r] = 1
        r += 1
        
        # M4 = A_22(B_21 - B_11)
        # U: A_22 (1,j2)
        # V: B_21 (j2,k1) - B_11 (j1,k1)
        # W: C_11 (k1,0) - C_21 (k1,1)
        # Wait, C_11 gets -M4, C_21 gets +M4.
        # So W: -e_{k1,0} + e_{k1,1}
        U[1*5+j2][r] = 1
        V[j2*5+k1][r] = 1
        V[j1*5+k1][r] = -1
        W[k1*2+0][r] = -1
        W[k1*2+1][r] = 1
        r += 1
        
        # M5 = (A_11 + A_12)B_22
        # U: A_11 (0,j1) + A_12 (0,j2)
        # V: B_22 (j2,k2)
        # W: C_12 (k2,0)
        U[0*5+j1][r] = 1
        U[0*5+j2][r] = 1
        V[j2*5+k2][r] = 1
        W[k2*2+0][r] = 1
        r += 1
        
        # M6 = (A_21 - A_11)(B_11 + B_12)
        # U: A_21 (1,j1) - A_11 (0,j1)
        # V: B_11 (j1,k1) + B_12 (j1,k2)
        # W: C_11 (k1,0)
        U[1*5+j1][r] = 1
        U[0*5+j1][r] = -1
        V[j1*5+k1][r] = 1
        V[j1*5+k2][r] = 1
        W[k1*2+0][r] = 1
        r += 1
        
        # M7 = (A_12 - A_22)(B_21 + B_22)
        # U: A_12 (0,j2) - A_22 (1,j2)
        # V: B_21 (j2,k1) + B_22 (j2,k2)
        # W: C_11 (k1,0) - C_22 (k2,1)
        U[0*5+j2][r] = 1
        U[1*5+j2][r] = -1
        V[j2*5+k1][r] = 1
        V[j2*5+k2][r] = 1
        W[k1*2+0][r] = 1
        W[k2*2+1][r] = -1
        r += 1
        
        return r

    # Block 1: j=0,1; k=0,1
    r = add_strassen_2x2x2(0, 1, 0, 1, r)
    
    # Block 2: j=0,1; k=2,3,4
    # k=2: 2x2x1
    r = add_trivial_2x2x1(0, 1, 2, r)
    # k=3,4: 2x2x2
    r = add_strassen_2x2x2(0, 1, 3, 4, r)
    
    # Block 3: j=2,3,4; k=0,1
    # j=2: 2x1x2
    r = add_trivial_2x1x2(2, 0, 1, r)
    # j=3,4: 2x2x2
    r = add_strassen_2x2x2(3, 4, 0, 1, r)
    
    # Block 4: j=2,3,4; k=2,3,4
    # j=2: 2x1x3
    # k=2: 2x1x1
    r = add_trivial_2x1x1(2, 2, r)
    # k=3: 2x1x1
    r = add_trivial_2x1x1(2, 3, r)
    # k=4: 2x1x1
    r = add_trivial_2x1x1(2, 4, r)
    # j=3,4: 2x2x3
    # k=2: 2x2x1
    r = add_trivial_2x2x1(3, 4, 2, r)
    # k=3,4: 2x2x2
    r = add_strassen_2x2x2(3, 4, 3, 4, r)
    
    return {"U": U, "V": V, "W": W}
