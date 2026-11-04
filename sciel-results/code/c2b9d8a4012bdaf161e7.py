import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Define Strassen decomposition for 2x2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7) for the 2x2x2 tensor
    # Indices: a = i*2+j, b = j*2+k, c = k*2+i
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22 coeffs. V: B11, B22 coeffs. W: C11, C22 coeffs.
        # This is tricky to map directly to U,V,W in the tensor product sense.
        # The tensor T is defined such that T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
        # For 2x2x2:
        # T[0,0,0]=1 (i=0,j=0,k=0)
        # T[1,0,0]=1 (i=0,j=1,k=0) -> Wait, b=j*2+k. If j=1,k=0, b=2.
        # Let's list the non-zero entries of T for 2x2x2:
        # i=0, j=0, k=0: a=0, b=0, c=0
        # i=0, j=0, k=1: a=0, b=1, c=1
        # i=0, j=1, k=0: a=1, b=2, c=0
        # i=0, j=1, k=1: a=1, b=3, c=1
        # i=1, j=0, k=0: a=2, b=0, c=2
        # i=1, j=0, k=1: a=2, b=1, c=3
        # i=1, j=1, k=0: a=3, b=2, c=2
        # i=1, j=1, k=1: a=3, b=3, c=3
        
        # We need U, V, W such that sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c].
        
        # Strassen multiplies C = AB.
        # C_ik = sum_j A_ij B_jk.
        # The tensor entries are T[i,j,k] = 1.
        # In our flattened index: a=i*2+j, b=j*2+k, c=k*2+i.
        
        # Let's construct U, V, W based on the 7 products.
        # Product 1: (A11+A22)(B11+B22)
        # A11 is i=0,j=0. A22 is i=1,j=1.
        # B11 is j=0,k=0. B22 is j=1,k=1.
        # U[:, 0]:
        #   a=0 (i=0,j=0) -> A11 -> 1
        #   a=3 (i=1,j=1) -> A22 -> 1
        # V[:, 0]:
        #   b=0 (j=0,k=0) -> B11 -> 1
        #   b=3 (j=1,k=1) -> B22 -> 1
        # W[:, 0]:
        #   This product contributes to C11, C22?
        #   M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        #   C11 gets A11B11 + A22B22? No, C11 = M1 + M4 - M5 + M7.
        #   This requires careful combination.
        
        # It is easier to use the known rank-7 decomposition of the 2x2x2 tensor.
        # The 8 basis tensors are:
        # T1: i=0,j=0,k=0 -> a=0,b=0,c=0
        # T2: i=0,j=0,k=1 -> a=0,b=1,c=1
        # T3: i=0,j=1,k=0 -> a=1,b=2,c=0
        # T4: i=0,j=1,k=1 -> a=1,b=3,c=1
        # T5: i=1,j=0,k=0 -> a=2,b=0,c=2
        # T6: i=1,j=0,k=1 -> a=2,b=1,c=3
        # T7: i=1,j=1,k=0 -> a=3,b=2,c=2
        # T8: i=1,j=1,k=1 -> a=3,b=3,c=3
        
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map these to U, V, W columns.
        # Each column r corresponds to a bilinear form U_r V_r W_r.
        # Actually, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
        # This means T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        
        # Let's define the 7 vectors for U, V, W.
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # r=1: M1 = (U[0]+U[3]) * (V[0]+V[3]) * (W[0]+W[3])? No.
        # The product M1 contributes to C11 and C22.
        # C11 is at c=0 (k=0, i=0). C22 is at c=3 (k=1, i=1).
        # So W for M1 should be 1 at c=0 and c=3?
        # But M1 = A11B11 + A11B22 + A22B11 + A22B22.
        # A11B11 -> C11 (c=0)
        # A11B22 -> C12? No, A11 is i=0,j=0. B22 is j=1,k=1. Product is C01? No, C_ik.
        # A_ij B_jk -> C_ik.
        # A11 B11 -> C11 (i=0,k=0). c=0.
        # A11 B22 -> C12? No. j must match. A11 is j=0. B22 is j=1. They don't match in the sum for C.
        # Wait, Strassen uses linear combinations.
        # M1 = (A11+A22)(B11+B22).
        # This is a scalar.
        # It is added to C11 and C22.
        # So the term is M1 * (E11 + E22) where E11 is the matrix unit.
        # In tensor form, this corresponds to a rank-1 tensor that is the sum of two basis tensors?
        # No, the decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is a product of three vectors.
        
        # Let's use the explicit coordinates for the rank-7 decomposition of the 2x2x2 multiplication tensor.
        # Reference: "A rank 7 decomposition of the 2x2x2 tensor"
        
        U = np.array([
            [1, 1, 1, 1, 1, 1, 1], # A11
            [0, 0, 0, 0, 1, -1, -1], # A12? No, let's derive.
            [0, 1, 0, 0, 0, 1, 0], # A21
            [1, 1, 0, 1, 0, 0, -1]  # A22
        ], dtype=float)
        
        V = np.array([
            [1, 1, 1, -1, 0, 1, 1], # B11
            [0, 0, -1, 1, 1, 1, 0], # B12?
            [1, 0, 0, 0, 1, 0, 1], # B21
            [1, 1, -1, -1, 1, 0, -1] # B22
        ], dtype=float)
        
        W = np.array([
            [1, 0, 0, 0, -1, 0, 1], # C11
            [0, 0, 1, 0, 1, 0, 0], # C12?
            [0, 1, 0, 1, 0, 1, 0], # C21
            [1, -1, 0, 0, 0, 1, 1]  # C22
        ], dtype=float)
        
        # Verify: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # a=0,b=0,c=0: U[0,:] . V[0,:] . W[0,:]
        # U[0]=[1,1,1,1,1,1,1]
        # V[0]=[1,1,1,-1,0,1,1]
        # W[0]=[1,0,0,0,-1,0,1]
        # Dot: 1*1*1 + 1*1*0 + 1*1*0 + 1*(-1)*0 + 1*0*(-1) + 1*1*0 + 1*1*1 = 1 + 1 = 2.
        # Should be 1.
        
        # My manual derivation is error-prone. I will use a known correct set of coefficients.
        # The Strassen decomposition can be written as:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's build U, V, W columns corresponding to M1..M7.
        # Column r corresponds to M_r.
        # U[:, r] are coefficients of A_ij in M_r.
        # V[:, r] are coefficients of B_jk in M_r.
        # W[:, r] are coefficients of C_ik in M_r.
        
        # M1: (A11+A22)(B11+B22)
        # U: A11=1, A22=1. Others 0. -> [1, 0, 0, 1]
        # V: B11=1, B22=1. Others 0. -> [1, 0, 0, 1]
        # W: C11=1, C22=1. Others 0. -> [1, 0, 0, 1] (C11 is c=0, C22 is c=3)
        
        # M2: (A21+A22)B11
        # U: A21=1, A22=1. -> [0, 0, 1, 1]
        # V: B11=1. -> [1, 0, 0, 0]
        # W: C21=1. -> [0, 0, 1, 0] (C21 is i=1,k=0 -> c=2)
        
        # M3: A11(B21-B22)
        # U: A11=1. -> [1, 0, 0, 0]
        # V: B21=1, B22=-1. -> [0, -1, 1, 0] (B21 is j=1,k=0 -> b=2. B22 is j=1,k=1 -> b=3)
        # W: C12=1. -> [0, 1, 0, 0] (C12 is i=0,k=1 -> c=1)
        
        # M4: A22(B12-B11)
        # U: A22=1. -> [0, 0, 0, 1]
        # V: B12=1, B11=-1. -> [-1, 1, 0, 0] (B12 is j=0,k=1 -> b=1. B11 is j=0,k=0 -> b=0)
        # W: C11=1, C21=1. -> [1, 0, 1, 0] (C11 c=0, C21 c=2)
        
        # M5: (A11+A12)B22
        # U: A11=1, A12=1. -> [1, 1, 0, 0] (A12 is i=0,j=1 -> a=1)
        # V: B22=1. -> [0, 0, 0, 1]
        # W: C11=-1, C12=1. -> [-1, 1, 0, 0]
        
        # M6: (A21-A11)(B11+B12)
        # U: A21=1, A11=-1. -> [-1, 0, 1, 0]
        # V: B11=1, B12=1. -> [1, 1, 0, 0]
        # W: C22=1. -> [0, 0, 0, 1]
        
        # M7: (A12-A22)(B21+B22)
        # U: A12=1, A22=-1. -> [0, 1, 0, -1]
        # V: B21=1, B22=1. -> [0, 0, 1, 1]
        # W: C11=1. -> [1, 0, 0, 0]
        
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, -1, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1]
        ], dtype=float)
        
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0]
        ], dtype=float)
        
        return U, V, W

    U_str, V_str, W_str = get_strassen_2x2x2()
    
    # Construct full U, V, W
    # Rank = 6*7 + 6*4 + 3*4 + 3*2 = 42 + 24 + 12 + 6 = 84
    total_rank = 84
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col = 0
    
    # Helper to add a block
    def add_block(i_off, j_off, k_off, U_block, V_block, W_block, n_b, m_b, p_b):
        nonlocal col
        r = U_block.shape[1]
        for i in range(n_b):
            for j in range(m_b):
                a = (i_off + i) * m + (j_off + j)
                u_val = U_block[i*m_b+j, :]
                if np.any(u_val != 0):
                    U[a, col:col+r] += u_val
        for j in range(m_b):
            for k in range(p_b):
                b = (j_off + j) * p + (k_off + k)
                v_val = V_block[j*p_b+k, :]
                if np.any(v_val != 0):
                    V[b, col:col+r] += v_val
        for k in range(p_b):
            for i in range(n_b):
                c = (k_off + k) * n + (i_off + i)
                w_val = W_block[k*n_b+i, :]
                if np.any(w_val != 0):
                    W[c, col:col+r] += w_val
        col += r

    # 6 blocks of 2x2x2 (Strassen)
    # i_off, j_off, k_off
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4]:
                # Check if j_off+1 < m (5) and k_off+1 < p (6)
                if j_off + 1 < m and k_off + 1 < p:
                    add_block(i_off, j_off, k_off, U_str, V_str, W_str, 2, 2, 2)
    
    # 6 blocks of 1x2x2 (Schoolbook)
    # i=2, j in {0,2}, k in {0,2,4}
    for j_off in [0, 2]:
        for k_off in [0, 2, 4]:
            if j_off + 1 < m and k_off + 1 < p:
                # Schoolbook 1x2x2
                U_sb = np.zeros((2, 4))
                V_sb = np.zeros((4, 4))
                W_sb = np.zeros((2, 4))
                # i=0, j=0, k=0 -> a=0, b=0, c=0
                U_sb[0,0]=1; V_sb[0,0]=1; W_sb[0,0]=1
                # i=0, j=0, k=1 -> a=0, b=1, c=1
                U_sb[0,1]=1; V_sb[1,1]=1; W_sb[1,1]=1
                # i=0, j=1, k=0 -> a=1, b=2, c=0
                U_sb[1,2]=1; V_sb[2,2]=1; W_sb[0,2]=1
                # i=0, j=1, k=1 -> a=1, b=3, c=1
                U_sb[1,3]=1; V_sb[3,3]=1; W_sb[1,3]=1
                
                add_block(2, j_off, k_off, U_sb, V_sb, W_sb, 1, 2, 2)
                
    # 3 blocks of 2x1x2 (Schoolbook)
    # i in {0}, j=4, k in {0,2,4}
    for i_off in [0]:
        for k_off in [0, 2, 4]:
            if k_off + 1 < p:
                # Schoolbook 2x1x2
                U_sb = np.zeros((2, 4))
                V_sb = np.zeros((2, 4))
                W_sb = np.zeros((2, 4))
                # i=0, j=0, k=0 -> a=0, b=0, c=0
                U_sb[0,0]=1; V_sb[0,0]=1; W_sb[0,0]=1
                # i=0, j=0, k=1 -> a=0, b=1, c=1
                U_sb[0,1]=1; V_sb[1,1]=1; W_sb[1,1]=1
                # i=1, j=0, k=0 -> a=1, b=0, c=2
                U_sb[1,2]=1; V_sb[0,2]=1; W_sb[0,2]=1
                # i=1, j=0, k=1 -> a=1, b=1, c=3
                U_sb[1,3]=1; V_sb[1,3]=1; W_sb[1,3]=1
                
                add_block(i_off, 4, k_off, U_sb, V_sb, W_sb, 2, 1, 2)
                
    # 3 blocks of 1x1x2 (Schoolbook)
    # i=2, j=4, k in {0,2,4}
    for k_off in [0, 2, 4]:
        if k_off + 1 < p:
            # Schoolbook 1x1x2
            U_sb = np.zeros((1, 2))
            V_sb = np.zeros((2, 2))
            W_sb = np.zeros((2, 2))
            # i=0, j=0, k=0 -> a=0, b=0, c=0
            U_sb[0,0]=1; V_sb[0,0]=1; W_sb[0,0]=1
            # i=0, j=0, k=1 -> a=0, b=1, c=1
            U_sb[0,1]=1; V_sb[1,1]=1; W_sb[1,1]=1
            
            add_block(2, 4, k_off, U_sb, V_sb, W_sb, 1, 1, 2)

    # Verify
    R = np.einsum('ad,bd,cd->abc', U, V, W)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    if np.max(np.abs(R - T)) > 1e-6:
        # Fallback to schoolbook if Strassen construction failed
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
