import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def strassen_222():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    n=2, m=2, p=2.
    U: (4, 7), V: (4, 7), W: (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A11 + A22)(B11 + B22)
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    
    # c=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1 # Wait, W is p*n. k*n+i.
    # C11 = M1... C12 = ...
    # M1 contributes to C11, C12, C21, C22?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map W correctly.
    # W[k*n + i, c]
    # k=0, i=0 -> C11. k=0, i=1 -> C12. k=1, i=0 -> C21. k=1, i=1 -> C22.
    
    # M1:
    W[0, 0] = 1  # C11
    W[1, 0] = 1  # C12? No, C12 = M3 + M5. M1 is not in C12.
    # Correction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 is in C11 and C22.
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # M2 is in C21 and C22.
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # M3 is in C12 and C22.
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # M4 is in C11 and C21.
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # M5 is in C12 and C11.
    W[1, 4] = 1  # C12
    W[0, 4] = -1 # C11
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # M6 is in C22.
    W[3, 5] = 1  # C22
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # M7 is in C11.
    W[0, 6] = 1  # C11
    
    return U, V, W

def schoolbook_221():
    """
    Returns U, V, W for 2x2x1 schoolbook decomposition.
    n=2, m=2, p=1.
    U: (4, 4), V: (2, 4), W: (2, 4).
    """
    n, m, p = 2, 2, 1
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def tensor_product(U1, V1, W1, U2, V2, W2):
    n1, m1, p1 = U1.shape[0] // m1, m1, V1.shape[0] // m1
    # Better to pass dimensions
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # 1. Strassen 2x2x2
    U_s, V_s, W_s = strassen_222()
    n_s, m_s, p_s = 2, 2, 2
    r_s = 7
    
    # 2. Schoolbook 2x2x1
    U_sb, V_sb, W_sb = schoolbook_221()
    n_sb, m_sb, p_sb = 2, 2, 1
    r_sb = 4
    
    # 3. Decomposition for 2x2x3: Rank 11
    # B (2x3) = B1 (2x2) + B2 (2x1)
    # C (2x3) = C1 (2x2) + C2 (2x1)
    # U_223 = [U_s, U_sb] (4 x 11)
    # V_223: V_s (2x2 part) and V_sb (2x1 part)
    # V_s is 2x2 (m=2, p=2). V_sb is 2x1 (m=2, p=1).
    # V_223 is 2x3.
    # Rows of V_223: j*3 + k.
    # V_s rows: j*2 + k1. k1 in 0,1.
    # V_sb rows: j*1 + k2. k2 in 0.
    # Mapping:
    # k_global 0 -> k1 0
    # k_global 1 -> k1 1
    # k_global 2 -> k2 0
    
    U_223 = np.hstack([U_s, U_sb])
    V_223 = np.zeros((2 * 3, 11))
    W_223 = np.zeros((3 * 2, 11))
    
    # Fill V_223
    for j in range(2):
        for k1 in range(2):
            V_223[j * 3 + k1, 0:7] = V_s[j * 2 + k1, 0:7]
        V_223[j * 3 + 2, 7:11] = V_sb[j * 1 + 0, 7:11]
        
    # Fill W_223
    # W_s is 2x2 (p=2, n=2). W_sb is 1x2 (p=1, n=2).
    # W_223 is 3x2 (p=3, n=2).
    # Rows of W_223: k*2 + i.
    # W_s rows: k1*2 + i.
    # W_sb rows: k2*2 + i.
    # Mapping:
    # k_global 0 -> k1 0
    # k_global 1 -> k1 1
    # k_global 2 -> k2 0
    
    for k1 in range(2):
        for i in range(2):
            W_223[k1 * 2 + i, 0:7] = W_s[k1 * 2 + i, 0:7]
    for i in range(2):
        W_223[2 * 2 + i, 7:11] = W_sb[0 * 2 + i, 7:11]
        
    # 4. Decomposition for 4x4x4: Rank 49
    # Tensor product of two Strassen 2x2x2
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            u_val = U_s[i1 * 2 + j1, c1] * U_s[i2 * 2 + j2, c2]
                            if u_val != 0:
                                U_444[(i1 * 2 + i2) * 4 + (j1 * 2 + j2), c] += u_val
                                
                            v_val = V_s[j1 * 2 + (0), c1] # Wait, V_s index is j*p + k. p=2.
                            # V_s[j1 * 2 + k1, c1]
                            # We need to sum over k1, k2? No, tensor product is element-wise product.
                            # V_444 index: j*4 + k.
                            # j = j1 * 2 + j2. k = k1 * 2 + k2.
                            # V_444[j*4 + k, c] += V_s[j1*2 + k1, c1] * V_s[j2*2 + k2, c2]
                            
                            # This loop is too slow. Use matrix operations or better loops.
    
    # Faster tensor product construction
    def tensor_product_matrices(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                u1 = U1[:, c1]
                u2 = U2[:, c2]
                v1 = V1[:, c1]
                v2 = V2[:, c2]
                w1 = W1[:, c1]
                w2 = W2[:, c2]
                
                # U
                # Index (i1 n2 + i2) m1 m2 + (j1 m2 + j2)
                # u1[i1 m1 + j1] * u2[i2 m2 + j2]
                # Reshape u1 to (n1, m1), u2 to (n2, m2)
                u1_mat = u1.reshape(n1, m1)
                u2_mat = u2.reshape(n2, m2)
                u_prod = np.outer(u1_mat, u2_mat) # (n1, m1, n2, m2)
                u_prod = u_prod.transpose(0, 2, 1, 3) # (n1, n2, m1, m2)
                u_flat = u_prod.reshape(n * m)
                U[:, c] = u_flat
                
                # V
                # Index (j1 m2 + j2) p1 p2 + (k1 p2 + k2)
                # v1[j1 p1 + k1] * v2[j2 p2 + k2]
                v1_mat = v1.reshape(m1, p1)
                v2_mat = v2.reshape(m2, p2)
                v_prod = np.outer(v1_mat, v2_mat) # (m1, p1, m2, p2)
                v_prod = v_prod.transpose(0, 2, 1, 3) # (m1, m2, p1, p2)
                v_flat = v_prod.reshape(m * p)
                V[:, c] = v_flat
                
                # W
                # Index (k1 p2 + k2) n1 n2 + (i1 n2 + i2)
                # w1[k1 n1 + i1] * w2[k2 n2 + i2]
                w1_mat = w1.reshape(p1, n1)
                w2_mat = w2.reshape(p2, n2)
                w_prod = np.outer(w1_mat, w2_mat) # (p1, n1, p2, n2)
                w_prod = w_prod.transpose(0, 2, 1, 3) # (p1, p2, n1, n2)
                w_flat = w_prod.reshape(p * n)
                W[:, c] = w_flat
                
        return U, V, W

    U_444, V_444, W_444 = tensor_product_matrices(U_s, V_s, W_s, U_s, V_s, W_s, 2, 2, 2, 2, 2, 2)
    
    # 5. Decomposition for 4x4x3: Rank 44
    # Tensor product of Schoolbook 2x2x1 and Decomp 2x2x3
    U_443, V_443, W_443 = tensor_product_matrices(U_sb, V_sb, W_sb, U_223, V_223, W_223, 2, 2, 1, 2, 2, 3)
    
    # 6. Combine 444 and 443
    U = np.hstack([U_444, U_443])
    V = np.zeros((28, 93))
    W = np.zeros((28, 93))
    
    # V_444 is 16x49. Rows j*4 + k1. j in 0..3, k1 in 0..3.
    # Global V rows j*7 + k.
    # For k < 4: k = k1.
    # For k >= 4: k = 4 + k2. k2 in 0..2.
    
    for j in range(4):
        for k1 in range(4):
            V[j * 7 + k1, 0:49] = V_444[j * 4 + k1, 0:49]
        for k2 in range(3):
            V[j * 7 + 4 + k2, 49:93] = V_443[j * 3 + k2, 49:93]
            
    # W_444 is 16x49. Rows k1*4 + i. k1 in 0..3, i in 0..3.
    # W_443 is 12x44. Rows k2*4 + i. k2 in 0..2, i in 0..3.
    # Global W rows k*4 + i.
    # For k < 4: k = k1.
    # For k >= 4: k = 4 + k2.
    
    for k1 in range(4):
        for i in range(4):
            W[k1 * 4 + i, 0:49] = W_444[k1 * 4 + i, 0:49]
    for k2 in range(3):
        for i in range(4):
            W[(4 + k2) * 4 + i, 49:93] = W_443[k2 * 4 + i, 49:93]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
