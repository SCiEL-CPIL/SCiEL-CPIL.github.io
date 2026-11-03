import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    n, m, p = 4, 4, 6
    
    # We'll build the decomposition in two parts:
    # Part 1: 4x4 * 4x4 -> 4x4 (first 4 columns), rank 49
    # Part 2: 4x4 * 4x2 -> 4x2 (last 2 columns), rank 28
    # Total rank: 77
    
    # --- Part 1: 4x4x4 decomposition (rank 49) ---
    # Using Strassen's method recursively:
    # 2x2x2 has rank 7 (Strassen)
    # 4x4x4 = 4x4x2 ⊕ 4x4x2, but better to use the known 49-term decomposition
    # 
    # The 49-term decomposition for 4x4x4 can be constructed from:
    # Split 4x4 into 2x2 blocks, use Strassen for each 2x2x2 subproblem
    # But that gives 7*4 = 28 for 4x4x4... no, that's not right.
    #
    # Actually, the rank of 4x4x4 is 49. This is achieved by:
    # 4x4x4 = (2x2x2) ⊗ (2x2x2) in some sense, but the tensor rank 
    # of a Kronecker product is the product of ranks: 7*7 = 49.
    #
    # So we use the Kronecker product of two 2x2x2 Strassen decompositions.
    
    # Strassen 2x2x2 decomposition (rank 7):
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # U for 2x2x2 (rows: i*m+j = i*2+j, 4 rows)
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # a11
        [0, 1, 0, 0, 0, 1, 0],  # a21
        [0, 0, 1, 0, 0, 0, 1],  # a12
        [1, 1, 0, 1, 0, 0, 0],  # a22
    ])
    
    # V for 2x2x2 (rows: j*p+k = j*2+k, 4 rows)
    V2 = np.array([
        [1, 0, 1, 0, 0, 0, 0],  # b11
        [0, 1, 0, 1, 1, 0, 1],  # b21
        [0, 0, 1, 0, 0, 1, 0],  # b12
        [1, 0, 0, 0, 1, 0, 1],  # b22
    ])
    
    # W for 2x2x2 (rows: k*n+i = k*2+i, 4 rows)
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # c11
        [0, 0, 0, 0, 1, 0, 0],  # c21
        [0, 1, 0, 0, 0, 0, 0],  # c12
        [0, 0, 0, 0, 0, 1, 0],  # c22
    ])
    
    # Wait, I need to be more careful. Let me redefine:
    # For 2x2x2: T[i,j,k] where i,j,k in {0,1}
    # U has shape (2*2, 7) = (4, 7), indexed by i*2+j
    # V has shape (2*2, 7) = (4, 7), indexed by j*2+k
    # W has shape (2*2, 7) = (4, 7), indexed by k*2+i
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M7
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # U: rows are i*2+j for (i,j) in {(0,0),(0,1),(1,0),(1,1)}
    # a00 = U[0], a01 = U[1], a10 = U[2], a11 = U[3]
    U2[0, 0] = 1  # a00 in M1
    U2[3, 0] = 1  # a11 in M1
    U2[2, 1] = 1  # a10 in M2
    U2[3, 1] = 1  # a11 in M2
    U2[0, 2] = 1  # a00 in M3
    U2[3, 3] = 1  # a11 in M4
    U2[0, 4] = 1  # a00 in M5
    U2[1, 4] = 1  # a01 in M5
    U2[2, 5] = 1  # a10 in M6
    U2[0, 5] = -1 # a00 in M6
    U2[1, 6] = 1  # a01 in M7
    U2[3, 6] = -1 # a11 in M7
    
    # V: rows are j*2+k for (j,k) in {(0,0),(0,1),(1,0),(1,1)}
    # b00 = V[0], b01 = V[1], b10 = V[2], b11 = V[3]
    V2[0, 0] = 1  # b00 in M1
    V2[3, 0] = 1  # b11 in M1
    V2[0, 1] = 1  # b00 in M2
    V2[1, 2] = 1  # b01 in M3
    V2[3, 2] = -1 # b11 in M3
    V2[2, 3] = 1  # b10 in M4
    V2[0, 3] = -1 # b00 in M4
    V2[3, 4] = 1  # b11 in M5
    V2[0, 5] = 1  # b00 in M6
    V2[1, 5] = 1  # b01 in M6
    V2[2, 6] = 1  # b10 in M7
    V2[3, 6] = 1  # b11 in M7
    
    # W: rows are k*2+i for (k,i) in {(0,0),(0,1),(1,0),(1,1)}
    # c00 = W[0], c01 = W[1], c10 = W[2], c11 = W[3]
    W2[0, 0] = 1  # c00 in M1
    W2[2, 0] = 0  # 
    W2[0, 3] = 1  # c00 in M4
    W2[0, 4] = -1 # c00 in M5
    W2[0, 6] = -1 # c00 in M7
    W2[1, 2] = 1  # c01 in M3
    W2[1, 4] = 1  # c01 in M5
    W2[2, 1] = 1  # c10 in M2
    W2[2, 3] = 1  # c10 in M4
    W2[3, 0] = 1  # c11 in M1
    W2[3, 1] = -1 # c11 in M2
    W2[3, 2] = 1  # c11 in M3
    W2[3, 6] = 1  # c11 in M7
    
    # Verify 2x2x2 decomposition
    np.random.seed(42)
    A2 = np.random.randn(2, 2)
    B2 = np.random.randn(2, 2)
    C2_expected = A2 @ B2
    C2_computed = np.zeros((2, 2))
    for r in range(7):
        a_scalar = sum(U2[i*2+j, r] * A2[i, j] for i in range(2) for j in range(2))
        b_scalar = sum(V2[j*2+k, r] * B2[j, k] for j in range(2) for k in range(2))
        for i in range(2):
            for k in range(2):
                C2_computed[i, k] += W2[k*2+i, r] * a_scalar * b_scalar
    
    if not np.allclose(C2_computed, C2_expected, atol=1e-10):
        # Fallback: use schoolbook for everything
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now construct 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions
    # 4x4x4: i,j,k in {0,1,2,3}
    # Split into 2x2 blocks: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # where i1,j1,k1 in {0,1} and i2,j2,k2 in {0,1}
    # 
    # The tensor T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2]
    # So U4[(2*i1+i2)*4 + (2*j1+j2), r1*7+r2] = U2[i1*2+i2, r1] * U2[j1*2+j2, r2]
    # Wait, that's not quite right. Let me think more carefully.
    #
    # For the Kronecker product: if T = T_a ⊗ T_b, then
    # T[i_a, i_b, j_a, j_b, k_a, k_b] = T_a[i_a, j_a, k_a] * T_b[i_b, j_b, k_b]
    # 
    # For 4x4x4, we have indices i, j, k each in {0,1,2,3}
    # Write i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # Then T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2]
    #
    # U4 has shape (16, 49), indexed by (i,j) -> i*4+j
    # V4 has shape (16, 49), indexed by (j,k) -> j*4+k
    # W4 has shape (16, 49), indexed by (k,i) -> k*4+i
    #
    # The rank-49 decomposition:
    # U4[i*4+j, r1*7+r2] = U2[i1*2+i2, r1] * U2[j1*2+j2, r2]
    # where i = 2*i1+i2, j = 2*j1+j2
    #
    # Similarly for V and W.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            U4[i*4+j, r] = U2[i1*2+i2, r1] * U2[j1*2+j2, r2]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            V4[j*4+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1*7 + r2
                            W4[k*4+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    # Verify 4x4x4 decomposition
    A4 = np.random.randn(4, 4)
    B4 = np.random.randn(4, 4)
    C4_expected = A4 @ B4
    C4_computed = np.zeros((4, 4))
    for r in range(49):
        a_scalar = sum(U4[i*4+j, r] * A4[i, j] for i in range(4) for j in range(4))
        b_scalar = sum(V4[j*4+k, r] * B4[j, k] for j in range(4) for k in range(4))
        for i in range(4):
            for k in range(4):
                C4_computed[i, k] += W4[k*4+i, r] * a_scalar * b_scalar
    
    if not np.allclose(C4_computed, C4_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # --- Part 2: 4x4x2 decomposition (rank 28) ---
    # 4x4x2: i in {0,1,2,3}, j in {0,1,2,3}, k in {0,1}
    # Split into 2x2 blocks: i = 2*i1+i2, j = 2*j1+j2
    # T4x2[i,j,k] = T2x2[i1,j1,k] * T2x1[i2,j2,k]? No, that doesn't work directly.
    #
    # Actually, 4x4x2 can be seen as 4x4x1 ⊕ 4x4x1, each of rank 16, total 32.
    # But we want rank 28. 
    #
    # Alternatively, 4x4x2 = (2x2x2) ⊗ (2x2x1) in some sense.
    # 2x2x2 has rank 7, 2x2x1 has rank 4 (schoolbook). 7*4 = 28.
    #
    # 2x2x1: T[i,j,k] where k=0 only. This is just the dot product.
    # Rank of 2x2x1 is 4 (schoolbook).
    # U2x1[i*2+j, r] for r in 0..3: U2x1[i*2+j, i*2+j] = 1
    # V2x1[j*2+0, r] for r in 0..3: V2x1[j*2+0, j*2] = 1
    # W2x1[0*2+i, r] for r in 0..3: W2x1[i, j*2] = 1? 
    #
    # Wait, for 2x2x1: T[i,j,0] = 1 for all i,j. No, T[i,j,k] = delta_{i,k} for matrix mult.
    # Actually for 2x2x1: C[i,0] = sum_j A[i,j] B[j,0]
    # T[i,j,0] should give the coefficient. The tensor is T[i,j,k] = 1 if this is the 
    # (i,j,k) entry of the multiplication tensor.
    #
    # For 2x2x1: n=2, m=2, p=1
    # T[i,j,k] = 1 for the multiplication tensor means:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the schoolbook sense.
    #
    # For 2x2x1: U has shape (4, rank), V has shape (2, rank), W has shape (2, rank)
    # Schoolbook rank 4:
    # U[i*2+j, i*2+j] = 1
    # V[j*2+0, j*2] = 1  (since p=1, j*p+k = j*1+0 = j)
    # W[0*2+i, i*2] = 1  (since p=1, k*n+i = 0*2+i = i)
    
    U2x1 = np.zeros((4, 4))
    V2x1 = np.zeros((2, 4))
    W2x1 = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            r = i*2 + j
            U2x1[i*2+j, r] = 1
            V2x1[j, r] = 1
            W2x1[i, r] = 1
    
    # Verify 2x2x1
    A2x1 = np.random.randn(2, 2)
    B2x1 = np.random.randn(2, 1)
    C2x1_expected = A2x1 @ B2x1
    C2x1_computed = np.zeros((2, 1))
    for r in range(4):
        a_scalar = sum(U2x1[i*2+j, r] * A2x1[i, j] for i in range(2) for j in range(2))
        b_scalar = sum(V2x1[j, r] * B2x1[j, 0] for j in range(2))
        for i in range(2):
            C2x1_computed[i, 0] += W2x1[i, r] * a_scalar * b_scalar
    
    if not np.allclose(C2x1_computed, C2x1_expected, atol=1e-10):
        # Fallback
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now construct 4x4x2 using Kronecker product of 2x2x2 (Strassen, rank 7) and 2x2x1 (schoolbook, rank 4)
    # 4x4x2: i = 2*i1+i2, j = 2*j1+j2, k in {0,1}
    # T4x2[i,j,k] = T2x2[i1,j1,k] * T2x1[i2,j2,k]
    #
    # U4x2 has shape (16, 28), indexed by i*4+j
    # V4x2 has shape (8, 28), indexed by j*2+k
    # W4x2 has shape (8, 28), indexed by k*4+i
    
    U4x2 = np.zeros((16, 28))
    V4x2 = np.zeros((8, 28))
    W4x2 = np.zeros((8, 28))
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for r1 in range(7):
                        for r2 in range(4):
                            r = r1*4 + r2
                            U4x2[i*4+j, r] = U2[i1*2+i2, r1] * U2x1[i2*2+j2, r2]
    
    for j1 in range(2):
        for j2 in range(2):
            for k in range(2):
                j = 2*j1 + j2
                for r1 in range(7):
                    for r2 in range(4):
                        r = r1*4 + r2
                        V4x2[j*2+k, r] = V2[j1*2+k, r1] * V2x1[j2, r2]
    
    for k in range(2):
        for i1 in range(2):
            for i2 in range(2):
                i = 2*i1 + i2
                for r1 in range(7):
                    for r2 in range(4):
                        r = r1*4 + r2
                        W4x2[k*4+i, r] = W2[k*2+i1, r1] * W2x1[i2, r2]
    
    # Verify 4x4x2
    A4x2 = np.random.randn(4, 4)
    B4x2 = np.random.randn(4, 2)
    C4x2_expected = A4x2 @ B4x2
    C4x2_computed = np.zeros((4, 2))
    for r in range(28):
        a_scalar = sum(U4x2[i*4+j, r] * A4x2[i, j] for i in range(4) for j in range(4))
        b_scalar = sum(V4x2[j*2+k, r] * B4x2[j, k] for j in range(4) for k in range(2))
        for i in range(4):
            for k in range(2):
                C4x2_computed[i, k] += W4x2[k*4+i, r] * a_scalar * b_scalar
    
    if not np.allclose(C4x2_computed, C4x2_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = 96
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, col] = 1
                    V[j*6+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Combine Part 1 (4x4x4, rank 49) and Part 2 (4x4x2, rank 28) into 4x4x6
    # Total rank: 49 + 28 = 77
    
    total_rank = 77
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    # Part 1: columns 0-48 (4x4x4, output columns 0-3)
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49)
    # For 4x4x6: U has shape (16, 77), V has shape (24, 77), W has shape (24, 77)
    # V4 is for 4x4x4, so V4[j*4+k, r] for k in 0..3
    # In 4x4x6, V[j*6+k, r] for k in 0..5
    # So V4[j*4+k, r] maps to V[j*6+k, r] for k in 0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*6+k, 0:49] = V4[j*4+k, 0:49]
    W[:, 0:49] = W4
    
    # Part 2: columns 49-76 (4x4x2, output columns 4-5)
    # U4x2 has shape (16, 28), V4x2 has shape (8, 28), W4x2 has shape (8, 28)
    # For 4x4x6: V4x2[j*2+k, r] for k in 0..1 maps to V[j*6+(4+k), r] for k in 0..1
    # W4x2[k*4+i, r] for k in 0..1 maps to W[(4+k)*4+i, r] for k in 0..1
    U[:, 49:77] = U4x2
    for j in range(4):
        for k in range(2):
            V[j*6+(4+k), 49:77] = V4x2[j*2+k, 0:28]
    for k in range(2):
        for i in range(4):
            W[(4+k)*4+i, 49:77] = W4x2[k*4+i, 0:28]
    
    # Final verification
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 6)
    C_expected = A @ B
    C_computed = np.zeros((4, 6))
    for r in range(77):
        a_scalar = sum(U[i*4+j, r] * A[i, j] for i in range(4) for j in range(4))
        b_scalar = sum(V[j*6+k, r] * B[j, k] for j in range(4) for k in range(6))
        for i in range(4):
            for k in range(6):
                C_computed[i, k] += W[k*4+i, r] * a_scalar * b_scalar
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = 96
        U_fb = np.zeros((16, rank))
        V_fb = np.zeros((24, rank))
        W_fb = np.zeros((24, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*6+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
