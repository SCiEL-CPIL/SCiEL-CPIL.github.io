import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Step 1: Decomposition of 3x2x3 with rank 12
    # Split 3x2x3 into 2x2x2 (rank 7), 2x2x1 (rank 2), 1x2x2 (rank 2), 1x2x1 (rank 1)
    # U, V, W for 2x2x2 (Strassen)
    U_222 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ])
    # Strassen's U, V, W for 2x2x2
    # U: (2x2) x 7
    U_222 = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ])
    # This is not correct. Let's use the standard Strassen matrices.
    # U: A rows, V: B cols, W: C
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U matrix (4x7): U[i, k] is coefficient of A_i in M_k
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, -1], # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, 0]   # A22
    ])
    
    # V matrix (4x7): V[j, k] is coefficient of B_j in M_k
    # B11, B12, B21, B22
    V_222 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B11
        [1, 0, 1, 0, 0, 1, 0],  # B12
        [1, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ])
    
    # W matrix (4x7): W[i, k] is coefficient of M_k in C_i
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 - M3
    W_222 = np.array([
        [1, 0, 0, -1, 1, 0, -1], # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, -1, 0, 0, 1, 0]  # C22
    ])
    
    # Verify: C = sum_k W[:,k] * M_k where M_k = sum_i U[i,k] A_i * sum_j V[j,k] B_j
    # This is the standard Strassen decomposition.
    
    # Decomposition of 2x2x1 (rank 2)
    # A is 2x2, B is 2x1. C is 2x1.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    U_221 = np.array([
        [1, 0],
        [0, 1],
        [0, 0], # Wait, A is 2x2, so 4 elements.
        [0, 0]
    ])
    # Correct U_221:
    # A11, A12, A21, A22
    U_221 = np.array([
        [1, 0], # A11
        [0, 1], # A12
        [0, 0], # A21
        [0, 0]  # A22
    ])
    # V_221: B is 2x1. B1, B2.
    V_221 = np.array([
        [1, 0], # B1
        [0, 1]  # B2
    ])
    # W_221: C is 2x1. C1, C2.
    W_221 = np.array([
        [1, 0], # C1
        [0, 1]  # C2
    ])
    # Check: C1 = 1*M1 + 0*M2 = A11 B1 + A12 B2. Correct.
    # C2 = 0*M1 + 1*M2 = A21 B1 + A22 B2. Wait, U_221 has A21, A22 as 0.
    # So M2 = A12 B2. C2 = A12 B2. Incorrect.
    
    # Correct 2x2x1 decomposition:
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # This requires 4 multiplications if done naively?
    # No, 2x2x1 rank is 2?
    # mu(2, 2, 1) = 2.
    # M1 = (A11 + A21) B1? No.
    # M1 = A11 B1 + A12 B2
    # M2 = A21 B1 + A22 B2
    # This is 2 multiplications of 2x1 vectors? No, scalar multiplications.
    # A11, A12, A21, A22 are scalars. B1, B2 are scalars.
    # So 4 scalar multiplications. Rank 4?
    # No, mu(2, 2, 1) = 2 is wrong. mu(2, 2, 1) = 4?
    # Actually, mu(n, m, 1) = n*m.
    # So mu(2, 2, 1) = 4.
    # My previous calculation was wrong.
    
    # Let's re-evaluate mu(3, 2, 3).
    # 2x2x2: rank 7.
    # 2x2x1: rank 4.
    # 1x2x2: rank 2.
    # 1x2x1: rank 2.
    # Total rank 7 + 4 + 2 + 2 = 15.
    # So mu(3, 2, 3) <= 15.
    # Then mu(3, 4, 6) <= 2 * 15 = 30.
    
    # Can we do better?
    # mu(2, 2, 1) = 4.
    # mu(1, 2, 2) = 2.
    # mu(1, 2, 1) = 2.
    # Total 15.
    
    # Is there a better decomposition for 3x2x3?
    # Maybe mu(3, 2, 3) = 14?
    # If so, rank 28.
    
    # Let's use rank 15 for 3x2x3.
    # Total rank 30.
    # Schoolbook is 72.
    # 30 is a significant improvement.

    # Construct 3x2x3 decomposition with rank 15.
    # Part 1: 2x2x2 (rank 7)
    # Part 2: 2x2x1 (rank 4)
    # Part 3: 1x2x2 (rank 2)
    # Part 4: 1x2x1 (rank 2)
    
    # U_323, V_323, W_323
    # U_323 shape: (3*2, 15) = (6, 15)
    # V_323 shape: (2*3, 15) = (6, 15)
    # W_323 shape: (3*3, 15) = (9, 15)
    
    U_323 = np.zeros((6, 15))
    V_323 = np.zeros((6, 15))
    W_323 = np.zeros((9, 15))
    
    # Part 1: 2x2x2 (indices 0-1, 0-1, 0-1)
    # U_222 is 4x7. Embed in U_323.
    # A indices: 0,1,2 (rows), 0,1 (cols).
    # U_323[i*m+j, k]
    # m=2.
    # A11 -> 0*2+0=0. A12 -> 0*2+1=1. A21 -> 1*2+0=2. A22 -> 1*2+1=3.
    U_323[0:4, 0:7] = U_222
    # B indices: 0,1 (rows), 0,1,2 (cols).
    # B11 -> 0*3+0=0. B12 -> 0*3+1=1. B21 -> 1*3+0=3. B22 -> 1*3+1=4.
    V_323[0, 0:7] = V_222[0]
    V_323[1, 0:7] = V_222[1]
    V_323[3, 0:7] = V_222[2]
    V_323[4, 0:7] = V_222[3]
    # C indices: 0,1,2 (rows), 0,1 (cols).
    # C11 -> 0*3+0=0. C12 -> 0*3+1=1. C21 -> 1*3+0=3. C22 -> 1*3+1=4.
    W_323[0, 0:7] = W_222[0]
    W_323[1, 0:7] = W_222[1]
    W_323[3, 0:7] = W_222[2]
    W_323[4, 0:7] = W_222[3]
    
    # Part 2: 2x2x1 (indices 0-1, 0-1, 1)
    # A is 2x2. B is 2x1. C is 2x1.
    # U_221 (4x4), V_221 (2x4), W_221 (2x4)
    U_221 = np.eye(4)
    V_221 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    W_221 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    # Wait, B is 2x1. B1, B2.
    # M1 = A11 B1. M2 = A12 B2. M3 = A21 B1. M4 = A22 B2.
    # C1 = M1 + M2. C2 = M3 + M4.
    U_221 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_221 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    W_221 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # Embed in U_323 (cols 7-10)
    U_323[0:4, 7:11] = U_221
    # B indices for col 1: B1 -> 0*3+1=1. B2 -> 1*3+1=4.
    V_323[1, 7:11] = V_221[0]
    V_323[4, 7:11] = V_221[1]
    # C indices for col 1: C1 -> 0*3+1=1. C2 -> 1*3+1=4.
    W_323[1, 7:11] = W_221[0]
    W_323[4, 7:11] = W_221[1]
    
    # Part 3: 1x2x2 (indices 2, 0-1, 0-1)
    # A is 1x2. B is 2x2. C is 1x2.
    # U_122 (2x2), V_122 (4x2), W_122 (2x2)
    # M1 = A1 B11. M2 = A2 B12.
    # C1 = M1. C2 = M2.
    U_122 = np.array([
        [1, 0],
        [0, 1]
    ])
    V_122 = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ])
    W_122 = np.array([
        [1, 0],
        [0, 1]
    ])
    
    # Embed in U_323 (cols 11-12)
    # A indices: row 2, col 0,1.
    # A21 -> 2*2+0=4. A22 -> 2*2+1=5.
    U_323[4, 11:13] = U_122[0]
    U_323[5, 11:13] = U_122[1]
    # B indices: 0,1, 0,1.
    # B11 -> 0. B12 -> 1. B21 -> 3. B22 -> 4.
    V_323[0, 11:13] = V_122[0]
    V_323[1, 11:13] = V_122[1]
    # C indices: row 2, col 0,1.
    # C21 -> 2*3+0=6. C22 -> 2*3+1=7.
    W_323[6, 11:13] = W_122[0]
    W_323[7, 11:13] = W_122[1]
    
    # Part 4: 1x2x1 (indices 2, 0-1, 1)
    # A is 1x2. B is 2x1. C is 1x1.
    # U_121 (2x2), V_121 (2x2), W_121 (1x2)
    # M1 = A1 B1. M2 = A2 B2.
    # C1 = M1 + M2.
    U_121 = np.array([
        [1, 0],
        [0, 1]
    ])
    V_121 = np.array([
        [1, 0],
        [0, 1]
    ])
    W_121 = np.array([
        [1, 1]
    ])
    
    # Embed in U_323 (cols 13-14)
    U_323[4, 13:15] = U_121[0]
    U_323[5, 13:15] = U_121[1]
    V_323[1, 13:15] = V_121[0]
    V_323[4, 13:15] = V_121[1]
    W_323[7, 13:15] = W_121[0]
    
    # Now we have U_323, V_323, W_323 of rank 15.
    
    # Step 2: Kronecker product with 1x2x2 (rank 2)
    # U_122, V_122, W_122
    U_122_k = np.array([
        [1, 0],
        [0, 1]
    ])
    V_122_k = np.array([
        [1, 0],
        [0, 1]
    ])
    W_122_k = np.array([
        [1, 0],
        [0, 1]
    ])
    
    # U = U_323 @ U_122_k (Kronecker)
    # U_323 is 6x15. U_122_k is 2x2.
    # U is (6*2) x (15*2) = 12x30.
    U = np.kron(U_323, U_122_k)
    V = np.kron(V_323, V_122_k)
    W = np.kron(W_323, W_122_k)
    
    return {"U": U, "V": V, "W": W}
