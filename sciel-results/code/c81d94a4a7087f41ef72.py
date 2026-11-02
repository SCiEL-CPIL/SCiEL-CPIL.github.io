import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 steps for 2x2 block matrices
    # A = [[A11, A12], [A21, A22]] where each block is 1x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x3
    # C = [[C11, C12], [C21, C22]] where each block is 1x3
    #
    # C11 = M1 - M2 + M4 - M7
    # C12 = M3 + M2
    # C21 = M1 + M5
    # C22 = M3 - M4 + M6 + M7
    #
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Each M is a 1x2 times 2x3 = 1x3 product
    # The 1x2 x 2x3 x 1x3 tensor has rank 6
    # Decomposition: for indices (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1,2}
    # T[i,j,k] = 1 if i*j*k is valid, else 0
    # Rank-1 tensors: (u_i, v_j, w_k) for each (i,j,k) pair
    # Actually, the 1x2 x 2x3 tensor decomposes as sum over i in {0,1}, j in {0,1}, k in {0,1,2}
    # of e_i ⊗ e_j ⊗ e_k, which is 6 terms total (2*2*3/... wait)
    
    # The 1x2 x 2x3 matrix multiplication: C = X * Y where X is 1x2, Y is 2x3
    # C_k = sum_j X_j * Y_jk for k=0,1,2
    # T[j,k] (for the inner product) but we have X_j and Y_jk
    # The tensor is: T_{x,y} where x indexes X (size 2), y indexes Y (size 6, since 2x3=6)
    # Actually let me think more carefully.
    
    # X is 1x2: entries X_0, X_1
    # Y is 2x3: entries Y_{j,k} for j=0,1 and k=0,1,2
    # C is 1x3: C_k = sum_j X_j * Y_{j,k}
    # 
    # The tensor T has indices: i for X (i=0,1), j for Y's row (j=0,1), k for Y's col/C (k=0,1,2)
    # T[i,j,k] = 1 if this combination contributes, i.e., always 1 for the structure
    # C_k = sum_{i,j} X_i * Y_{j,k} * delta_{i,j}... no
    # C_k = sum_j X_j * Y_{j,k}
    # So T[i,j,k] = 1 if i == j, else 0? No, that's not right either.
    # 
    # Actually: C_k = sum_j X_j * Y_{j,k}
    # The bilinear form: for each k, C_k = sum_j (X_j)(Y_{j,k})
    # The tensor T_{i,j,k} where i indexes X, j indexes Y's first index, k indexes Y's second index
    # T[i,j,k] = 1 if i == j, else 0
    # This tensor has rank 2 (for each k, it's the identity 2x2 matrix)
    # Wait no, that gives rank 2*3 = 6 if we consider all k independently
    # 
    # Actually the full tensor T is in R^2 ⊗ R^6 ⊗ R^3 (X is 2-dim, Y is 6-dim flattened, C is 3-dim)
    # But structurally, for each k, the slice T[:,:,k] is the 2x2 identity (mapping X_j to Y_{j,k})
    # So T = sum_{j=0,1} sum_{k=0,1,2} e_j ⊗ (e_j ⊗ e_k) ⊗ e_k
    # That's 6 rank-1 terms!
    
    # So for each Strassen step, we need 6 rank-1 tensors for the 1x2 * 2x3 * 1x3 multiplication
    # Total: 7 * 6 = 42
    
    # Now let's build the decomposition
    # For each step s in {1,...,7} and each (j,k) in {0,1}x{0,1,2}:
    # U column: has 1 at position corresponding to X_j in step s
    # V column: has 1 at position corresponding to Y_{j,k} in step s  
    # W column: has 1 at position corresponding to C_k in step s, with sign from Strassen formula
    
    # Define the 7 Strassen steps with their X (1x2), Y (2x3) linear combinations
    # X is a linear combo of A entries, Y is a linear combo of B entries
    
    # A indices: A[i,j] for i=0,1 and j=0,1,2,3
    # A11 = [A[0,0], A[0,1]], A12 = [A[0,2], A[0,3]]
    # A21 = [A[1,0], A[1,1]], A22 = [A[1,2], A[1,3]]
    
    # B indices: B[i,j] for i=0,1,2,3 and j=0,1,2,3,4,5
    # B11 = B[0:2, 0:3], B12 = B[0:2, 3:6]
    # B21 = B[2:4, 0:3], B22 = B[2:4, 3:6]
    
    # Each step: X = sum of A entries with coeffs, Y = sum of B entries with coeffs
    # X has 2 components (the 1x2 vector), Y has 6 components (the 2x3 matrix flattened)
    
    # Step definitions: (X_coeffs, Y_coeffs)
    # X_coeffs: list of 2 lists, each list is [(A_idx, coeff), ...]
    # Y_coeffs: list of 6 lists, each list is [(B_idx, coeff), ...]
    
    steps = []
    
    # M1 = (A11 + A22)(B11 + B22)
    # X = [A[0,0]+A[1,2], A[0,1]+A[1,3]]
    # Y = B11 + B22 = B[0:2,0:3] + B[2:4,3:6]
    X1 = [
        [(0,0,1), (1,2,1)],  # X[0] = A[0,0] + A[1,2]
        [(0,1,1), (1,3,1)],  # X[1] = A[0,1] + A[1,3]
    ]
    Y1 = []
    for j in range(2):
        for k in range(3):
            Y1.append([(j, k, 1), (j+2, k+3, 1)])
    steps.append((X1, Y1))
    
    # M2 = (A21 + A22)B11
    X2 = [
        [(1,0,1), (1,2,1)],  # X[0] = A[1,0] + A[1,2]
        [(1,1,1), (1,3,1)],  # X[1] = A[1,1] + A[1,3]
    ]
    Y2 = []
    for j in range(2):
        for k in range(3):
            Y2.append([(j, k, 1)])
    steps.append((X2, Y2))
    
    # M3 = A11(B12 - B22)
    X3 = [
        [(0,0,1)],  # X[0] = A[0,0]
        [(0,1,1)],  # X[1] = A[0,1]
    ]
    Y3 = []
    for j in range(2):
        for k in range(3):
            Y3.append([(j, k+3, 1), (j+2, k+3, -1)])
    steps.append((X3, Y3))
    
    # M4 = A22(B21 - B11)
    X4 = [
        [(1,2,1)],  # X[0] = A[1,2]
        [(1,3,1)],  # X[1] = A[1,3]
    ]
    Y4 = []
    for j in range(2):
        for k in range(3):
            Y4.append([(j+2, k, 1), (j, k, -1)])
    steps.append((X4, Y4))
    
    # M5 = (A11 + A12)B22
    X5 = [
        [(0,0,1), (0,2,1)],  # X[0] = A[0,0] + A[0,2]
        [(0,1,1), (0,3,1)],  # X[1] = A[0,1] + A[0,3]
    ]
    Y5 = []
    for j in range(2):
        for k in range(3):
            Y5.append([(j+2, k+3, 1)])
    steps.append((X5, Y5))
    
    # M6 = (A21 - A11)(B11 + B12)
    X6 = [
        [(1,0,1), (0,0,-1)],  # X[0] = A[1,0] - A[0,0]
        [(1,1,1), (0,1,-1)],  # X[1] = A[1,1] - A[0,1]
    ]
    Y6 = []
    for j in range(2):
        for k in range(3):
            Y6.append([(j, k, 1), (j, k+3, 1)])
    steps.append((X6, Y6))
    
    # M7 = (A12 - A22)(B21 + B22)
    X7 = [
        [(0,2,1), (1,2,-1)],  # X[0] = A[0,2] - A[1,2]
        [(0,3,1), (1,3,-1)],  # X[1] = A[0,3] - A[1,3]
    ]
    Y7 = []
    for j in range(2):
        for k in range(3):
            Y7.append([(j+2, k, 1), (j+2, k+3, 1)])
    steps.append((X7, Y7))
    
    # Now assign each M to C blocks with signs:
    # C11 = M1 - M2 + M4 - M7
    # C12 = M3 + M2
    # C21 = M1 + M5
    # C22 = M3 - M4 + M6 + M7
    
    # C block assignments: for each step s and (j,k), which C block and position
    # C11: i=0, k=0,1,2 -> W index = k*n + 0 = k*2 + 0 = 2k
    # C12: i=0, k=3,4,5 -> W index = k*n + 0 = 2k
    # C21: i=1, k=0,1,2 -> W index = k*n + 1 = 2k+1
    # C22: i=1, k=3,4,5 -> W index = k*n + 1 = 2k+1
    
    # For step s, the k-th element of C (k=0,1,2) maps to:
    # M1: C11[k] with coeff +1, C21[k] with coeff +1
    # M2: C11[k] with coeff -1, C12[k] with coeff +1
    # M3: C12[k] with coeff +1, C22[k] with coeff +1
    # M4: C11[k] with coeff +1, C22[k] with coeff -1
    # M5: C21[k] with coeff +1
    # M6: C22[k] with coeff +1
    # M7: C11[k] with coeff -1, C22[k] with coeff +1
    
    # W assignments: list of (block_idx, k_offset, coeff) for each step
    # block_idx: 0=C11, 1=C12, 2=C21, 3=C22
    # k_offset: 0,1,2 within the block
    w_assignments = [
        [(0, 0, 1), (2, 0, 1)],    # M1: C11 + C21
        [(0, 0, -1), (1, 0, 1)],   # M2: -C11 + C12
        [(1, 0, 1), (3, 0, 1)],    # M3: C12 + C22
        [(0, 0, 1), (3, 0, -1)],   # M4: C11 - C22
        [(2, 0, 1)],               # M5: C21
        [(3, 0, 1)],               # M6: C22
        [(0, 0, -1), (3, 0, 1)],   # M7: -C11 + C22
    ]
    
    col = 0
    for s in range(7):
        X_coeffs = steps[s][0]
        Y_coeffs = steps[s][1]
        w_assign = w_assignments[s]
        
        for j in range(2):
            for k in range(3):
                # U: X_j component
                for (a_i, a_j, coeff) in X_coeffs[j]:
                    U[a_i * m + a_j, col] = coeff
                
                # V: Y_{j,k} component
                for (b_i, b_j, coeff) in Y_coeffs[j * 3 + k]:
                    V[b_i * p + b_j, col] = coeff
                
                # W: C block assignments
                for (block_idx, k_off, coeff) in w_assign:
                    # block_idx 0=C11 (i=0, k=0-2), 1=C12 (i=0, k=3-5)
                    # 2=C21 (i=1, k=0-2), 3=C22 (i=1, k=3-5)
                    if block_idx == 0:
                        c_i, c_k = 0, k_off
                    elif block_idx == 1:
                        c_i, c_k = 0, k_off + 3
                    elif block_idx == 2:
                        c_i, c_k = 1, k_off
                    else:
                        c_i, c_k = 1, k_off + 3
                    W[c_k * n + c_i, col] = coeff
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = 2, 4, 6
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for l in range(U.shape[1]):
                for j in range(m):
                    C_computed[i, k] += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    error = np.max(np.abs(C_computed - C_expected))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")

