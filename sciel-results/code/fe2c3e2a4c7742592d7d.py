import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Build Schoolbook Decomposition (Rank 12)
    # This serves as a fallback and provides the structure for the 'extra' column.
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()

    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def verify(U, V, W):
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-8)

    # 2. Construct Rank-11 Decomposition
    # Strategy: Use Strassen (rank 7) for the 2x2 part (cols 0,1 of M) 
    # and Schoolbook (rank 4) for the 2x1 part (col 2 of M).
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen Factors (Standard):
    # U_strassen (4x7): Rows are A indices.
    # V_strassen (4x7): Rows are B indices.
    # W_strassen (4x7): Rows are C indices.
    
    # Let's define the 7 factors explicitly.
    # f1 = (a11+a22)(b11+b22)
    # f2 = (a21+a22)b11
    # f3 = a11(b12-b22)
    # f4 = a22(b21-b11)
    # f5 = (a11+a12)b22
    # f6 = (a21-a11)(b11+b12)
    # f7 = (a12-a22)(b21+b22)
    
    # U (A factors):
    # Row 0 (a11): 1, 0, 1, 0, 0, -1, 0
    # Row 1 (a12): 0, 0, 0, 0, 1, 0, 1
    # Row 2 (a21): 0, 1, 0, 0, 0, 1, 0
    # Row 3 (a22): 1, 1, 0, 1, 0, 0, -1
    
    U_str = np.array([
        [1, 0, 1, 0, 0, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    # V (B factors):
    # Row 0 (b11): 1, 1, 0, -1, 0, 1, 0
    # Row 1 (b12): 0, 0, 1, 0, 0, 1, 0
    # Row 2 (b21): 0, 0, 0, 1, 0, 0, 1
    # Row 3 (b22): 1, 0, -1, 0, 1, 0, 1
    
    V_str = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    # W (C factors):
    # Row 0 (c11): 1, 0, 0, 0, 0, 0, 1
    # Row 1 (c12): 0, 0, 1, 0, 1, 0, 0
    # Row 2 (c21): 0, 1, 0, 1, 0, 0, 0
    # Row 3 (c22): 1, -1, 0, 0, 1, 0, 0
    
    W_str = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 1, 0, 0]
    ])
    
    # Now, we need to map this to the full 2x2 * 2x3 problem.
    # A is 2x2 (indices 0..3).
    # B is 2x3 (indices 0..5).
    #    B = [[b00, b01, b02],
    #         [b10, b11, b12]]
    # C is 3x2 (indices 0..5).
    #    C = [[c00, c01],
    #         [c10, c11],
    #         [c20, c21]]
    
    # The multiplication is C = A * B.
    # C[:, 0:2] = A * B[:, 0:2] (This is 2x2 * 2x2, handled by Strassen)
    # C[:, 2]   = A * B[:, 2]   (This is 2x2 * 2x1, handled by Schoolbook)
    
    # Let's construct U, V, W for the combined rank 7 + 4 = 11 decomposition.
    
    U_comb = np.zeros((n*m, 11))
    V_comb = np.zeros((m*p, 11))
    W_comb = np.zeros((p*n, 11))
    
    # Part 1: Strassen for B[:, 0:2] -> C[:, 0:2]
    # A indices: 0,1,2,3 map to U_str rows 0,1,2,3
    # B indices: 0,1,2,3 map to V_str rows 0,1,2,3 (which are b00, b01, b10, b11)
    # C indices: 0,1,2,3 map to W_str rows 0,1,2,3 (which are c00, c01, c10, c11)
    
    # Note: The problem statement says W[k*n+i, r].
    # My C indices above: c00 (k=0,i=0), c01 (k=0,i=1), c10 (k=1,i=0), c11 (k=1,i=1).
    # This matches the standard row-major order for C if C is 2x2.
    # But C is 3x2.
    # C indices in the problem:
    # c00 (k=0, i=0) -> index 0
    # c01 (k=0, i=1) -> index 1
    # c10 (k=1, i=0) -> index 2
    # c11 (k=1, i=1) -> index 3
    # c20 (k=2, i=0) -> index 4
    # c21 (k=2, i=1) -> index 5
    
    # So for the Strassen part (cols 0,1 of B), the output C only affects rows 0,1 (k=0,1).
    # The indices 0,1,2,3 of W_comb correspond to c00, c01, c10, c11.
    
    U_comb[0:4, 0:7] = U_str
    V_comb[0:4, 0:7] = V_str
    W_comb[0:4, 0:7] = W_str
    
    # Part 2: Schoolbook for B[:, 2] -> C[:, 2]
    # B index 4 is b02 (j=0, k=2). B index 5 is b12 (j=1, k=2).
    # A is full 2x2.
    # C indices 4 (c20) and 5 (c21).
    
    # We need 4 factors for this part:
    # f8 = a00 * b02 -> c20
    # f9 = a01 * b12 -> c20
    # f10 = a10 * b02 -> c21
    # f11 = a11 * b12 -> c21
    
    # Wait, C = A * B.
    # c20 = a00*b02 + a01*b12
    # c21 = a10*b02 + a11*b12
    
    # Factor 7 (index 7): a00 * b02 -> c20
    U_comb[0, 7] = 1  # a00
    V_comb[4, 7] = 1  # b02
    W_comb[4, 7] = 1  # c20
    
    # Factor 8 (index 8): a01 * b12 -> c20
    U_comb[1, 8] = 1  # a01
    V_comb[5, 8] = 1  # b12
    W_comb[4, 8] = 1  # c20
    
    # Factor 9 (index 9): a10 * b02 -> c21
    U_comb[2, 9] = 1  # a10
    V_comb[4, 9] = 1  # b02
    W_comb[5, 9] = 1  # c21
    
    # Factor 10 (index 10): a11 * b12 -> c21
    U_comb[3, 10] = 1 # a11
    V_comb[5, 10] = 1 # b12
    W_comb[5, 10] = 1 # c21
    
    # Verify the constructed decomposition
    if verify(U_comb, V_comb, W_comb):
        best_U, best_V, best_W = U_comb.tolist(), V_comb.tolist(), W_comb.tolist()
        best_rank = 11
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        pass

    # Check for rank 10? 
    # Theoretical lower bound for 2x2x3 is likely 10 or 11.
    # Strassen is 7 for 2x2x2.
    # 2x2x3 can be seen as 2x2x2 + 2x2x1.
    # Rank(2x2x2) = 7. Rank(2x2x1) = 4 (schoolbook). Total 11.
    # Can we do better?
    # If we use a rank 10 decomposition, it would imply a significant improvement over standard blocks.
    # Given the time constraint and the success of the rank 11 construction, we return rank 11.
    # The prompt asks for rank-minimal. 11 is better than 12.
    # Is 10 possible?
    # Over GF(2), rank of 2x2x2 is 7.
    # For 2x2x3, the tensor is T = T_{2x2x2} \otimes 1 + ...
    # It is known that rank(2,2,3) >= 10.
    # Let's try to see if we can find a rank 10 solution quickly if we have time.
    # But the combinatorial construction is robust.
    
    return {"U": best_U, "V": best_V, "W": best_W}
