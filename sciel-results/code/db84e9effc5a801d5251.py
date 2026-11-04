"""Matrix multiplication 3x4x6 rank-minimal tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the tensor decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_computed = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c].reshape(n, m)
        v_c = V_arr[:, c].reshape(m, p)
        w_c = W_arr[:, c].reshape(p, n)
        T_computed += u_c[:, :, None] * v_c[:, None, :] * w_c[None, :, :]
    
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-10)

def _strassen_2x2():
    """
    Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) where:
    U[i*2+j, c] corresponds to A[i,j]
    V[j*2+k, c] corresponds to B[j,k]  
    W[k*2+i, c] corresponds to C[i,k] (trace convention: W[k*n+i] with n=2)
    """
    # Strassen's 7 products for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4, M1-M2+M3+M6]]
    
    # U matrix: rows indexed by (i,j) for A[i,j], cols by product index
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # Row mapping: i*2+j -> row index
    U = np.zeros((4, 7))
    # M1: (a+d)(e+h) -> U[0,0]=1 (a), U[3,0]=1 (d)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    # M2: (c+d)e -> U[2,1]=1 (c), U[3,1]=1 (d)
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    # M3: a(f-h) -> U[0,2]=1 (a)
    U[0, 2] = 1  # a
    # M4: d(g-e) -> U[3,3]=1 (d)
    U[3, 3] = 1  # d
    # M5: (a+b)h -> U[0,4]=1 (a), U[1,4]=1 (b)
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    # M6: (c-a)(e+f) -> U[2,5]=1 (c), U[0,5]=-1 (a)
    U[2, 5] = 1  # c
    U[0, 5] = -1  # a
    # M7: (b-d)(g+h) -> U[1,6]=1 (b), U[3,6]=-1 (d)
    U[1, 6] = 1  # b
    U[3, 6] = -1  # d
    
    # V matrix: rows indexed by (j,k) for B[j,k], cols by product index
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    V = np.zeros((4, 7))
    # M1: (a+d)(e+h) -> V[0,0]=1 (e), V[3,0]=1 (h)
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    # M2: (c+d)e -> V[0,1]=1 (e)
    V[0, 1] = 1  # e
    # M3: a(f-h) -> V[1,2]=1 (f), V[3,2]=-1 (h)
    V[1, 2] = 1  # f
    V[3, 2] = -1  # h
    # M4: d(g-e) -> V[2,3]=1 (g), V[0,3]=-1 (e)
    V[2, 3] = 1  # g
    V[0, 3] = -1  # e
    # M5: (a+b)h -> V[3,4]=1 (h)
    V[3, 4] = 1  # h
    # M6: (c-a)(e+f) -> V[0,5]=1 (e), V[1,5]=1 (f)
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    # M7: (b-d)(g+h) -> V[2,6]=1 (g), V[3,6]=1 (h)
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    
    # W matrix: rows indexed by (k,i) for C[i,k] (trace convention W[k*n+i])
    # C indices: C[0,0], C[0,1], C[1,0], C[1,1]
    # W[k*2+i]: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    W = np.zeros((4, 7))
    # C[0,0] = M1+M4-M5+M7
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # M5
    W[0, 6] = 1   # M7
    # C[0,1] = M3+M5
    W[1, 2] = 1   # M3
    W[1, 4] = 1   # M5
    # C[1,0] = M2+M4
    W[2, 1] = 1   # M2
    W[2, 3] = 1   # M4
    # C[1,1] = M1-M2+M3+M6
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6
    
    return U, V, W

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Strategy: Partition the 3x4x6 multiplication into blocks
    # A is 3x4, B is 4x6, C is 3x6
    # 
    # Split A into rows: A_top (2x4), A_bottom (1x4)
    # Split B into cols: B_left (4x3), B_right (4x3)
    # 
    # C = [[A_top @ B_left, A_top @ B_right],
    #      [A_bottom @ B_left, A_bottom @ B_right]]
    # 
    # For the 2x2x2 sub-blocks, we can use Strassen (rank 7 instead of 8)
    # But we need to be careful about the non-square blocks
    #
    # Alternative approach: Use a more systematic decomposition
    # 
    # Let's try to construct a decomposition with rank significantly less than 72
    # 
    # Key insight: We can use the fact that matrix multiplication tensor has
    # structure. For 3x4x6, let's try to find a good decomposition.
    #
    # Approach 1: Block decomposition with Strassen for 2x2 sub-blocks
    # A (3x4) = [A11 A12; A21 A22] where A11 is 2x2, etc.
    # But 4 columns doesn't split evenly into 2x2 blocks for all positions
    #
    # Let's try: Split A into 2x2 blocks:
    # A = [[a11 a12], [a21 a22]] where each ai is 2x2 or 1x2
    # Actually, 3 rows and 4 cols: 
    #   Row 0-1, Col 0-1: 2x2 block
    #   Row 0-1, Col 2-3: 2x2 block  
    #   Row 2, Col 0-1: 1x2 block
    #   Row 2, Col 2-3: 1x2 block
    #
    # Similarly for B (4x6):
    #   Row 0-1, Col 0-2: 2x3 block
    #   Row 0-1, Col 3-5: 2x3 block
    #   Row 2-3, Col 0-2: 2x3 block
    #   Row 2-3, Col 3-5: 2x3 block
    #
    # This gets complicated. Let me try a different approach.
    #
    # Approach 2: Use known lower bounds and construct explicitly
    # The rank of the 3x4x6 matrix multiplication tensor is at most 72 (schoolbook)
    # and at least max(12, 24, 18) = 24
    #
    # Let's try to construct a decomposition with rank around 36-48
    #
    # For now, let's use a hybrid: Strassen for 2x2 sub-blocks where possible
    # and schoolbook for the rest
    #
    # Actually, let's try a cleaner approach:
    # Decompose into independent 2x2x2 Strassen blocks + remaining terms
    #
    # Split A (3x4) as:
    #   A = [A_00 A_01; A_10 A_11] where A_00 is 2x2, A_01 is 2x2,
    #        A_10 is 1x2, A_11 is 1x2
    #
    # Split B (4x6) as:
    #   B = [B_00 B_01 B_02; B_10 B_11 B_12] where B_00 is 2x2, B_01 is 2x2,
    #        B_02 is 2x2, B_10 is 2x2, B_11 is 2x2, B_12 is 2x2
    #        Wait, 4 rows split into 2+2, 6 cols split into 2+2+2
    #
    # C (3x6) = A @ B
    # C[0:2, 0:2] = A_00 @ B_00 + A_01 @ B_10  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C[0:2, 2:4] = A_00 @ B_01 + A_01 @ B_11
    # C[0:2, 4:6] = A_00 @ B_02 + A_01 @ B_12
    # C[2:3, 0:2] = A_10 @ B_00 + A_11 @ B_10
    # C[2:3, 2:4] = A_10 @ B_01 + A_11 @ B_11
    # C[2:3, 4:6] = A_10 @ B_02 + A_11 @ B_12
    #
    # For the top-left 2x2 blocks (C[0:2,0:2], C[0:2,2:4], C[0:2,4:6]):
    # Each is sum of two 2x2x2 multiplications
    # We can use Strassen for each, so rank 7+7=14 per block, total 3*14=42
    #
    # For the bottom row blocks (C[2:3,0:2], etc.):
    # Each is 1x2 @ 2x2 = 1x2, which is just 2 scalar multiplications
    # Rank 2 per block, total 3*2=6
    #
    # Total rank: 42 + 6 = 48
    #
    # But we can do better! The 1x2 @ 2x2 = 1x2 can be combined with adjacent terms
    #
    # Let me implement this:

    # Strassen 2x2 decomposition
    S_U, S_V, S_W = _strassen_2x2()
    
    # Initialize with maximum possible rank
    max_rank = n * m * p  # 72
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    col = 0
    
    # Top 2 rows of A, left 2 cols of B -> C[0:2, 0:2]
    # A_00 @ B_00: rows 0-1 of A, cols 0-1 of A; rows 0-1 of B, cols 0-1 of B
    # U: A indices (i,j) where i in [0,2), j in [0,2) -> row = i*4 + j
    # V: B indices (j,k) where j in [0,2), k in [0,2) -> row = j*6 + k
    # W: C indices (k,i) where k in [0,2), i in [0,2) -> row = k*3 + i
    
    # Add Strassen for A_00 @ B_00
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j, col + c] += S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + k, col + c] += S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[k * n + i, col + c] += S_W[k * 2 + i, c]
    col += 7
    
    # A_01 @ B_10: rows 0-1 of A, cols 2-3 of A; rows 2-3 of B, cols 0-1 of B
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + (j + 2), col + c] += S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[(j + 2) * p + k, col + c] += S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[k * n + i, col + c] += S_W[k * 2 + i, c]
    col += 7
    
    # C[0:2, 2:4]: A_00 @ B_01 + A_01 @ B_11
    # A_00 @ B_01: rows 0-1 of A, cols 0-1; rows 0-1 of B, cols 2-3
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j, col + c] += S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + (k + 2), col + c] += S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * n + i, col + c] += S_W[k * 2 + i, c]
    col += 7
    
    # A_01 @ B_11: rows 0-1 of A, cols 2-3; rows 2-3 of B, cols 2-3
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + (j + 2), col + c] += S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[(j + 2) * p + (k + 2), col + c] += S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[(k + 2) * n + i, col + c] += S_W[k * 2 + i, c]
    col += 7
    
    # C[0:2, 4:6]: A_00 @ B_02 + A_01 @ B_12
    # A_00 @ B_02: rows 0-1 of A, cols 0-1; rows 0-1 of B, cols 4-5
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + j, col + c] += S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[j * p + (k + 4), col + c] += S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[(k + 4) * n + i, col + c] += S_W[k * 2 + i, c]
    col += 7
    
    # A_01 @ B_12: rows 0-1 of A, cols 2-3; rows 2-3 of B, cols 4-5
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * m + (j + 2), col + c] += S_U[i * 2 + j, c]
        for j in range(2):
            for k in range(2):
                V[(j + 2) * p + (k + 4), col + c] += S_V[j * 2 + k, c]
        for k in range(2):
            for i in range(2):
                W[(k + 4) * n + i, col + c] += S_W[k * 2 + i, c]
    col += 7
    
    # Bottom row: C[2:3, 0:2] = A_10 @ B_00 + A_11 @ B_10
    # A_10: row 2 of A, cols 0-1 (1x2)
    # B_00: rows 0-1 of B, cols 0-1 (2x2)
    # Result: 1x2
    # This is just 2 scalar products: sum over j of A[2,j]*B[j,k]
    # For k=0: A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # For k=1: A[2,0]*B[0,1] + A[2,1]*B[1,1]
    # Rank 2 for this block
    
    # A[2,0]*B[0,0] contributes to C[2,0]
    U[2 * m + 0, col] = 1
    V[0 * p + 0, col] = 1
    W[0 * n + 2, col] = 1
    col += 1
    
    # A[2,1]*B[1,0] contributes to C[2,0]
    U[2 * m + 1, col] = 1
    V[1 * p + 0, col] = 1
    W[0 * n + 2, col] = 1
    col += 1
    
    # A[2,0]*B[0,1] contributes to C[2,1]
    U[2 * m + 0, col] = 1
    V[0 * p + 1, col] = 1
    W[1 * n + 2, col] = 1
    col += 1
    
    # A[2,1]*B[1,1] contributes to C[2,1]
    U[2 * m + 1, col] = 1
    V[1 * p + 1, col] = 1
    W[1 * n + 2, col] = 1
    col += 1
    
    # C[2:3, 2:4] = A_10 @ B_01 + A_11 @ B_11
    # A_10 @ B_01: A[2,0:2] @ B[0:2, 2:4]
    # A[2,0]*B[0,2] -> C[2,2]
    U[2 * m + 0, col] = 1
    V[0 * p + 2, col] = 1
    W[2 * n + 2, col] = 1
    col += 1
    
    # A[2,1]*B[1,2] -> C[2,2]
    U[2 * m + 1, col] = 1
    V[1 * p + 2, col] = 1
    W[2 * n + 2, col] = 1
    col += 1
    
    # A[2,0]*B[0,3] -> C[2,3]
    U[2 * m + 0, col] = 1
    V[0 * p + 3, col] = 1
    W[3 * n + 2, col] = 1
    col += 1
    
    # A[2,1]*B[1,3] -> C[2,3]
    U[2 * m + 1, col] = 1
    V[1 * p + 3, col] = 1
    W[3 * n + 2, col] = 1
    col += 1
    
    # A_11 @ B_11: A[2,2:4] @ B[2:4, 2:4]
    # A[2,2]*B[2,2] -> C[2,2]
    U[2 * m + 2, col] = 1
    V[2 * p + 2, col] = 1
    W[2 * n + 2, col] = 1
    col += 1
    
    # A[2,3]*B[3,2] -> C[2,2]
    U[2 * m + 3, col] = 1
    V[3 * p + 2, col] = 1
    W[2 * n + 2, col] = 1
    col += 1
    
    # A[2,2]*B[2,3] -> C[2,3]
    U[2 * m + 2, col] = 1
    V[2 * p + 3, col] = 1
    W[3 * n + 2, col] = 1
    col += 1
    
    # A[2,3]*B[3,3] -> C[2,3]
    U[2 * m + 3, col] = 1
    V[3 * p + 3, col] = 1
    W[3 * n + 2, col] = 1
    col += 1
    
    # C[2:3, 4:6] = A_10 @ B_02 + A_11 @ B_12
    # A_10 @ B_02: A[2,0:2] @ B[0:2, 4:6]
    # A[2,0]*B[0,4] -> C[2,4]
    U[2 * m + 0, col] = 1
    V[0 * p + 4, col] = 1
    W[4 * n + 2, col] = 1
    col += 1
    
    # A[2,1]*B[1,4] -> C[2,4]
    U[2 * m + 1, col] = 1
    V[1 * p + 4, col] = 1
    W[4 * n + 2, col] = 1
    col += 1
    
    # A[2,0]*B[0,5] -> C[2,5]
    U[2 * m + 0, col] = 1
    V[0 * p + 5, col] = 1
    W[5 * n + 2, col] = 1
    col += 1
    
    # A[2,1]*B[1,5] -> C[2,5]
    U[2 * m + 1, col] = 1
    V[1 * p + 5, col] = 1
    W[5 * n + 2, col] = 1
    col += 1
    
    # A_11 @ B_12: A[2,2:4] @ B[2:4, 4:6]
    # A[2,2]*B[2,4] -> C[2,4]
    U[2 * m + 2, col] = 1
    V[2 * p + 4, col] = 1
    W[4 * n + 2, col] = 1
    col += 1
    
    # A[2,3]*B[3,4] -> C[2,4]
    U[2 * m + 3, col] = 1
    V[3 * p + 4, col] = 1
    W[4 * n + 2, col] = 1
    col += 1
    
    # A[2,2]*B[2,5] -> C[2,5]
    U[2 * m + 2, col] = 1
    V[2 * p + 5, col] = 1
    W[5 * n + 2, col] = 1
    col += 1
    
    # A[2,3]*B[3,5] -> C[2,5]
    U[2 * m + 3, col] = 1
    V[3 * p + 5, col] = 1
    W[5 * n + 2, col] = 1
    col += 1
    
    # Total rank used: col
    rank = col
    print(f"Constructed decomposition with rank {rank}")
    
    # Verify
    U_list = U[:n*m, :rank].tolist()
    V_list = V[:m*p, :rank].tolist()
    W_list = W[:p*n, :rank].tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, rank):
        print("Decomposition verified successfully!")
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        print("Decomposition verification failed, falling back to schoolbook")
    
    # Fallback: schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col2 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col2] = 1
                V_school[j * p + k][col2] = 1
                W_school[k * n + i][col2] = 1
                col2 += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
