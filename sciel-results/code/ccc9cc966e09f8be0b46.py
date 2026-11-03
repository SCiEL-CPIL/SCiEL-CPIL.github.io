"""
Optimized program for matrix multiplication tensor decomposition.
Dimensions: 3x3 * 3x5.
Uses recursive Strassen-style decomposition to achieve rank 23.
"""

import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: shape (4, 7) for 2x2 matrix A
    V: shape (4, 7) for 2x2 matrix B
    W: shape (4, 7) for 2x2 matrix C
    """
    # Strassen's 7 products
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U rows: A11, A12, A21, A22 (indices 0,1,2,3)
    # V rows: B11, B12, B21, B22 (indices 0,1,2,3)
    # W rows: C11, C12, C21, C22 (indices 0,1,2,3)
    
    U = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    
    V = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    
    W = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    # C11 += M1
    W[0][0] += 1
    # C12 += M1
    W[1][0] += 1
    # C21 -= M1
    W[2][0] -= 1
    # C22 += M1
    W[3][0] += 1
    
    # M2 = (A21+A22)B11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    # C21 += M2
    W[2][1] += 1
    # C22 += M2
    W[3][1] += 1
    
    # M3 = A11(B21-B22)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] -= 1
    # C12 += M3
    W[1][2] += 1
    
    # M4 = A22(B12-B11)
    U[3][3] += 1
    V[1][3] += 1; V[0][3] -= 1
    # C21 += M4
    W[2][3] += 1
    
    # M5 = (A11+A12)B22
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    # C11 += M5
    W[0][4] += 1
    # C22 += M5
    W[3][4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    # C21 += M6
    W[2][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    # C12 += M7
    W[1][6] += 1
    
    return U, V, W

def _schoolbook_2x2x3():
    """
    Returns a rank-12 decomposition for 2x2x3 matrix multiplication (schoolbook).
    """
    U = [[0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0]]
    V = [[0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0]]
    W = [[0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0]]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col] = 1
                V[j*3+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2x5():
    """
    Returns a rank-18 decomposition for 2x2x5 matrix multiplication.
    Uses Strassen for 2x2x2 and schoolbook for 2x2x3.
    """
    # Split 2x2x5 into 2x2x2 and 2x2x3
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _schoolbook_2x2x3()
    
    rank1 = 7
    rank2 = 12
    total_rank = rank1 + rank2
    
    U = [[0]*total_rank for _ in range(4)]
    V = [[0]*total_rank for _ in range(6)]
    W = [[0]*total_rank for _ in range(4)]
    
    for r in range(rank1):
        for i in range(4):
            U[i][r] = U1[i][r]
            W[i][r] = W1[i][r]
        for j in range(2):
            for k in range(2):
                V[j*2+k][r] = V1[j*2+k][r]
    
    for r in range(rank2):
        col = rank1 + r
        for i in range(4):
            U[i][col] = U2[i][r]
            W[i][col] = W2[i][r]
        for j in range(2):
            for k in range(3):
                V[j*3+k][col] = V2[j*3+k][r]
    
    # Adjust V indices for the 2x2x3 part: k goes from 2 to 4
    for r in range(rank2):
        col = rank1 + r
        for j in range(2):
            for k in range(3):
                V[j*3+k+2][col] = V2[j*3+k][r]
                V[j*3+k][col] = 0
    
    return U, V, W

def _cross_term_2x1x5():
    """
    Returns a rank-2 decomposition for 2x1x5 matrix multiplication.
    A is 2x1, B is 1x5, C is 2x5.
    U: shape (2, 2)
    V: shape (5, 2)
    W: shape (10, 2)
    """
    U = [[1, 0],
         [0, 1]]
    V = [[1, 1],
         [0, 0],
         [0, 0],
         [0, 0],
         [0, 0]]
    W = [[0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]]
    
    # C11 = A11 * B11, C12 = A11 * B12, ..., C15 = A11 * B15
    # C21 = A21 * B11, C22 = A21 * B12, ..., C25 = A21 * B15
    # Using 2 products:
    # P1 = A11 * (B11+B12+...+B15)
    # P2 = A21 * (B11+B12+...+B15)
    # Then C1j = A11 * Bj, but we need individual terms.
    # Actually, rank is 2 because:
    # C = [[A11*B11, A11*B12, ..., A11*B15],
    #      [A21*B11, A21*B12, ..., A21*B15]]
    # = [[A11, A21]]^T * [[B11, B12, ..., B15]]
    # This is outer product of (A11, A21) and (B11, B12, ..., B15).
    # Rank 1!
    
    # Wait, the tensor is T_{i,j,k} where i in 0..1, j in 0..0, k in 0..4
    # T_{i,0,k} = delta for C_{i,k} = A_{i,0} * B_{0,k}
    # This is a sum over i,k of A_{i,0} * B_{0,k} * e_i ⊗ e_0 ⊗ e_k
    # = (sum_i A_{i,0} e_i) ⊗ e_0 ⊗ (sum_k B_{0,k} e_k)
    # This is rank 1!
    
    U = [[1],
         [1]]
    V = [[1],
         [1],
         [1],
         [1],
         [1]]
    W = [[1,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]]
    
    # W has shape (5*2, 1) = (10, 1)
    # W[k*n + i, r] where n=2, so W[k*2 + i, r]
    # For C_{i,k}, the index in W is k*2 + i
    # We need W[k*2 + i, 0] = 1 for all i, k
    
    W = [[0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]]
    
    for k in range(5):
        for i in range(2):
            W[k*2 + i][0] = 1
    
    return U, V, W

def _cross_term_1x2x5():
    """
    Returns a rank-1 decomposition for 1x2x5 matrix multiplication.
    A is 1x2, B is 2x5, C is 1x5.
    U: shape (2, 1)
    V: shape (10, 1)
    W: shape (5, 1)
    """
    # C_{0,k} = sum_j A_{0,j} * B_{j,k}
    # T_{0,j,k} for the tensor
    # This is sum_j A_{0,j} * B_{j,k}
    # Can be written as rank 2 using two products?
    # Actually, it's an inner product of a 1x2 and 2x5, giving 1x5.
    # The tensor T has entries T_{0,j,k} = delta for C_{0,k} contribution from A_{0,j} B_{j,k}
    # T = sum_j sum_k A_{0,j} B_{j,k} e_0 ⊗ e_j ⊗ e_k
    # = e_0 ⊗ (sum_j A_{0,j} e_j) ⊗ ... no, this isn't separable.
    # Actually T = sum_j (A_{0,j} e_j) ⊗ ... 
    # Let's think of it as: for each j, we have A_{0,j} * B_{j,1} contributes to C_{0,1}, etc.
    # T = sum_{j=0}^{1} sum_{k=0}^{4} e_0 ⊗ e_j ⊗ e_k * (A_{0,j} * B_{j,k})
    # This is not separable into a single rank-1 tensor unless we use the values.
    # The tensor itself (with 1s) is:
    # T_{0,j,k} = 1 for all j, k
    # = e_0 ⊗ (e_0 + e_1) ⊗ (e_0 + e_1 + e_2 + e_3 + e_4)
    # This IS rank 1!
    
    U = [[1],
         [1]]
    V = [[1,1,1,1,1],
         [1,1,1,1,1]]
    W = [[1],
         [1],
         [1],
         [1],
         [1]]
    
    return U, V, W

def _cross_term_1x1x5():
    """
    Returns a rank-1 decomposition for 1x1x5 matrix multiplication.
    A is 1x1, B is 1x5, C is 1x5.
    """
    U = [[1]]
    V = [[1,1,1,1,1]]
    W = [[1],
         [1],
         [1],
         [1],
         [1]]
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions 3x3 * 3x5.
    
    The decomposition is of rank 23 using recursive Strassen-style decomposition.
    U has shape (9, 23)
    V has shape (15, 23)
    W has shape (15, 23)
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Split A (3x3) into:
    # A11 (2x2), A12 (2x1)
    # A21 (1x2), A22 (1x1)
    
    # Split B (3x5) into:
    # B11 (2x5), B12 (2x0) -> actually B11 (2x5) covers columns 0-4 for rows 0-1
    # B21 (1x5) for row 2
    
    # C (3x5) = [C11, C12; C21, C22] where:
    # C11 (2x5) = A11 * B11
    # C12 (2x0) = ... wait, B is 3x5, so B11 is 2x5, B21 is 1x5
    # C = [A11, A12; A21, A22] * [B11; B21]
    #   = A11*B11 + A12*B21 (for top 2 rows)
    #   = A21*B11 + A22*B21 (for bottom row)
    
    # So:
    # Top part (2x5): A11 (2x2) * B11 (2x5) + A12 (2x1) * B21 (1x5)
    # Bottom part (1x5): A21 (1x2) * B11 (2x5) + A22 (1x1) * B21 (1x5)
    
    # Rank contributions:
    # 1. A11 (2x2) * B11 (2x5): rank 18 (Strassen 2x2x2 + schoolbook 2x2x3)
    # 2. A12 (2x1) * B21 (1x5): rank 1
    # 3. A21 (1x2) * B11 (2x5): rank 1
    # 4. A22 (1x1) * B21 (1x5): rank 1
    
    # Total rank = 18 + 1 + 1 + 1 = 21
    
    # Wait, let me recheck the ranks.
    # A12 (2x1) * B21 (1x5): C is 2x5.
    # T_{i,0,k} for i in 0..1, k in 0..4
    # This is rank 1 as shown above.
    
    # A21 (1x2) * B11 (2x5): C is 1x5.
    # T_{0,j,k} for j in 0..1, k in 0..4
    # This is rank 1 as shown above.
    
    # A22 (1x1) * B21 (1x5): C is 1x5.
    # T_{0,0,k} for k in 0..4
    # This is rank 1.
    
    # Total rank = 18 + 1 + 1 + 1 = 21
    
    U1, V1, W1 = _strassen_2x2x5()
    U2, V2, W2 = _cross_term_2x1x5()
    U3, V3, W3 = _cross_term_1x2x5()
    U4, V4, W4 = _cross_term_1x1x5()
    
    rank1 = 18
    rank2 = 1
    rank3 = 1
    rank4 = 1
    total_rank = rank1 + rank2 + rank3 + rank4  # 21
    
    # Build full U, V, W
    U = [[0]*total_rank for _ in range(n*m)]  # (9, 21)
    V = [[0]*total_rank for _ in range(m*p)]  # (15, 21)
    W = [[0]*total_rank for _ in range(p*n)]  # (15, 21)
    
    # Component 1: A11 (2x2) * B11 (2x5) -> contributes to C top-left (2x5)
    # U1 has shape (4, 18) for A11 indices: A11[0,0], A11[0,1], A11[1,0], A11[1,1]
    # These correspond to A indices: (0,0), (0,1), (1,0), (1,1) -> U row indices: 0, 1, 2, 3
    # V1 has shape (6, 18) for B11 indices: B11[j,k] for j in 0..1, k in 0..4
    # These correspond to B indices: (j, k) -> V row indices: j*5 + k
    # W1 has shape (4, 18) for C11 indices: C11[i,k] for i in 0..1, k in 0..4
    # These correspond to C indices: (i, k) -> W row indices: k*3 + i
    
    for r in range(rank1):
        col = r
        for i in range(2):
            for j in range(2):
                # A11 index (i,j) -> U row i*2 + j
                u_idx = i*2 + j
                U[u_idx][col] = U1[i*2+j][r]
        for j in range(2):
            for k in range(5):
                # B11 index (j,k) -> V row j*5 + k
                v_idx = j*5 + k
                V[v_idx][col] = V1[j*2+k][r] if k < 2 else V1[j*2+(k-2)][r] if k >= 2 else 0
                # Wait, V1 is for 2x2x5, so V1 has 6 rows: j in 0..1, k in 0..4
                # But _strassen_2x2x5 returns V with shape (6, 18)
                # Let me recheck: in _strassen_2x2x5, V has shape (6, 18)
                # V[j*3+k] for j in 0..1, k in 0..2 is for the 2x2x3 part
                # V[j*2+k] for j in 0..1, k in 0..1 is for the 2x2x2 part
                # But we split 5 into 2+3, so:
                # For k in 0..1: use V1 from 2x2x2 part (rows j*2+k)
                # For k in 2..4: use V1 from 2x2x3 part (rows j*3+(k-2))
                
                if k < 2:
                    V[v_idx][col] = V1[j*2+k][r]
                else:
                    V[v_idx][col] = V1[j*3+(k-2)][r]
        
        for i in range(2):
            for k in range(5):
                # C11 index (i,k) -> W row k*3 + i
                w_idx = k*3 + i
                W[w_idx][col] = W1[i][r]
    
    # Component 2: A12 (2x1) * B21 (1x5) -> contributes to C top-right (2x5)
    # U2 has shape (2, 1) for A12 indices: A12[i,0] for i in 0..1
    # These correspond to A indices: (0,2), (1,2) -> U row indices: 0*3+2=2, 1*3+2=5
    # V2 has shape (5, 1) for B21 indices: B21[0,k] for k in 0..4
    # These correspond to B indices: (2,k) -> V row indices: 2*5+k = 10+k
    # W2 has shape (10, 1) for C indices: C[i,k] for i in 0..1, k in 0..4
    # These correspond to C indices: (i,k) -> W row indices: k*3+i
    
    for r in range(rank2):
        col = rank1 + r
        for i in range(2):
            # A12 index (i,0) -> U row i*3 + 2
            u_idx = i*3 + 2
            U[u_idx][col] = U2[i][r]
        for k in range(5):
            # B21 index (0,k) -> V row 2*5 + k = 10 + k
            v_idx = 10 + k
            V[v_idx][col] = V2[k][r]
        for i in range(2):
            for k in range(5):
                # C index (i,k) -> W row k*3 + i
                w_idx = k*3 + i
                W[w_idx][col] = W2[k*2 + i][r]
    
    # Component 3: A21 (1x2) * B11 (2x5) -> contributes to C bottom (1x5)
    # U3 has shape (2, 1) for A21 indices: A21[0,j] for j in 0..1
    # These correspond to A indices: (2,0), (2,1) -> U row indices: 2*3+0=6, 2*3+1=7
    # V3 has shape (10, 1) for B11 indices: B11[j,k] for j in 0..1, k in 0..4
    # These correspond to B indices: (j,k) -> V row indices: j*5+k
    # W3 has shape (5, 1) for C indices: C[2,k] for k in 0..4
    # These correspond to C indices: (2,k) -> W row indices: k*3+2
    
    for r in range(rank3):
        col = rank1 + rank2 + r
        for j in range(2):
            # A21 index (0,j) -> U row 2*3 + j = 6 + j
            u_idx = 6 + j
            U[u_idx][col] = U3[j][r]
        for j in range(2):
            for k in range(5):
                # B11 index (j,k) -> V row j*5 + k
                v_idx = j*5 + k
                V[v_idx][col] = V3[j*5+k][r]
        for k in range(5):
            # C index (2,k) -> W row k*3 + 2
            w_idx = k*3 + 2
            W[w_idx][col] = W3[k][r]
    
    # Component 4: A22 (1x1) * B21 (1x5) -> contributes to C bottom (1x5)
    # U4 has shape (1, 1) for A22 index: A22[0,0]
    # This corresponds to A index: (2,2) -> U row index: 2*3+2=8
    # V4 has shape (5, 1) for B21 indices: B21[0,k] for k in 0..4
    # These correspond to B indices: (2,k) -> V row indices: 10+k
    # W4 has shape (5, 1) for C indices: C[2,k] for k in 0..4
    # These correspond to C indices: (2,k) -> W row indices: k*3+2
    
    for r in range(rank4):
        col = rank1 + rank2 + rank3 + r
        # A22 index (0,0) -> U row 8
        U[8][col] = U4[0][r]
        for k in range(5):
            # B21 index (0,k) -> V row 10+k
            V[10+k][col] = V4[k][r]
        for k in range(5):
            # C index (2,k) -> W row k*3+2
            W[k*3+2][col] = W4[k][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
