"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # Matrix multiplication tensor T[i, j, k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # T has shape (n, m, p) where T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k]
    # contributes to C[i,k]. Actually the tensor is: T[i,j,k] = delta_{i,k} ... no.
    # The matrix multiplication tensor: for C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's the structure tensor)
    # Actually T[i,j,k] is 1 when the (i,j) entry of A and (j,k) entry of B multiply
    # to contribute to (i,k) entry of C. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right either. The tensor is defined such that:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid indices.
    
    # Build the full tensor
    T = np.ones((n, m, p))
    
    # Use a greedy rank-1 extraction to find a sparse decomposition
    # Start with schoolbook and try to reduce
    
    # First, build schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # This is a simple heuristic: try to combine columns
    
    # Actually, let's use a different structural approach:
    # Split the 7 columns into 4 + 3
    # For the 4x4x4 part, use a known good decomposition
    # For the 4x4x3 part, use schoolbook (rank 12)
    
    # Known: 4x4x4 matrix multiplication has rank 49 (Strassen-like)
    # Let me build a 4x4x4 decomposition using Strassen recursively
    
    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 matrix multiplication with minimal known rank."""
        # Use the 7-term Strassen for 2x2, then combine
        # 4x4 = [[A11, A12], [A21, A22]] where each is 2x2
        # This gives rank 7*4 = 28? No, that's not right.
        # Actually, the known optimal for 4x4x4 is 49.
        # Let me just use schoolbook for 4x4x4 (rank 64) and 4x4x3 (rank 12) = 76 total
        # That's worse than 112... wait no, schoolbook 4x4x7 is 112.
        # 64 + 12 = 76 < 112. Good.
        
        # But we can do better. Let me use a 2x2 Strassen for the 4x4 block.
        # 4x4x4: split into 2x2 blocks, use Strassen (rank 7) for each of 4 sub-problems?
        # No, Strassen computes a 2x2 product with 7 multiplications of 1x1.
        # For 4x4, we'd have 4 2x2 sub-products, each with rank 7, total 28.
        # But there's overlap. The actual rank of 4x4x4 tensor is 49.
        
        # Let me just use a simple decomposition and return it.
        # For now, use schoolbook for simplicity in this block.
        nn, mm, pp = 4, 4, 4
        r = nn * mm * pp
        U = np.zeros((nn * mm, r))
        V = np.zeros((mm * pp, r))
        W = np.zeros((pp * nn, r))
        c = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, c] = 1
                    V[j * pp + k, c] = 1
                    W[k * nn + i, c] = 1
                    c += 1
        return U, V, W
    
    # 4x4x4 schoolbook: rank 64
    U1, V1, W1 = strassen_4x4x4()
    
    # 4x4x3 schoolbook: rank 12
    nn, mm, pp = 4, 4, 3
    r2 = nn * mm * pp
    U2 = np.zeros((nn * mm, r2))
    V2 = np.zeros((mm * pp, r2))
    W2 = np.zeros((pp * nn, r2))
    c = 0
    for i in range(nn):
        for j in range(mm):
            for k in range(pp):
                U2[i * mm + j, c] = 1
                V2[j * pp + k, c] = 1
                W2[k * nn + i, c] = 1
                c += 1
    
    # Combine: U has shape (16, 64+12), V has shape (16, 64+12) but V2 has shape (12, 12)
    # Wait, V for 4x4x4 has shape (4*4, 64) = (16, 64)
    # V for 4x4x3 has shape (4*3, 12) = (12, 12)
    # These don't match! V must have the same row count for both parts.
    # The issue: in the first part, B is 4x4, so V rows are indexed by (j,k) with j in 0..3, k in 0..3
    # In the second part, B is 4x3, so V rows are indexed by (j,k) with j in 0..3, k in 0..2
    # But in the original problem, B is 4x7, so V rows are (j,k) with j in 0..3, k in 0..6
    # So I need to map the V indices correctly.
    
    # Let me rebuild this properly.
    # Original: A is 4x4, B is 4x7, C is 4x7
    # U: shape (4*4, rank) = (16, rank), indexed by (i,j)
    # V: shape (4*7, rank) = (28, rank), indexed by (j,k) where j in 0..3, k in 0..6
    # W: shape (7*4, rank) = (28, rank), indexed by (k,i) where k in 0..6, i in 0..3
    
    # For the first 4 columns of B (k=0..3):
    # U1: (16, r1), V1: (4*4, r1) = (16, r1) -> maps to rows 0..15 of V (j*4+k for k=0..3)
    # W1: (4*4, r1) = (16, r1) -> maps to rows k*4+i for k=0..3
    
    # For the last 3 columns of B (k=4..6):
    # U2: (16, r2), V2: (4*3, r2) = (12, r2) -> maps to rows j*7+k for j=0..3, k=4..6
    # W2: (3*4, r2) = (12, r2) -> maps to rows k*4+i for k=4..6
    
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    
    U = np.zeros((16, total_rank))
    V = np.zeros((28, total_rank))
    W = np.zeros((28, total_rank))
    
    # First part (k=0..3)
    U[:, :r1] = U1
    # V1 has shape (16, r1), rows indexed by j*4+k for j in 0..3, k in 0..3
    # In V, row j*7+k for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, :r1] = V1[j * 4 + k, :]
    # W1 has shape (16, r1), rows indexed by k*4+i for k in 0..3, i in 0..3
    # In W, row k*4+i for k in 0..3, i in 0..3
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :r1] = W1[k * 4 + i, :]
    
    # Second part (k=4..6)
    U[:, r1:r1+r2] = U2
    # V2 has shape (12, r2), rows indexed by j*3+k' for j in 0..3, k' in 0..2
    # In V, row j*7+(4+k') for j in 0..3, k' in 0..2
    for j in range(4):
        for kp in range(3):
            k = 4 + kp
            V[j * 7 + k, r1:r1+r2] = V2[j * 3 + kp, :]
    # W2 has shape (12, r2), rows indexed by k'*4+i for k' in 0..2, i in 0..3
    # In W, row (4+k')*4+i for k' in 0..2, i in 0..3
    for kp in range(3):
        k = 4 + kp
        for i in range(4):
            W[k * 4 + i, r1:r1+r2] = W2[kp * 4 + i, :]
    
    # Convert to list of lists with integer entries
    U_list = [[int(U[i, j]) for j in range(total_rank)] for i in range(16)]
    V_list = [[int(V[i, j]) for j in range(total_rank)] for i in range(28)]
    W_list = [[int(W[i, j]) for j in range(total_rank)] for i in range(28)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
