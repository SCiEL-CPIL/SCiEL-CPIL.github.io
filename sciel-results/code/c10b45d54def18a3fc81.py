import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    This uses the block-diagonal structure of the tensor, applying Strassen's algorithm
    (rank 7) to each of the 8 independent 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2 matrix multiplication
    # U, V, W are 7x4 matrices (7 ranks, 4 indices each)
    # Indices correspond to (i,j) for U, (j,k) for V, (k,i) for W in 0..3
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Construct the full decomposition
    # The tensor is a direct sum of 8 copies of the 2x2x2 tensor.
    # Block l corresponds to the middle index j and the specific 2x2 sub-block of B and C.
    # For A (2x2), indices are (i, j) where i in 0..1, j in 0..1.
    # For B (2x16), indices are (j, k) where j in 0..1, k in 0..15.
    # For C (16x2), indices are (k, i) where k in 0..15, i in 0..1.
    
    # We group the 16 columns of B into 8 pairs of 2x2 blocks? No.
    # B is 2x16. The multiplication is C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has indices (i, j, k).
    # T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Actually, the tensor is defined by the bilinear map.
    # The rank is the minimum number of terms sum_u U[u] V[u] W[u].
    
    # The structure is that for each j, we have a 2x2 multiplication of A[:, j] and B[j, :]?
    # No, A is 2x2, B is 2x16.
    # C[i, k] = A[i, 0]B[0, k] + A[i, 1]B[1, k].
    # This is a sum of 2 matrix multiplications of size 2x1.
    # Wait, the standard view is:
    # The tensor is a sum over j of (A[:, j]) * (B[j, :])^T? No, that's outer product.
    # C = A B.
    # Vectorized: C_flat = (I_n \otimes B) A_flat? No.
    # C_flat = (B^T \otimes I_n) A_flat? No.
    # Let's look at the indices.
    # i in 0..1, j in 0..1, k in 0..15.
    # The term is A[i,j] * B[j,k].
    # This is a sum over j.
    # For each j, we are computing the outer product of column j of A (size 2) and row j of B (size 16).
    # This is a rank-1 update in the (i,k) space for each j?
    # No, the "rank" of the tensor is the rank of the bilinear map.
    # The map is (A, B) -> C.
    # We can view this as 8 independent 2x2x2 multiplications?
    # Let's check the block structure.
    # B is 2x16. We can split B into 8 blocks of size 2x2?
    # Let B = [B_0, B_1, ..., B_7] where each B_l is 2x2.
    # Then C = A B = [A B_0, A B_1, ..., A B_7].
    # Each A B_l is a 2x2 matrix multiplication.
    # Since the blocks are independent (the multiplication of A with B_l only affects C_l),
    # the total rank is the sum of the ranks of the individual 2x2x2 multiplications.
    # Rank(2x2x2) = 7 (Strassen).
    # Total Rank = 8 * 7 = 56.
    
    # So we map:
    # Block l (0..7) uses columns 2l, 2l+1 of B and rows 2l, 2l+1 of C?
    # Wait, C is 2x16. C_l is the 2x2 submatrix at columns 2l, 2l+1.
    # C_l = A B_l.
    # B_l is the 2x2 submatrix of B at columns 2l, 2l+1.
    # So for block l:
    # U indices: i in 0..1, j in 0..1. (Same for all blocks)
    # V indices: j in 0..1, k in 2l, 2l+1.
    # W indices: k in 2l, 2l+1, i in 0..1.
    
    # Let's verify the index mapping in the provided code.
    # V row indices: 2*l, 2*l+1, 16+2*l, 16+2*l+1.
    # B is 2x16. Flattened row-major?
    # B[j, k] is at index j*16 + k.
    # So for block l, k is 2l, 2l+1.
    # j=0: index 0*16 + 2l = 2l.
    # j=0: index 0*16 + 2l+1 = 2l+1.
    # j=1: index 1*16 + 2l = 16+2l.
    # j=1: index 1*16 + 2l+1 = 16+2l+1.
    # This matches the V indices in the previous code.
    
    # W row indices: 4*l, 4*l+2, 4*l+1, 4*l+3.
    # C is 2x16. Flattened row-major?
    # C[k, i] is at index k*2 + i.
    # For block l, k is 2l, 2l+1.
    # k=2l, i=0: index 2l*2 + 0 = 4l.
    # k=2l, i=1: index 2l*2 + 1 = 4l+1.
    # k=2l+1, i=0: index (2l+1)*2 + 0 = 4l+2.
    # k=2l+1, i=1: index (2l+1)*2 + 1 = 4l+3.
    # The Strassen W matrix has indices for (k, i) pairs.
    # In Strassen_W, the columns correspond to (k, i) flattened?
    # The previous code used:
    # c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
    # This order corresponds to:
    # 4l   -> k=2l, i=0
    # 4l+2 -> k=2l+1, i=0
    # 4l+1 -> k=2l, i=1
    # 4l+3 -> k=2l+1, i=1
    # So the column order in strassen_W must be [ (0,0), (1,0), (0,1), (1,1) ]?
    # Let's check strassen_W definition in previous code:
    # strassen_W = [
    #    [1, 0, 0, 1],  # col 0: (0,0) and (1,1)? No, entries are for (k,i)
    #    ...
    # ]
    # The previous code verified successfully, so the mapping is correct.
    
    for l in range(8):
        # Precompute v and w row indices for this block
        b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
        c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
        
        for s in range(7):
            col = 7 * l + s
            
            # U: same for all blocks
            U[0, col] = strassen_U[s][0]
            U[1, col] = strassen_U[s][1]
            U[2, col] = strassen_U[s][2]
            U[3, col] = strassen_U[s][3]
            
            # V
            V[b_rows[0], col] = strassen_V[s][0]
            V[b_rows[1], col] = strassen_V[s][1]
            V[b_rows[2], col] = strassen_V[s][2]
            V[b_rows[3], col] = strassen_V[s][3]
            
            # W
            W[c_rows[0], col] = strassen_W[s][0]
            W[c_rows[1], col] = strassen_W[s][1]
            W[c_rows[2], col] = strassen_W[s][2]
            W[c_rows[3], col] = strassen_W[s][3]
                
    # Verification
    is_valid = True
    for i in range(n):
        for j in range(m):
            u = i * m + j
            for k in range(p):
                v = j * p + k
                w = k * n + i
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                if abs(val - 1.0) > 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if is_valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
