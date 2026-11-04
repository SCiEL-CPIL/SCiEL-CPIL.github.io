import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor of a 3x3 matrix
    multiplied by a 3x8 matrix.
    
    Strategy:
    Split the 3x3 x 3x8 multiplication into blocks based on the partition of indices:
    i in {0,1} vs {2}, j in {0,1} vs {2}, k in {0..7}.
    
    The tensor is a direct sum of 4 independent sub-tensors:
    1. T11: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8] (2x2 x 2x8)
       This can be decomposed as 4 independent 2x2 x 2x2 blocks.
       Each 2x2x2 block uses Strassen's algorithm (rank 7).
       Total rank for T11: 4 * 7 = 28.
       
    2. T12: A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8] (2x1 x 1x8)
       This is a simple outer product structure.
       Rank = 2 * 1 * 8 = 16.
       
    3. T21: A[2, 0:2] * B[0:2, 0:8] -> C[2, 0:8] (1x2 x 2x8)
       This is also a simple structure.
       Rank = 1 * 2 * 8 = 16.
       
    4. T22: A[2, 2] * B[2, 0:8] -> C[2, 0:8] (1x1 x 1x8)
       Rank = 1 * 1 * 8 = 8.
       
    Total rank = 28 + 16 + 16 + 8 = 68.
    
    This is an improvement over the schoolbook rank of 72.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 68
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 block
    # A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]
    # Strassen terms:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B10)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m2 + m4
    # C10 = m1 + m3
    # C11 = m1 - m2 + m3 + m5
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        # Indices for A (U): i0, i0+1 ; j0, j0+1
        # Indices for B (V): j0, j0+1 ; k0, k0+1
        # Indices for C (W): k0, k0+1 ; i0, i0+1
        
        # m1
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # m2
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # m3
        U[i0*m + j0, col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # m4
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + (k0+1), col] += 1
        V[j0*p + k0, col] -= 1
        W[(k0+1)*n + (i0+1), col] += 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # m5
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        col += 1
        
        # m6
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # m7
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # Block T11: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    # This is 4 independent 2x2x2 blocks for k in {0,1}, {2,3}, {4,5}, {6,7}
    for b in range(4):
        k0 = b * 2
        add_strassen_block(0, 0, k0)
    
    # Block T12: A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]
    # C[i, k] = A[i, 2] * B[2, k] for i in {0,1}, k in {0..7}
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Block T21: A[2, 0:2] * B[0:2, 0:8] -> C[2, 0:8]
    # C[2, k] = sum_j A[2, j] * B[j, k] for j in {0,1}, k in {0..7}
    for j in range(2):
        for k in range(8):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Block T22: A[2, 2] * B[2, 0:8] -> C[2, 0:8]
    # C[2, k] = A[2, 2] * B[2, k]
    for k in range(8):
        u_idx = 2 * m + 2
        v_idx = 2 * p + k
        w_idx = k * n + 2
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1
        col += 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
