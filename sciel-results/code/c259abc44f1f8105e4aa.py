"""Matrix multiplication 3x4 by 4x7: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C = A @ B
        Crec = np.zeros((n, p))
        for c in range(U.shape[1]):
            L = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    L[i] += U[i*m+j, c] * A[i, j]
            R = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    R[k] += V[j*p+k, c] * B[j, k]
            for k in range(p):
                for i in range(n):
                    Crec[i, k] += L[i] * R[k] * W[k*n+i, c]
        if not np.allclose(Crec, C, atol=1e-9):
            return False
    return True

def _construct_3x2x2_rank10():
    """
    Returns U, V, W for 3x2 by 2x2 multiplication with rank 10.
    A is 3x2, B is 2x2, C is 3x2.
    U: (6, 10), V: (4, 10), W: (6, 10).
    """
    n, m, p = 3, 2, 2
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3, A[2,0]=4, A[2,1]=5
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[1,0]=1, C[2,0]=2, C[0,1]=3, C[1,1]=4, C[2,1]=5
    # W index: C[i,k] -> W[k*n+i] = W[k*3+i]
    # So C[0,0]->W[0], C[1,0]->W[1], C[2,0]->W[2], C[0,1]->W[3], C[1,1]->W[4], C[2,1]->W[5]
    
    # Using the 10-term scheme:
    # M1 = A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # M2 = A[0,1]*B[1,0] -> C[0,0]
    U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
    # M3 = A[1,0]*B[0,0] -> C[1,0]
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # M4 = A[1,1]*B[1,0] -> C[1,0]
    U[3, 3] = 1; V[2, 3] = 1; W[1, 3] = 1
    # M5 = A[2,0]*B[0,0] -> C[2,0]
    U[4, 4] = 1; V[0, 4] = 1; W[2, 4] = 1
    # M6 = A[2,1]*B[1,0] -> C[2,0]
    U[5, 5] = 1; V[2, 5] = 1; W[2, 5] = 1
    
    # M7 = (A[0,0]+A[1,0]+A[2,0])*(B[0,1]+B[1,1])
    U[0, 6] = 1; U[2, 6] = 1; U[4, 6] = 1
    V[1, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1; W[4, 6] = 1; W[5, 6] = 1
    # M8 = (A[0,1]+A[1,1]+A[2,1])*(B[0,1]+B[1,1])
    U[1, 7] = 1; U[3, 7] = 1; U[5, 7] = 1
    V[1, 7] = 1; V[3, 7] = 1
    W[3, 7] = 1; W[4, 7] = 1; W[5, 7] = 1
    
    # M9 = (A[0,0]-A[1,0])*(B[0,1]-B[1,1])
    U[0, 8] = 1; U[2, 8] = -1
    V[1, 8] = 1; V[3, 8] = -1
    W[3, 8] = 1; W[4, 8] = -1
    # M10 = (A[0,1]-A[1,1])*(B[0,1]-B[1,1])
    U[1, 9] = 1; U[3, 9] = -1
    V[1, 9] = 1; V[3, 9] = -1
    W[3, 9] = 1; W[4, 9] = -1
    
    return U, V, W

def construct():
    """Construct a rank-minimal exact decomposition for 3x4 * 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try block decomposition: split m=4 into 2+2
    # A (3x4) = [A1 (3x2) | A2 (3x2)]
    # B (4x7) = [B1 (2x7); B2 (2x7)]
    # C = A1*B1 + A2*B2
    
    # For each 3x2 * 2x7 multiplication, split the 7 columns into 2+2+2+1
    # Use rank-10 for 3x2x2 blocks, rank-6 for 3x2x1 block
    # Total per A_i*B_i: 3*10 + 6 = 36
    # Total: 2 * 36 = 72
    
    try:
        U_blocks, V_blocks, W_blocks = _construct_3x2x2_rank10()
        if U_blocks is not None:
            # Verify the 3x2x2 rank-10 decomposition
            if _verify(U_blocks, V_blocks, W_blocks):
                rank_total = 72
                U = np.zeros((n*m, rank_total))
                V = np.zeros((m*p, rank_total))
                W = np.zeros((p*n, rank_total))
                
                col = 0
                for block_idx in range(2):  # Two 2-row blocks of B
                    for sub_block in range(4):  # 3 blocks of 2 cols, 1 block of 1 col
                        if sub_block < 3:
                            # 3x2x2 block
                            rank_block = 10
                            # U indices: for A1 (rows 0-1 of A) or A2 (rows 2-3)
                            # A[i,j] where i in [0,2), j in [block_idx*2, block_idx*2+2)
                            # U index: i*m + j
                            for r in range(rank_block):
                                for i in range(3):
                                    for j in range(2):
                                        if U_blocks[i*2+j, r] != 0:
                                            U[i*m + (block_idx*2+j), col+r] = U_blocks[i*2+j, r]
                                for j in range(2):
                                    for k in range(2):
                                        if V_blocks[j*2+k, r] != 0:
                                            V[(block_idx*2+j)*p + (sub_block*2+k), col+r] = V_blocks[j*2+k, r]
                                for i in range(3):
                                    for k in range(2):
                                        if W_blocks[k*3+i, r] != 0:
                                            W[(sub_block*2+k)*n + i, col+r] = W_blocks[k*3+i, r]
                            col += rank_block
                        else:
                            # 3x2x1 block - use schoolbook (rank 6)
                            rank_block = 6
                            for r in range(rank_block):
                                i = r // 2
                                j = (r % 2)
                                # U[i, j]
                                U[i*m + (block_idx*2+j), col+r] = 1
                                # V[j, 0] for the last column
                                V[(block_idx*2+j)*p + 6, col+r] = 1
                                # W[0, i] for column 6
                                W[6*n + i, col+r] = 1
                            col += rank_block
                
                # Verify
                if _verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
