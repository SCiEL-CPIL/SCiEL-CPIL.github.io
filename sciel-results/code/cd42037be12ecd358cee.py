"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Convention: A (2x2) * B (2x2) -> C (2x2).
    Tensor indices: A[i,j], B[j,k], C[k,i] where i,k in {0,1} (output), j in {0,1} (inner).
    
    The Strassen algorithm computes:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    We need to express this as a tensor decomposition.
    U has shape (n*m, rank) = (4, 7) where index is i*m+j = i*2+j
    V has shape (m*p, rank) = (4, 7) where index is j*p+k = j*2+k
    W has shape (p*n, rank) = (4, 7) where index is k*n+i = k*2+i
    
    For each of the 7 multiplications M_c, we need:
    U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    Let's build each M_c as a sum of rank-1 terms.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A11 + A22) * (B11 + B22)
    # A11 is at i=0,j=0; A22 is at i=1,j=1
    # B11 is at j=0,k=0; B22 is at j=1,k=1
    # (A11+A22)*(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # But this is a single "multiplication" in Strassen, so we treat it as one rank-7 term.
    # Actually, each M_c is ONE multiplication, so it's a single rank-1 term in the decomposition.
    # The coefficients in U, V, W capture the sums.
    
    # M1: (A11+A22)*(B11+B22)
    # U contributions: A11 (i=0,j=0) -> U[0,0]=1, A22 (i=1,j=1) -> U[3,0]=1
    # V contributions: B11 (j=0,k=0) -> V[0,0]=1, B22 (j=1,k=1) -> V[3,0]=1
    # W contributions: C11 gets +1, C22 gets +1
    # C11: k=0,i=0 -> W[0,0]=1; C22: k=1,i=1 -> W[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)*B11
    # A21: i=1,j=0 -> U[2,1]=1; A22: i=1,j=1 -> U[3,1]=1
    # B11: j=0,k=0 -> V[0,1]=1
    # C21: k=0,i=1 -> W[2,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A11*(B12-B22)
    # A11: i=0,j=0 -> U[0,2]=1
    # B12: j=0,k=1 -> V[1,2]=1; B22: j=1,k=1 -> V[3,2]=-1
    # C12: k=1,i=0 -> W[1,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4: A22*(B21-B11)
    # A22: i=1,j=1 -> U[3,3]=1
    # B21: j=1,k=0 -> V[2,3]=1; B11: j=0,k=0 -> V[0,3]=-1
    # C11: k=0,i=0 -> W[0,3]=1; C21: k=0,i=1 -> W[2,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A11+A12)*B22
    # A11: i=0,j=0 -> U[0,4]=1; A12: i=0,j=1 -> U[1,4]=1
    # B22: j=1,k=1 -> V[3,4]=1
    # C11: k=0,i=0 -> W[0,4]=-1; C12: k=1,i=0 -> W[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    # A21: i=1,j=0 -> U[2,5]=1; A11: i=0,j=0 -> U[0,5]=-1
    # B11: j=0,k=0 -> V[0,5]=1; B12: j=0,k=1 -> V[1,5]=1
    # C22: k=1,i=1 -> W[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    # A12: i=0,j=1 -> U[1,6]=1; A22: i=1,j=1 -> U[3,6]=-1
    # B21: j=1,k=0 -> V[2,6]=1; B22: j=1,k=1 -> V[3,6]=1
    # C11: k=0,i=0 -> W[0,6]=1; C22: k=1,i=1 -> W[3,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m times m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W


def _block_decompose(n, m, p, block_sizes, use_strassen_2x2):
    """
    Decompose n x m * m x p using block partition.
    block_sizes: list of (ni, mi, pi) for each block.
    """
    all_U = []
    all_V = []
    all_W = []
    
    # Track row/col offsets
    row_offsets = [0]  # for A (n x m)
    for ni, mi, pi in block_sizes:
        row_offsets.append(row_offsets[-1] + ni)
    col_offsets = [0]  # for B (m x p)
    for ni, mi, pi in block_sizes:
        col_offsets.append(col_offsets[-1] + pi)
    inner_offsets = [0]  # for m dimension
    for ni, mi, pi in block_sizes:
        inner_offsets.append(inner_offsets[-1] + mi)
    
    for idx, (ni, mi, pi) in enumerate(block_sizes):
        if use_strassen_2x2 and ni == 2 and mi == 2 and pi == 2:
            Ub, Vb, Wb = _strassen_2x2x2()
        else:
            Ub, Vb, Wb = _schoolbook_block(ni, mi, pi)
        
        # Embed into full matrices
        U_full = np.zeros((n*m, Ub.shape[1]))
        V_full = np.zeros((m*p, Vb.shape[1]))
        W_full = np.zeros((p*n, Wb.shape[1]))
        
        r0 = row_offsets[idx]
        r1 = row_offsets[idx+1]
        c0 = col_offsets[idx]
        c1 = col_offsets[idx+1]
        i0 = inner_offsets[idx]
        i1 = inner_offsets[idx+1]
        
        for r in range(Ub.shape[0]):
            local_i = r // mi
            local_j = r % mi
            global_i = r0 + local_i
            global_j = i0 + local_j
            U_full[global_i * m + global_j, :] = Ub[r, :]
        
        for c in range(Vb.shape[0]):
            local_j = c // pi
            local_k = c % pi
            global_j = i0 + local_j
            global_k = c0 + local_k
            V_full[global_j * p + global_k, :] = Vb[c, :]
        
        for w in range(Wb.shape[0]):
            local_k = w // ni
            local_i = w % ni
            global_k = c0 + local_k
            global_i = r0 + local_i
            W_full[global_k * n + global_i, :] = Wb[w, :]
        
        all_U.append(U_full)
        all_V.append(V_full)
        all_W.append(W_full)
    
    U = np.hstack(all_U)
    V = np.hstack(all_V)
    W = np.hstack(all_W)
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # Build the tensor T[k*n+i, i*m+j, j*p+k] and check
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give 1 if i,j,k are valid, else 0
    # Actually, the matmul tensor is: T_{i,j,k} = delta(i,j,k) where the product is sum_k A[i,k]*B[k,j]
    # In our convention: A is n x m, B is m x p, C is n x p
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # Tensor: for each (i, j, k), the contribution is A[i,k] * B[k,j] to C[i,j]
    # U index: i*m + k (row of A, col of A)
    # V index: k*p + j (row of B, col of B)
    # W index: j*n + i (row of C, col of C) -- wait, the convention says W[k*n+i, r]
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has index k*n+i where k is the column of C and i is the row of C? 
    # Actually for C (n x p), row is i (0..n-1), col is j (0..p-1).
    # The convention W[k*n+i] suggests k is the "outer" index and i is the "inner".
    # In matmul C[i,j] = sum_k A[i,k] B[k,j], the tensor has indices (i, j, k).
    # U[i*m+k], V[k*p+j], W[j*n+i] -- this makes sense: W's first index is j (col of C) * n + i (row of C).
    
    # Let's verify: sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] should equal 1 for all i,j,k
    # (since A[i,k]*B[k,j] contributes 1 to C[i,j] for each k)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0
                for c in range(rank):
                    val += U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c]
                if abs(val - 1) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the 2+2+1 block partition with Strassen on the 2x2x2 blocks
    # Block sizes: (2,2,2), (2,2,2), (1,1,1)
    # But we need to partition n=5, m=5, p=5 consistently.
    # n partition: 2+2+1
    # m partition: 2+2+1
    # p partition: 2+2+1
    
    block_sizes = [(2,2,2), (2,2,2), (1,1,1)]
    
    # Verify partition sums
    assert sum(b[0] for b in block_sizes) == n
    assert sum(b[1] for b in block_sizes) == m
    assert sum(b[2] for b in block_sizes) == p
    
    U, V, W = _block_decompose(n, m, p, block_sizes, use_strassen_2x2=True)
    
    rank = U.shape[1]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = _schoolbook_block(n, m, p)
        rank = U.shape[1]
    
    # Convert to lists of lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
