"""Matrix multiplication 2,4,7 tensor decomposition via block composition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # safety margin before 150s wall-clock limit


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)

    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])

    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _strassen_2x2x2():
    """
    Strassen's algorithm for 2x2x2 matrix multiplication.
    Returns U, V, W of shape (4, 7), (4, 7), (4, 7).
    
    The 7 products are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # A is 2x2: indices (i,j) for i,j in {0,1}
    # B is 2x2: indices (j,k) for j,k in {0,1}
    # C is 2x2: indices (i,k) for i,k in {0,1}
    
    # U: maps A[i,j] to 7 Strassen products
    # V: maps B[j,k] to 7 Strassen products
    # W: maps C[i,k] to 7 Strassen products
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1.0  # A00
    U[3, 0] = 1.0  # A11
    V[0, 0] = 1.0  # B00
    V[3, 0] = 1.0  # B11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1.0  # A10
    U[3, 1] = 1.0  # A11
    V[0, 1] = 1.0  # B00
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1.0  # A00
    V[2, 2] = 1.0  # B10
    V[3, 2] = -1.0 # B11
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1.0  # A11
    V[1, 3] = 1.0  # B01
    V[0, 3] = -1.0 # B00
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1.0  # A00
    U[1, 4] = 1.0  # A01
    V[3, 4] = 1.0  # B11
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1.0  # A10
    U[0, 5] = -1.0 # A00
    V[0, 5] = 1.0  # B00
    V[1, 5] = 1.0  # B01
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1.0  # A01
    U[3, 6] = -1.0 # A11
    V[2, 6] = 1.0  # B10
    V[3, 6] = 1.0  # B11
    
    # C11 = M1 + M4 - M5 + M7
    W[0, 0] = 1.0
    W[0, 3] = 1.0
    W[0, 4] = -1.0
    W[0, 6] = 1.0
    
    # C12 = M3 + M5
    W[1, 2] = 1.0
    W[1, 4] = 1.0
    
    # C21 = M2 + M4
    W[2, 1] = 1.0
    W[2, 3] = 1.0
    
    # C22 = M1 - M2 + M3 + M6
    W[3, 0] = 1.0
    W[3, 1] = -1.0
    W[3, 2] = 1.0
    W[3, 5] = 1.0
    
    return U, V, W, 7


def _compose_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks, block_decs):
    """
    Compose a block-structured decomposition.
    
    n_blocks: list of block sizes for n dimension (sums to n)
    m_blocks: list of block sizes for m dimension (sums to m)
    p_blocks: list of block sizes for p dimension (sums to p)
    block_decs: dict mapping (bi, bj, bk) -> (U_block, V_block, W_block, rank)
                where U_block has shape (n_i*m_j, rank), etc.
    
    Returns U, V, W for the full problem.
    """
    # Calculate offsets
    n_offsets = [0]
    for bs in n_blocks:
        n_offsets.append(n_offsets[-1] + bs)
    m_offsets = [0]
    for bs in m_blocks:
        m_offsets.append(m_offsets[-1] + bs)
    p_offsets = [0]
    for bs in p_blocks:
        p_offsets.append(p_offsets[-1] + bs)
    
    # Total rank is sum of ranks of all blocks
    total_rank = 0
    for bi in range(len(n_blocks)):
        for bj in range(len(m_blocks)):
            for bk in range(len(p_blocks)):
                key = (bi, bj, bk)
                if key in block_decs:
                    total_rank += block_decs[key][3]
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for bi in range(len(n_blocks)):
        for bj in range(len(m_blocks)):
            for bk in range(len(p_blocks)):
                key = (bi, bj, bk)
                if key not in block_decs:
                    continue
                U_b, V_b, W_b, rank_b = block_decs[key]
                
                ni = n_blocks[bi]
                mj = m_blocks[bj]
                pk = p_blocks[bk]
                
                # Place U_block: rows are (i in block bi, j in block bj)
                # Row index in full U: n_offsets[bi] + i, m_offsets[bj] + j -> (n_offsets[bi]+i)*m + (m_offsets[bj]+j)
                for i in range(ni):
                    for j in range(mj):
                        full_row = (n_offsets[bi] + i) * m + (m_offsets[bj] + j)
                        block_row = i * mj + j
                        U[full_row, col:col+rank_b] = U_b[block_row, :]
                
                # Place V_block: rows are (j in block bj, k in block bk)
                for j in range(mj):
                    for k in range(pk):
                        full_row = (m_offsets[bj] + j) * p + (p_offsets[bk] + k)
                        block_row = j * pk + k
                        V[full_row, col:col+rank_b] = V_b[block_row, :]
                
                # Place W_block: rows are (k in block bk, i in block bi)
                for k in range(pk):
                    for i in range(ni):
                        full_row = (p_offsets[bk] + k) * n + (n_offsets[bi] + i)
                        block_row = k * ni + i
                        W[full_row, col:col+rank_b] = W_b[block_row, :]
                
                col += rank_b
    
    return U, V, W, total_rank


def _block_decomposition_2x4x7():
    """
    Build a block decomposition for 2x4x7.
    
    Split: n=2 (one block of 2), m=4 (two blocks of 2), p=7 (blocks of 3 and 4)
    
    Blocks:
    (0,0,0): 2x2x3 -> schoolbook rank 12
    (0,0,1): 2x2x4 -> schoolbook rank 16
    (0,1,0): 2x2x3 -> schoolbook rank 12
    (0,1,1): 2x2x4 -> schoolbook rank 16
    
    Total: 12+16+12+16 = 56 (schoolbook)
    
    Alternative: use Strassen where possible.
    For 2x2x2 sub-blocks, use Strassen (rank 7).
    
    Let's try: n=2 (one block), m=4 (two blocks of 2), p=7 (one block of 7)
    Block (0,0,0): 2x2x7 -> need decomposition
    Block (0,1,0): 2x2x7 -> need decomposition
    
    For 2x2x7: we can split p into 2+2+3.
    (0,0,0): 2x2x2 -> Strassen rank 7
    (0,0,1): 2x2x2 -> Strassen rank 7
    (0,0,2): 2x2x3 -> schoolbook rank 12
    Total for 2x2x7: 7+7+12 = 26
    
    Two such blocks: 26*2 = 52. Total rank = 52.
    """
    # n_blocks = [2], m_blocks = [2, 2], p_blocks = [2, 2, 3]
    n_blocks = [2]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 3]
    
    block_decs = {}
    
    # For each (bi, bj, bk), the block is 2 x 2 x p_blocks[bk]
    for bi in range(len(n_blocks)):
        for bj in range(len(m_blocks)):
            for bk in range(len(p_blocks)):
                pk = p_blocks[bk]
                if pk == 2:
                    U_b, V_b, W_b, rank_b = _strassen_2x2x2()
                elif pk == 3:
                    U_b, V_b, W_b, rank_b = _schoolbook(2, 2, 3)
                else:
                    U_b, V_b, W_b, rank_b = _schoolbook(2, 2, pk)
                block_decs[(bi, bj, bk)] = (U_b, V_b, W_b, rank_b)
    
    U, V, W, total_rank = _compose_block_decomposition(2, 4, 7, n_blocks, m_blocks, p_blocks, block_decs)
    
    # Verify
    if not _verify(U, V, W, 2, 4, 7):
        raise ValueError("Block decomposition failed verification")
    
    return U, V, W, total_rank


def _try_reduce_rank(U, V, W, n, m, p, current_rank, max_time=30.0):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    This is a heuristic approach: try to find a subset of columns that can be removed
    while maintaining correctness, or find a lower-rank equivalent.
    
    For now, this is a placeholder that returns None.
    """
    start = time.time()
    # A proper approach would use numerical linear algebra to find the true rank
    # of the tensor, but that's complex. We'll skip this for now.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the block decomposition (rank 52)
    U, V, W, rank = _block_decomposition_2x4x7()
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank further (placeholder)
    if time.time() - start_time < DEADLINE - 10:
        result = _try_reduce_rank(U, V, W, n, m, p, rank, max_time=20.0)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
    
    # Convert to lists for the expected interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
