import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, rank, seed=42, num_tests=20):
    """Verify that the decomposition exactly reproduces matrix multiplication (vectorized)."""
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randint(-100, 100, size=(n, m)).astype(np.int64)
        B = rng.randint(-100, 100, size=(m, p)).astype(np.int64)
        C_expected = A @ B
        
        # Vectorized: compute all scalar products at once
        ua = A.ravel() @ U  # shape (rank,)
        vb = B.ravel() @ V  # shape (rank,)
        Wc = W.T.reshape(rank, p, n).transpose(0, 2, 1)  # shape (rank, n, p)
        C_recomputed = np.einsum('c,c,cij->ij', ua, vb, Wc)
        
        if not np.array_equal(C_recomputed, C_expected):
            return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _strassen_222():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, c] corresponds to A[i,j], V[i*2+j, c] to B[i,j], W[i*2+j, c] to C[i,j]
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[1, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = A11(B21-B22)
    U[0, 2] = 1
    V[1, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = A22(B11-B21)
    U[3, 3] = 1
    V[0, 3] = 1
    V[1, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[1, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1
    U[3, 6] = -1
    V[1, 6] = 1
    V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_225():
    """
    Construct a decomposition for (2,2,5) matrix multiplication.
    Uses Strassen for paired columns and schoolbook for the last column.
    Returns U (4, rank), V (10, rank), W (10, rank) with rank <= 18.
    """
    strassen_U, strassen_V, strassen_W = _strassen_222()
    
    total_rank = 18
    U = np.zeros((4, total_rank), dtype=np.int64)
    V = np.zeros((10, total_rank), dtype=np.int64)
    W = np.zeros((10, total_rank), dtype=np.int64)
    
    col_offset = 0
    
    # Pair (col 0, col 1): use Strassen
    for c in range(7):
        U[:, col_offset + c] = strassen_U[:, c]
        for i in range(2):
            for j in range(2):
                sv = strassen_V[i*2+j, c]
                if sv != 0:
                    V[i*5+j, col_offset + c] += sv
        for i in range(2):
            for j in range(2):
                sw = strassen_W[i*2+j, c]
                if sw != 0:
                    W[i*5+j, col_offset + c] += sw
    col_offset += 7
    
    # Pair (col 2, col 3): use Strassen
    for c in range(7):
        U[:, col_offset + c] = strassen_U[:, c]
        for i in range(2):
            for j in range(2):
                sv = strassen_V[i*2+j, c]
                if sv != 0:
                    V[i*5+(2+j), col_offset + c] += sv
        for i in range(2):
            for j in range(2):
                sw = strassen_W[i*2+j, c]
                if sw != 0:
                    W[i*5+(2+j), col_offset + c] += sw
    col_offset += 7
    
    # Column 4: schoolbook 2x2x1, rank 4
    for i in range(2):
        for j in range(2):
            U[i*2+j, col_offset] = 1
            V[j*5+4, col_offset] = 1
            W[i*5+4, col_offset] = 1
            col_offset += 1
    
    return U, V, W, total_rank

def _construct_125():
    """
    Construct a decomposition for (1,2,5) matrix multiplication.
    A is 1x2, B is 2x5, C is 1x5.
    Rank is 10 (each output depends on 2 inputs).
    Returns U (2, 10), V (10, 10), W (5, 10).
    """
    rank = 10
    U = np.zeros((2, rank), dtype=np.int64)
    V = np.zeros((10, rank), dtype=np.int64)
    W = np.zeros((5, rank), dtype=np.int64)
    
    col = 0
    for k in range(5):
        U[0, col] = 1
        V[0*5+k, col] = 1
        W[k, col] = 1
        col += 1
        U[1, col] = 1
        V[1*5+k, col] = 1
        W[k, col] = 1
        col += 1
    
    return U, V, W, rank

def _construct_325():
    """
    Construct a decomposition for (3,2,5) matrix multiplication.
    A is 3x2, B is 2x5, C is 3x5.
    Split A into A_top (1x2) and A_bot (2x2).
    C = A_top @ B + A_bot @ B
    Returns U (6, rank), V (10, rank), W (15, rank).
    """
    U_125, V_125, W_125, rank_125 = _construct_125()
    U_225, V_225, W_225, rank_225 = _construct_225()
    
    total_rank = rank_125 + rank_225  # 10 + 18 = 28
    U = np.zeros((6, total_rank), dtype=np.int64)
    V = np.zeros((10, total_rank), dtype=np.int64)
    W = np.zeros((15, total_rank), dtype=np.int64)
    
    U[0:2, 0:rank_125] = U_125
    V[0:10, 0:rank_125] = V_125
    W[0:5, 0:rank_125] = W_125
    
    offset = rank_125
    U[2:6, offset:offset+rank_225] = U_225
    V[0:10, offset:offset+rank_225] = V_225
    W[5:15, offset:offset+rank_225] = W_225
    
    return U, V, W, total_rank

def _construct_345():
    """
    Construct a decomposition for (3,4,5) matrix multiplication.
    Split A (3x4) into A_left (3x2) and A_right (3x2).
    Split B (4x5) into B_top (2x5) and B_bot (2x5).
    C = A_left @ B_top + A_right @ B_bot
    Each part is (3,2,5).
    Returns U (12, rank), V (20, rank), W (15, rank).
    """
    U_325, V_325, W_325, rank_325 = _construct_325()
    
    total_rank = 2 * rank_325  # 2 * 28 = 56
    U = np.zeros((12, total_rank), dtype=np.int64)
    V = np.zeros((20, total_rank), dtype=np.int64)
    W = np.zeros((15, total_rank), dtype=np.int64)
    
    U[0:6, 0:rank_325] = U_325
    V[0:10, 0:rank_325] = V_325
    W[0:15, 0:rank_325] = W_325
    
    offset = rank_325
    U[6:12, offset:offset+rank_325] = U_325
    V[10:20, offset:offset+rank_325] = V_325
    W[0:15, offset:offset+rank_325] = W_325
    
    return U, V, W, total_rank

def _greedy_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by identifying columns that are linear combinations
    of others. For each pair of columns, check if one rank-1 tensor is a scalar
    multiple of the other (proportional in all three factors simultaneously).
    """
    rank = U.shape[1]
    U_f = U.astype(np.float64)
    V_f = V.astype(np.float64)
    W_f = W.astype(np.float64)
    
    removable = set()
    
    for c1 in range(rank):
        if c1 in removable:
            continue
        for c2 in range(c1 + 1, rank):
            if c2 in removable:
                continue
            u1, u2 = U_f[:, c1], U_f[:, c2]
            v1, v2 = V_f[:, c1], V_f[:, c2]
            w1, w2 = W_f[:, c1], W_f[:, c2]
            
            # Check if all three vectors are proportional
            nz_u1 = np.nonzero(u1)[0]
            nz_u2 = np.nonzero(u2)[0]
            if len(nz_u1) == 0 or len(nz_u2) == 0:
                continue
            if not np.array_equal(nz_u1, nz_u2):
                continue
            a = u1[nz_u1[0]] / u2[nz_u2[0]]
            if not np.allclose(u1[nz_u1], a * u2[nz_u2], atol=1e-10):
                continue
            
            nz_v1 = np.nonzero(v1)[0]
            nz_v2 = np.nonzero(v2)[0]
            if len(nz_v1) == 0 or len(nz_v2) == 0:
                continue
            if not np.array_equal(nz_v1, nz_v2):
                continue
            b = v1[nz_v1[0]] / v2[nz_v2[0]]
            if not np.allclose(v1[nz_v1], b * v2[nz_v2], atol=1e-10):
                continue
            
            nz_w1 = np.nonzero(w1)[0]
            nz_w2 = np.nonzero(w2)[0]
            if len(nz_w1) == 0 or len(nz_w2) == 0:
                continue
            if not np.array_equal(nz_w1, nz_w2):
                continue
            c = w1[nz_w1[0]] / w2[nz_w2[0]]
            if not np.allclose(w1[nz_w1], c * w2[nz_w2], atol=1e-10):
                continue
            
            # All three proportional: tensor_c1 = a*b*c * tensor_c2
            removable.add(c1)
            break
    
    if not removable:
        return U, V, W, rank
    
    keep = [i for i in range(rank) if i not in removable]
    U_new = U[:, keep]
    V_new = V[:, keep]
    W_new = W[:, keep]
    new_rank = len(keep)
    
    return U_new, V_new, W_new, new_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 4, 5
    
    try:
        U_best, V_best, W_best, rank_best = _construct_345()
        
        if time.time() < deadline:
            if _verify_decomposition(U_best, V_best, W_best, n, m, p, rank_best, seed=42, num_tests=30):
                # Try greedy rank reduction
                if time.time() < deadline - 10:
                    U_red, V_red, W_red, rank_red = _greedy_rank_reduction(
                        U_best, V_best, W_best, n, m, p, deadline
                    )
                    if rank_red < rank_best:
                        if _verify_decomposition(U_red, V_red, W_red, n, m, p, rank_red, seed=42, num_tests=30):
                            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
                
                return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    except Exception:
        pass
    
    U_best, V_best, W_best, rank_best = _schoolbook_decomposition(n, m, p)
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
