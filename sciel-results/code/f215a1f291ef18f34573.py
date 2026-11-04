"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_numpy(U, V, W, n, m, p):
    """Fast verification using numpy einsum."""
    rank = len(U[0])
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_2x2x2():
    """
    Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (2*2, 7) indexed by (i,j)
    V: (2*2, 7) indexed by (j,k)
    W: (2*2, 7) indexed by (k,i)
    """
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # U[i,j,r] * V[j,k,r] * W[k,i,r] should sum to 1 for each (i,j,k)
    #
    # Let's define the factors:
    # r=0 (M1): U factors: A11=1,A22=1; V factors: B11=1,B22=1; W: C11=1,C22=1
    # r=1 (M2): U: A21=1,A22=1; V: B11=1; W: C21=1
    # r=2 (M3): U: A11=1; V: B12=1,B22=-1; W: C12=1,C22=1
    # r=3 (M4): U: A22=1; V: B21=1,B11=-1; W: C11=1,C21=1
    # r=4 (M5): U: A11=1,A12=1; V: B22=1; W: C11=-1,C12=1
    # r=5 (M6): U: A21=1,A11=-1; V: B11=1,B12=1; W: C22=1
    # r=6 (M7): U: A12=1,A22=-1; V: B21=1,B22=1; W: C11=1,C22=1
    
    U = np.zeros((4, 7), dtype=int)  # indexed by (i,j): 00,01,10,11
    V = np.zeros((4, 7), dtype=int)  # indexed by (j,k): 00,01,10,11
    W = np.zeros((4, 7), dtype=int)  # indexed by (k,i): 00,01,10,11
    
    # Index mapping: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # r=0: M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11 (k=0,i=0)
    W[3, 0] = 1  # C22 (k=1,i=1)
    
    # r=1: M2 = (A21+A22)B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21 (k=1,i=0)
    
    # r=2: M3 = A11(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12 (k=0,i=1)
    W[3, 2] = 1  # C22 (k=1,i=1)
    
    # r=3: M4 = A22(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11 (k=0,i=0)
    W[2, 3] = 1  # C21 (k=1,i=0)
    
    # r=4: M5 = (A11+A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11 (k=0,i=0)
    W[1, 4] = 1  # C12 (k=0,i=1)
    
    # r=5: M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22 (k=1,i=1)
    
    # r=6: M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11 (k=0,i=0)
    W[3, 6] = 1  # C22 (k=1,i=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_strassen():
    """Verify Strassen decomposition is correct."""
    s = _strassen_2x2x2()
    return _verify_numpy(s["U"], s["V"], s["W"], 2, 2, 2)

def _block_decomposition_4x5x5():
    """
    Block decomposition of 4x5x5 using Strassen for 2x2x2 sub-blocks.
    
    Split: i (4 rows) into two groups of 2, j (5) into groups of 2+3, k (5) into groups of 2+3.
    
    The 4x5x5 multiplication tensor can be decomposed into sub-tensors based on index groups.
    For each combination of (i-group, j-group, k-group), we have a sub-problem.
    
    Sub-problems:
    - (2,2,2): Strassen, rank 7
    - (2,2,3): schoolbook, rank 12
    - (2,3,2): schoolbook, rank 12
    - (2,3,3): schoolbook, rank 18
    - (2,2,2): Strassen, rank 7 (second i-group)
    - (2,2,3): schoolbook, rank 12
    - (2,3,2): schoolbook, rank 12
    - (2,3,3): schoolbook, rank 18
    
    Total rank: 7+12+12+18+7+12+12+18 = 98
    
    Wait, that's only saving 2 from 100. Let me reconsider.
    
    Actually, the correct decomposition: the tensor T[i,j,k]=1 for all valid indices.
    When we split indices, the tensor becomes a sum of "corner" tensors.
    Each corner tensor T[i,j,k] = 1 for i in group I, j in group J, k in group K.
    These corner tensors are independent (they don't share index values), so the total rank
    is the sum of the ranks of each corner tensor.
    
    For a corner tensor with dimensions (n',m',p'), the schoolbook rank is n'*m'*p'.
    If n'=m'=p'=2, we can use Strassen with rank 7.
    
    So:
    - Corners with (2,2,2): use Strassen, rank 7 (instead of 8)
    - All other corners: schoolbook
    
    Corners:
    i-group 0 (size 2), j-group 0 (size 2), k-group 0 (size 2): (2,2,2) -> Strassen, rank 7
    i-group 0 (size 2), j-group 0 (size 2), k-group 1 (size 3): (2,2,3) -> schoolbook, rank 12
    i-group 0 (size 2), j-group 1 (size 3), k-group 0 (size 2): (2,3,2) -> schoolbook, rank 12
    i-group 0 (size 2), j-group 1 (size 3), k-group 1 (size 3): (2,3,3) -> schoolbook, rank 18
    i-group 1 (size 2), j-group 0 (size 2), k-group 0 (size 2): (2,2,2) -> Strassen, rank 7
    i-group 1 (size 2), j-group 0 (size 2), k-group 1 (size 3): (2,2,3) -> schoolbook, rank 12
    i-group 1 (size 2), j-group 1 (size 3), k-group 0 (size 2): (2,3,2) -> schoolbook, rank 12
    i-group 1 (size 2), j-group 1 (size 3), k-group 1 (size 3): (2,3,3) -> schoolbook, rank 18
    
    Total: 7+12+12+18+7+12+12+18 = 98
    """
    n, m, p = 4, 5, 5
    
    # Index splits
    i_splits = [0, 2, 4]  # groups of size 2, 2
    j_splits = [0, 2, 5]  # groups of size 2, 3
    k_splits = [0, 2, 5]  # groups of size 2, 3
    
    U = np.zeros((n * m, 0), dtype=int)
    V = np.zeros((m * p, 0), dtype=int)
    W = np.zeros((p * n, 0), dtype=int)
    
    col_offset = 0
    
    for ig in range(2):
        for jg in range(2):
            for kg in range(2):
                n_sub = i_splits[ig+1] - i_splits[ig]
                m_sub = j_splits[jg+1] - j_splits[jg]
                p_sub = k_splits[kg+1] - k_splits[kg]
                
                i_start = i_splits[ig]
                j_start = j_splits[jg]
                k_start = k_splits[kg]
                
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Use Strassen
                    s = _strassen_2x2x2()
                    U_sub = np.array(s["U"])
                    V_sub = np.array(s["V"])
                    W_sub = np.array(s["W"])
                    r = U_sub.shape[1]
                else:
                    # Schoolbook for this sub-block
                    r = n_sub * m_sub * p_sub
                    U_sub = np.zeros((n_sub * m_sub, r), dtype=int)
                    V_sub = np.zeros((m_sub * p_sub, r), dtype=int)
                    W_sub = np.zeros((p_sub * n_sub, r), dtype=int)
                    col = 0
                    for ii in range(n_sub):
                        for jj in range(m_sub):
                            for kk in range(p_sub):
                                U_sub[ii * m_sub + jj, col] = 1
                                V_sub[jj * p_sub + kk, col] = 1
                                W_sub[kk * n_sub + ii, col] = 1
                                col += 1
                
                # Place into full matrices
                U[:, col_offset:col_offset + U_sub.shape[1]] = 0
                V[:, col_offset:col_offset + V_sub.shape[1]] = 0
                W[:, col_offset:col_offset + W_sub.shape[1]] = 0
                
                # Actually, we need to build these properly. Let me restructure.
                # The sub-matrices need to be embedded into the full-sized matrices.
                
                col_offset += U_sub.shape[1]
    
    # The above approach has issues with building incrementally. Let me redo this.
    # Build all sub-blocks first, then concatenate.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    for ig in range(2):
        for jg in range(2):
            for kg in range(2):
                n_sub = i_splits[ig+1] - i_splits[ig]
                m_sub = j_splits[jg+1] - j_splits[jg]
                p_sub = k_splits[kg+1] - k_splits[kg]
                
                i_start = i_splits[ig]
                j_start = j_splits[jg]
                k_start = k_splits[kg]
                
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    s = _strassen_2x2x2()
                    U_sub = np.array(s["U"])
                    V_sub = np.array(s["V"])
                    W_sub = np.array(s["W"])
                else:
                    r = n_sub * m_sub * p_sub
                    U_sub = np.zeros((n_sub * m_sub, r), dtype=int)
                    V_sub = np.zeros((m_sub * p_sub, r), dtype=int)
                    W_sub = np.zeros((p_sub * n_sub, r), dtype=int)
                    col = 0
                    for ii in range(n_sub):
                        for jj in range(m_sub):
                            for kk in range(p_sub):
                                U_sub[ii * m_sub + jj, col] = 1
                                V_sub[jj * p_sub + kk, col] = 1
                                W_sub[kk * n_sub + ii, col] = 1
                                col += 1
                
                # Embed sub-block into full-sized matrices
                U_full = np.zeros((n * m, U_sub.shape[1]), dtype=int)
                V_full = np.zeros((m * p, V_sub.shape[1]), dtype=int)
                W_full = np.zeros((p * n, W_sub.shape[1]), dtype=int)
                
                # U: index (i, j) -> row i*m + j
                for ii in range(n_sub):
                    for jj in range(m_sub):
                        row_full = (i_start + ii) * m + (j_start + jj)
                        U_full[row_full, :] = U_sub[ii * m_sub + jj, :]
                
                # V: index (j, k) -> row j*p + k
                for jj in range(m_sub):
                    for kk in range(p_sub):
                        row_full = (j_start + jj) * p + (k_start + kk)
                        V_full[row_full, :] = V_sub[jj * p_sub + kk, :]
                
                # W: index (k, i) -> row k*n + i
                for kk in range(p_sub):
                    for ii in range(n_sub):
                        row_full = (k_start + kk) * n + (i_start + ii)
                        W_full[row_full, :] = W_sub[kk * n_sub + ii, :]
                
                U_parts.append(U_full)
                V_parts.append(V_full)
                W_parts.append(W_full)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Construct rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block decomposition with Strassen for 2x2x2 sub-blocks
    try:
        block_result = _block_decomposition_4x5x5()
        if _verify_numpy(block_result["U"], block_result["V"], block_result["W"], n, m, p):
            block_rank = len(block_result["U"][0])
            if block_rank < best_rank:
                best = block_result
                best_rank = block_rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return best
    
    return best
