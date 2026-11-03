import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Standard Strassen algorithm for 2x2 matrix multiplication.
    Returns 7 triples (U_coeff, V_coeff, W_coeff) where each coefficient is a 2x2 matrix
    representing the bilinear form: M_c = sum U_c[i,j]*A[i,j] * V_c[i,j]*B[i,j]
    and C = sum W_c * M_c.
    """
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    U = []
    V = []
    W = []
    
    # M1 = (a+d)(e+h)
    u = np.array([[1,0],[1,0]])
    v = np.array([[1,0],[0,1]])
    w = np.array([[1,0],[0,1]])
    U.append(u); V.append(v); W.append(w)
    
    # M2 = (c+d)e
    u = np.array([[0,0],[1,1]])
    v = np.array([[1,0],[0,0]])
    w = np.array([[0,0],[1,0]])
    U.append(u); V.append(v); W.append(w)
    
    # M3 = a(f-h)
    u = np.array([[1,0],[0,0]])
    v = np.array([[0,1],[0,-1]])
    w = np.array([[0,1],[0,0]])
    U.append(u); V.append(v); W.append(w)
    
    # M4 = d(g-f)
    u = np.array([[0,0],[0,1]])
    v = np.array([[0,1],[0,-1]])
    w = np.array([[0,0],[1,1]])
    U.append(u); V.append(v); W.append(w)
    
    # M5 = (a+b)h
    u = np.array([[1,1],[0,0]])
    v = np.array([[0,0],[0,1]])
    w = np.array([[0,1],[0,0]])
    U.append(u); V.append(v); W.append(w)
    
    # M6 = (c-a)(e+f)
    u = np.array([[-1,0],[1,0]])
    v = np.array([[1,1],[0,0]])
    w = np.array([[1,-1],[0,0]])
    U.append(u); V.append(v); W.append(w)
    
    # M7 = (b-d)(g+h)
    u = np.array([[0,1],[0,-1]])
    v = np.array([[0,1],[1,1]])
    w = np.array([[0,0],[1,-1]])
    U.append(u); V.append(v); W.append(w)
    
    return U, V, W


def _verify_strassen():
    """Verify that Strassen decomposition is correct."""
    U, V, W = _strassen_2x2x2()
    for _ in range(100):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C_true = A @ B
        C_strassen = np.zeros((2, 2))
        for c in range(7):
            M_c = np.sum(U[c] * A) * np.sum(V[c] * B)
            C_strassen += W[c] * M_c
        if not np.allclose(C_true, C_strassen, atol=1e-10):
            return False
    return True


def _build_block_decomposition():
    """
    Build block decomposition:
    n=3 = 2+1, m=5 = 2+2+1, p=6 = 2+2+2
    
    Blocks of A: 3x5 -> split rows into [2,1], cols into [2,2,1] -> 6 blocks
    Blocks of B: 5x6 -> split rows into [2,2,1], cols into [2,2,2] -> 9 blocks
    
    For each pair (A_block, B_block) where A_block is (ni x mj) and B_block is (mj x pk),
    we need to compute their product contribution to C_block (ni x pk).
    
    Use Strassen for 2x2x2 blocks, schoolbook otherwise.
    """
    n, m, p = DIMENSIONS
    n_splits = [2, 1]  # row splits of A
    m_splits = [2, 2, 1]  # col splits of A = row splits of B
    p_splits = [2, 2, 2]  # col splits of B
    
    n_blocks = len(n_splits)
    m_blocks = len(m_splits)
    p_blocks = len(p_splits)
    
    U_all = []
    V_all = []
    W_all = []
    
    Strassen_OK = _verify_strassen()
    
    for ib in range(n_blocks):
        ni = n_splits[ib]
        for jb in range(m_blocks):
            mj = m_splits[jb]
            for kb in range(p_blocks):
                pk = p_splits[kb]
                
                # A_block: ni x mj, B_block: mj x pk -> C_block: ni x pk
                if ni == 2 and mj == 2 and pk == 2:
                    if Strassen_OK:
                        U_s, V_s, W_s = _strassen_2x2x2()
                        for c in range(7):
                            # U factor: ni*mj = 4 entries, each is a 2x2 matrix flattened
                            U_row = []
                            for i in range(ni):
                                for j in range(mj):
                                    U_row.append(U_s[c][i, j])
                            # V factor: mj*pk = 4 entries
                            V_row = []
                            for j in range(mj):
                                for k in range(pk):
                                    V_row.append(V_s[c][j, k])
                            # W factor: pk*ni = 4 entries (output index k*n+i)
                            W_row = []
                            for k in range(pk):
                                for i in range(ni):
                                    W_row.append(W_s[c][i, k])
                            U_all.append(U_row)
                            V_all.append(V_row)
                            W_all.append(W_row)
                    else:
                        # Schoolbook fallback: ni*mj*pk = 8 terms
                        for i in range(ni):
                            for j in range(mj):
                                for k in range(pk):
                                    U_row = [0.0] * (ni * mj)
                                    U_row[i * mj + j] = 1.0
                                    V_row = [0.0] * (mj * pk)
                                    V_row[j * pk + k] = 1.0
                                    W_row = [0.0] * (pk * ni)
                                    W_row[k * ni + i] = 1.0
                                    U_all.append(U_row)
                                    V_all.append(V_row)
                                    W_all.append(W_row)
                else:
                    # Schoolbook: ni*mj*pk terms
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                U_row = [0.0] * (ni * mj)
                                U_row[i * mj + j] = 1.0
                                V_row = [0.0] * (mj * pk)
                                V_row[j * pk + k] = 1.0
                                W_row = [0.0] * (pk * ni)
                                W_row[k * ni + i] = 1.0
                                U_all.append(U_row)
                                V_all.append(V_row)
                                W_all.append(W_row)
    
    # Now map local block indices to global indices
    # A is n x m, row ib offset, col jb offset
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    # Rebuild with global indexing
    U_global = {}
    V_global = {}
    W_global = {}
    
    rank = 0
    for ib in range(n_blocks):
        ni = n_splits[ib]
        for jb in range(m_blocks):
            mj = m_splits[jb]
            for kb in range(p_blocks):
                pk = p_splits[kb]
                
                if ni == 2 and mj == 2 and pk == 2 and Strassen_OK:
                    U_s, V_s, W_s = _strassen_2x2x2()
                    for c in range(7):
                        col = rank
                        rank += 1
                        U_g = [0.0] * (n * m)
                        V_g = [0.0] * (m * p)
                        W_g = [0.0] * (p * n)
                        
                        for i in range(ni):
                            for j in range(mj):
                                U_g[(n_offsets[ib] + i) * m + (m_offsets[jb] + j)] = U_s[c][i, j]
                        
                        for j in range(mj):
                            for k in range(pk):
                                V_g[(m_offsets[jb] + j) * p + (p_offsets[kb] + k)] = V_s[c][j, k]
                        
                        for k in range(pk):
                            for i in range(ni):
                                W_g[(p_offsets[kb] + k) * n + (n_offsets[ib] + i)] = W_s[c][i, k]
                        
                        U_global[col] = U_g
                        V_global[col] = V_g
                        W_global[col] = W_g
                else:
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                col = rank
                                rank += 1
                                U_g = [0.0] * (n * m)
                                V_g = [0.0] * (m * p)
                                W_g = [0.0] * (p * n)
                                
                                U_g[(n_offsets[ib] + i) * m + (m_offsets[jb] + j)] = 1.0
                                V_g[(m_offsets[jb] + j) * p + (p_offsets[kb] + k)] = 1.0
                                W_g[(p_offsets[kb] + k) * n + (n_offsets[ib] + i)] = 1.0
                                
                                U_global[col] = U_g
                                V_global[col] = V_g
                                W_global[col] = W_g
    
    U_mat = [U_global[i] for i in range(rank)]
    V_mat = [V_global[i] for i in range(rank)]
    W_mat = [W_global[i] for i in range(rank)]
    
    return U_mat, V_mat, W_mat


def _verify_decomposition(U, V, W, n=None, m=None, p=None):
    """Verify that the decomposition exactly computes matrix multiplication."""
    if n is None or m is None or p is None:
        n, m, p = DIMENSIONS
    rank = len(U)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(50):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        C_dec = np.zeros((n, p))
        for c in range(rank):
            # Extract factors
            U_c = U_arr[:, c].reshape(n, m)
            V_c = V_arr[:, c].reshape(m, p)
            W_c = W_arr[:, c].reshape(p, n)  # W[k*n+i, c]
            M_c = np.sum(U_c * A) * np.sum(V_c * B)
            # C += M_c * W_c transposed appropriately
            # W_c[k, i] corresponds to C[i, k]
            C_dec += M_c * W_c.T
        if not np.allclose(C_true, C_dec, atol=1e-9):
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build block decomposition
    U, V, W = _build_block_decomposition()
    
    # Verify
    if time.time() < deadline:
        if _verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    n, m, p = DIMENSIONS
    rank = n * m * p
    U_school = [[0] * rank for _ in range(n * m)]
    V_school = [[0] * rank for _ in range(m * p)]
    W_school = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    if time.time() < deadline:
        if _verify_decomposition(U_school, V_school, W_school):
            return {"U": U_school, "V": V_school, "W": W_school}
    
    # Last resort: return schoolbook even without verification
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
