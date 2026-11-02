"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_2x2x2():
    """
    Build the 7-term Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns list of 7 tuples (U_vec, V_vec, W_vec) where each vec is a list of 4 coeffs.
    
    Indexing convention:
    U index: i*2+j for row i, col j of A (A is 2x2)
    V index: j*2+k for row j, col k of B (B is 2x2)
    W index: k*2+i for col k, row i of C (C is 2x2, trace convention W[k*n+i])
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U indices: A00=0, A01=1, A10=2, A11=3
    # V indices: B00=0, B01=1, B10=2, B11=3
    # W indices: C[0,0]=W[0*2+0]=0, C[0,1]=W[1*2+0]=2, C[1,0]=W[0*2+1]=1, C[1,1]=W[1*2+1]=3
    
    # M1 = (A00+A11)(B00+B11): U has A00(0)=1, A11(3)=1; V has B00(0)=1, B11(3)=1
    # W: C00(0)=1, C11(3)=1
    M1 = ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])
    
    # M2 = (A10+A11)B00: U has A10(2)=1, A11(3)=1; V has B00(0)=1
    # W: C10(1)=1, C11(3)=-1
    M2 = ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1])
    
    # M3 = A00(B10-B11): U has A00(0)=1; V has B10(2)=1, B11(3)=-1
    # W: C01(2)=1
    M3 = ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0])
    
    # M4 = A11(B01-B00): U has A11(3)=1; V has B01(1)=1, B00(0)=-1
    # W: C00(0)=1, C10(1)=1
    M4 = ([0, 0, 0, 1], [0, -1, 0, 0], [1, 1, 0, 0])
    
    # M5 = (A00+A01)B11: U has A00(0)=1, A01(1)=1; V has B11(3)=1
    # W: C00(0)=-1, C01(2)=1
    M5 = ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0])
    
    # M6 = (A10-A00)(B00+B01): U has A10(2)=1, A00(0)=-1; V has B00(0)=1, B01(1)=1
    # W: C11(3)=1
    M6 = ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1])
    
    # M7 = (A01-A11)(B10+B11): U has A01(1)=1, A11(3)=-1; V has B10(2)=1, B11(3)=1
    # W: C00(0)=1
    M7 = ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    
    return [M1, M2, M3, M4, M5, M6, M7]

def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    terms = _build_strassen_2x2x2()
    T = np.zeros((4, 4, 4), dtype=float)
    for U, V, W in terms:
        T += np.einsum('i,j,k->ijk', np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float))
    
    # Target: T[i,j,k]=1 iff valid (A,B,C) triple
    # A idx: i=0->A00, i=1->A01, i=2->A10, i=3->A11
    # B idx: j=0->B00, j=1->B01, j=2->B10, j=3->B11
    # C idx: k=0->C00, k=1->C10, k=2->C01, k=3->C11
    # C00 = A00*B00 + A01*B10 -> (0,0,0), (1,2,0)
    # C01 = A00*B01 + A01*B11 -> (0,1,2), (1,3,2)
    # C10 = A10*B00 + A11*B10 -> (2,0,1), (3,2,1)
    # C11 = A10*B01 + A11*B11 -> (2,1,3), (3,3,3)
    target = np.zeros((4, 4, 4), dtype=float)
    for (i, j, k) in [(0,0,0),(1,2,0),(0,1,2),(1,3,2),(2,0,1),(3,2,1),(2,1,3),(3,3,3)]:
        target[i, j, k] = 1
    
    return np.allclose(T, target, atol=1e-10)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    return np.allclose(T_decomp, target, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Check if Strassen is valid
    use_strassen = False
    try:
        use_strassen = _verify_strassen()
    except Exception:
        use_strassen = False
    
    if not use_strassen:
        return _schoolbook(n, m, p)
    
    # Partition: n = 2+1, m = 2+2+1, p = 2+2+1
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Compute offsets
    n_offsets = []
    off = 0
    for sz in n_parts:
        n_offsets.append(off)
        off += sz
    m_offsets = []
    off = 0
    for sz in m_parts:
        m_offsets.append(off)
        off += sz
    p_offsets = []
    off = 0
    for sz in p_parts:
        p_offsets.append(off)
        off += sz
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    strassen_terms = _build_strassen_2x2x2()
    
    for ni, ni_sz in enumerate(n_parts):
        for mj, mj_sz in enumerate(m_parts):
            for pk, pk_sz in enumerate(p_parts):
                n_off = n_offsets[ni]
                m_off = m_offsets[mj]
                p_off = p_offsets[pk]
                
                if ni_sz == 2 and mj_sz == 2 and pk_sz == 2:
                    # Use Strassen for 2x2x2 block
                    for U_vec, V_vec, W_vec in strassen_terms:
                        for u_rel in range(4):
                            if U_vec[u_rel] != 0:
                                u_idx = (n_off + u_rel // 2) * m + (m_off + u_rel % 2)
                                U_entries.append((u_idx, U_vec[u_rel]))
                        for v_rel in range(4):
                            if V_vec[v_rel] != 0:
                                v_idx = (m_off + v_rel // 2) * p + (p_off + v_rel % 2)
                                V_entries.append((v_idx, V_vec[v_rel]))
                        for w_rel in range(4):
                            if W_vec[w_rel] != 0:
                                w_idx = (p_off + w_rel // 2) * n + (n_off + w_rel % 2)
                                W_entries.append((w_idx, W_vec[w_rel]))
                else:
                    # Schoolbook for this sub-block
                    for i in range(ni_sz):
                        for j in range(mj_sz):
                            for k in range(pk_sz):
                                u_idx = (n_off + i) * m + (m_off + j)
                                v_idx = (m_off + j) * p + (p_off + k)
                                w_idx = (p_off + k) * n + (n_off + i)
                                U_entries.append((u_idx, 1))
                                V_entries.append((v_idx, 1))
                                W_entries.append((w_idx, 1))
    
    rank = len(U_entries)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        u_idx, coeff = U_entries[col]
        U[u_idx][col] = coeff
        v_idx, coeff = V_entries[col]
        V[v_idx][col] = coeff
        w_idx, coeff = W_entries[col]
        W[w_idx][col] = coeff
    
    # Verify before returning
    try:
        if not _verify_decomposition(U, V, W, n, m, p):
            return _schoolbook(n, m, p)
    except Exception:
        return _schoolbook(n, m, p)
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
