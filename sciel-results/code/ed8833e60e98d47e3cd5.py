"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    """Build target tensor T[i, j, k] for n x m * m x p matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                T[i1 * m + j1, j1 * p + k2, k2 * n + i1] = 1.0
    return T

def _build_schoolbook(n, m, p):
    """Build schoolbook rank-nmp decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i1 in range(n):
        for j1 in range(m):
            for k2 in range(p):
                U[i1 * m + j1, col] = 1.0
                V[j1 * p + k2, col] = 1.0
                W[k2 * n + i1, col] = 1.0
                col += 1
    return U, V, W, rank

def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Verify that U,V,W exactly decompose tensor T."""
    T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(T_check - T)) < tol

def _is_half_integer(x, tol=1e-9):
    """Check if value is integer or half-integer within tolerance."""
    r = x * 2
    return np.all(np.abs(r - np.round(r)) < tol)

def _wft_2x2_10(n=2, p=10):
    """
    Apply Winograd Fourier Transform Algorithm to 2x2 x 2xp multiplication.
    
    The 2x2x(2p) tensor can be decomposed using cyclic group structure.
    For 2x2x10: we split the 10 columns into 5 groups of 2.
    Each 2x2x2 block has rank 7 (Strassen) or we can use WFTA.
    
    Actually, let's think more carefully. The 2x2x10 multiplication:
    A is 2x2, B is 2x10, C is 2x10.
    
    We can view this as 5 independent 2x2x2 multiplications (each column pair).
    But that gives rank 5*7 = 35 with Strassen.
    
    WFTA approach: Use the fact that for cyclic convolution of length 2,
    we can use the DFT to diagonalize. For length-2 cyclic convolution:
    [a0, a1] * [b0, b1] = [a0*b0 + a1*b1, a0*b1 + a1*b0]
    
    Using WFTA: transform A and B using DFT_2 = [[1,1],[1,-1]], multiply pointwise, inverse transform.
    This requires 3 multiplications (one for each frequency bin, but DC and Nyquist are special).
    
    Actually for length-2: DFT gives 2 frequency bins. Multiplication in frequency domain
    requires 2 multiplications. But we need 3 for the time-domain result because
    the inverse DFT mixes them.
    
    Let me reconsider. The standard result: 2x2x2 matrix mult has rank 7 (Strassen),
    not reducible further over reals.
    
    However, for the 2x2x10 problem, we can use a different approach.
    Think of it as: C = A * B where A is 2x2, B is 2x10.
    
    C[:, 2j] = A[:,0]*B[:,0] + A[:,1]*B[:,2j]  ... wait no.
    
    Let me re-index. A[i, j] for i,j in {0,1}. B[j, k] for j in {0,1}, k in {0,...,9}.
    C[i, k] = sum_j A[i,j] * B[j,k]
    
    So C[i, 2j] = A[i,0]*B[0,2j] + A[i,1]*B[1,2j]
    C[i, 2j+1] = A[i,0]*B[0,2j+1] + A[i,1]*B[1,2j+1]
    
    For each pair of output columns (2j, 2j+1), we have a 2x2x2 multiplication
    with inputs A (2x2), B_pair (2x2), outputs C_pair (2x2).
    
    So 2x2x10 = 5 copies of 2x2x2. Rank = 5 * 7 = 35 with Strassen.
    
    Can we do better? The total tensor is a direct sum of 5 independent 2x2x2 tensors.
    The rank of a direct sum is the sum of ranks (over reals, for these tensors).
    So minimum rank is 5 * 7 = 35.
    
    Wait, but the directive says "reduce its rank from the Strassen-derived 35 to 20".
    That would require rank-4 decompositions of 2x2x2, which don't exist over reals.
    
    Let me re-read the directive: "apply the Winograd Fourier Transform Algorithm (WFTA) to the 
    2x2x10 component using its cyclic group structure to reduce its rank from the Strassen-derived 
    35 to 20, yielding a total rank of 40."
    
    Hmm, 20 for 2x2x10 would mean average rank 4 per 2x2x2 block, which is impossible.
    
    Unless the decomposition couples the 5 blocks in a clever way. Let me think...
    
    Actually, maybe the idea is different. Perhaps we don't split into 5 independent 2x2x2 blocks,
    but use a global WFTA structure.
    
    Alternative: Think of 2x2x10 as related to polynomial multiplication.
    A(x) = A[0,0] + A[0,1]*x (row 0), etc.
    
    Actually, let me try a different decomposition. Consider:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    
    We can write this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    For each i (2 values) and k (10 values), that's 20 outputs.
    The inputs are: A has 4 entries, B has 20 entries. Total 24 inputs.
    
    Lower bound on rank: max(nm, mp, pn) = max(4, 20, 20) = 20.
    Upper bound: schoolbook = 4*20/2... no, schoolbook for 2x2x10 is 2*2*10 = 40.
    
    Wait, I think I was confusing myself. Let me redo.
    
    For the full 2x3x10 problem:
    - U shape: (n*m, rank) = (6, rank)
    - V shape: (m*p, rank) = (30, rank)  
    - W shape: (p*n, rank) = (20, rank)
    
    The 1x1x10 sub-problem (row 2 of A, which is 1x3, times B which is 3x10):
    Actually, let me think of it differently.
    
    A is 2x3, B is 3x10.
    Split A into A_top (2x2) and A_bottom (2x1) [rows 0-1, col 0-1 and row 0-1, col 2]
    Split B into B_top (2x10) and B_bottom (1x10) [rows 0-1, and row 2]
    
    C = A_top * B_top + A_bottom * B_bottom
    
    A_top * B_top: 2x2 * 2x10 -> 2x10, rank 35 (5 * Strassen-7)
    A_bottom * B_bottom: 2x1 * 1x10 -> 2x10, rank 20 (schoolbook: 2*1*10=20)
    
    Total: 35 + 20 = 55. Same as before.
    
    To get to 40, we'd need A_top * B_top to have rank 20, i.e., 4 per 2x2x2 block.
    That's not possible with independent blocks.
    
    BUT: what if we use a shared structure? The WFTA idea might be to process all 5 blocks
    simultaneously using a global transform that shares some multiplications.
    
    Let me try: use the DFT-2 on the "block index" dimension.
    Block index d in {0,1,2,3,4}. For each d, we have:
    C[:, 2d:2d+2] = A * B[:, 2d:2d+2]
    
    If we apply a length-5 DFT on the block index... but 5 is prime, and the structure
    isn't obviously cyclic in a way that helps.
    
    I think the directive's claim of rank 20 for 2x2x10 may be incorrect or I'm missing
    a key insight. Let me just implement the best I can: rank 55 via Strassen, and try
    to find improvements via optimization.
    """
    pass

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_r U[i,r]*V[j,r]*W[k,r] = delta(i+k, j) for 2x2*2x2->2x2.
    
    Indexing: i = (i1, i2), j = (j1, j2), k = (k1, k2)
    T[i,j,k] = 1 if i2 == j1 and k1 == i1, i.e., T[(i1,i2),(j1,j2),(k1,k2)] = delta(i2,j1)*delta(k1,i1)
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B11)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M4
    C10 = M2 + M5
    C11 = M1 - M2 + M3 + M6
    """
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    U = np.zeros((4, 7))  # A entries
    V = np.zeros((4, 7))  # B entries
    W = np.zeros((4, 7))  # C entries
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10 += M2
    W[3, 1] = -1  # C11 -= M2
    
    # M3 = A00*(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C01 += M3
    W[3, 2] = 1  # C11 += M3
    
    # M4 = A11*(B01-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = -1  # C00 -= M4
    W[1, 3] = 1  # C01 += M4
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1  # C00 += M5
    W[2, 4] = 1  # C10 += M5
    
    # M6 = (A10-A00)*(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (A01-A11)*(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W

def _build_strassen55(n=2, m=3, p=10):
    """
    Build rank-55 decomposition for 2x3x10 matrix multiplication.
    
    Split: A (2x3) = [A_top (2x2) | A_bottom (2x1)]
           B (3x10) = [B_top (2x10); B_bottom (1x10)]
    
    C = A_top * B_top + A_bottom * B_bottom
    
    A_top * B_top: 5 independent 2x2x2 blocks, each rank 7 -> rank 35
    A_bottom * B_bottom: 2x1x10, rank 20 (schoolbook)
    
    Total: 55
    """
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20
    rank = 55
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen 2x2x2 building block
    S_U, S_V, S_W = _strassen_2x2()
    
    # Part 1: A_top (2x2) * B_top (2x10) -> contributes to C (2x10)
    # A_top indices: A[i, j] for i in {0,1}, j in {0,1} -> U index i*m + j = i*3 + j
    # B_top indices: B[j, k] for j in {0,1}, k in {0,...,9} -> V index j*p + k = j*10 + k
    # C indices: C[i, k] for i in {0,1}, k in {0,...,9} -> W index k*n + i = k*2 + i
    
    for block in range(5):  # 5 blocks of 2 columns each
        col_offset = block * 7
        for r in range(7):
            # U: A_top entries (i, j) with i in {0,1}, j in {0,1}
            # U index = i * 3 + j
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, col_offset + r] = S_U[i * 2 + j, r]
            
            # V: B_top entries (j, k) with j in {0,1}, k in {2*block, 2*block+1}
            # V index = j * 10 + k
            for j in range(2):
                for dk in range(2):
                    k = 2 * block + dk
                    V[j * 10 + k, col_offset + r] = S_V[j * 2 + dk, r]
            
            # W: C entries (i, k) with i in {0,1}, k in {2*block, 2*block+1}
            # W index = k * 2 + i
            for i in range(2):
                for dk in range(2):
                    k = 2 * block + dk
                    W[k * 2 + i, col_offset + r] = S_W[i * 2 + dk, r]
    
    # Part 2: A_bottom (2x1) * B_bottom (1x10) -> contributes to C (2x10)
    # A_bottom: A[i, 2] for i in {0,1} -> U index i*3 + 2
    # B_bottom: B[2, k] for k in {0,...,9} -> V index 2*10 + k = 20 + k
    # C: C[i, k] -> W index k*2 + i
    # Schoolbook: rank 2*1*10 = 20
    
    col_offset = 35
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, col_offset + i * 10 + k] = 1.0
            V[20 + k, col_offset + i * 10 + k] = 1.0
            W[k * 2 + i, col_offset + i * 10 + k] = 1.0
    
    return U, V, W, rank

def _random_search_rank(T, nm, mp, pn, r, n_iter=50, tol=1e-8, deadline=None):
    """Try to find rank-r decomposition via alternating least squares."""
    for seed in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        np.random.seed(seed)
        
        U_c = np.random.randn(nm, r) * 0.1
        V_c = np.random.randn(mp, r) * 0.1
        W_c = np.random.randn(pn, r) * 0.1
        
        best_err = np.inf
        for it in range(15):
            if deadline is not None and time.time() > deadline:
                break
            
            # Solve for U
            M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
            try:
                U_c = T.reshape(nm, mp * pn) @ np.linalg.pinv(M.T)
            except:
                break
            
            # Solve for V
            N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
            try:
                V_c = T.reshape(mp, nm * pn) @ np.linalg.pinv(N.T)
            except:
                break
            
            # Solve for W
            P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
            try:
                W_c = T.reshape(pn, nm * mp) @ np.linalg.pinv(P.T)
            except:
                break
            
            T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
            err = np.max(np.abs(T_check - T))
            best_err = min(best_err, err)
        
        if best_err < tol:
            # Try rounding to half-integers
            U_r = np.round(U_c * 2) / 2
            V_r = np.round(V_c * 2) / 2
            W_r = np.round(W_c * 2) / 2
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-10):
                return U_r, V_r, W_r
            # Also check if unrounded is close enough
            if _verify_decomposition(U_c, V_c, W_c, T, tol=1e-10):
                return U_c, V_c, W_c
    
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T = _build_target_tensor(n, m, p)
    
    # Start with schoolbook
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try Strassen-55
    U_s, V_s, W_s, rank_s = _build_strassen55(n, m, p)
    if _verify_decomposition(U_s, V_s, W_s, T, tol=1e-10):
        best_rank = rank_s
        best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    else:
        # Debug: check error
        err = np.max(np.abs(np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s) - T))
        print(f"Strassen-55 error: {err}")
    
    deadline = time.time() + 130  # safety margin
    
    # Try to find lower rank decompositions
    for r in range(best_rank - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break
        
        result = _random_search_rank(T, nm, mp, pn, r, n_iter=20, tol=1e-8, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = r
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
