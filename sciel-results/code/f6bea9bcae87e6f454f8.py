"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115  # 5s safety margin before 120s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[a, b, c] where
    T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    Indexing: a = i*m+j, b = j'*p+k, c = i'*n+k'
    """
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * n + k
                T[a, b, c] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def _construct_strassen_block_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    V index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    W index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (c+d)h
    U[2, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1  # C21
    W[3, 1] = 1  # C22
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12
    W[3, 2] = -1 # C22
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (a+b)e
    U[0, 4] = 1; U[1, 4] = 1
    V[0, 4] = 1
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = -1 # C22
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W


def _construct_2x4x6_strassen():
    """
    Construct a Strassen-based decomposition for 2x4 * 4x6 -> 2x6.
    
    Strategy: Split the 4x6 matrix B into two 2x6 blocks B_top (rows 0-1) and B_bot (rows 2-3).
    Split A into A_left (cols 0-1) and A_right (cols 2-3).
    C = A_left @ B_top + A_right @ B_bot
    
    Each term A_left @ B_top is 2x2 * 2x6. We can further split B_top into three 2x2 blocks.
    So C = sum over 6 blocks of (2x2 * 2x2).
    
    Using Strassen for each 2x2*2x2 gives 7 terms per block, total 42 terms.
    But we can do better: use the fact that A_left is shared across the 3 B_top blocks.
    
    Alternative: Use a 2x4 * 4x2 multiplication which has rank 12 (known optimal).
    Then 2x4 * 4x6 = three copies of 2x4 * 4x2, giving rank 36.
    
    Even better: We can use a 2x4 * 4x6 directly.
    The trivial rank is 2*4*6 = 48.
    
    Let's use the block approach with Strassen on 2x2 blocks:
    Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    Split B (4x6) into B1 (2x6, rows 0-1) and B2 (2x6, rows 2-3).
    C = A1*B1 + A2*B2.
    
    For A1*B1 (2x2 * 2x6): split B1 into three 2x2 blocks.
    Use Strassen: 3 * 7 = 21 terms.
    Similarly for A2*B2: 21 terms.
    Total: 42 terms.
    
    But wait - the W indices overlap! C is 2x6, and both A1*B1 and A2*B2 contribute to all of C.
    So the total rank is 42, not 21+21 with overlap issues.
    
    Actually, let me reconsider. The rank of 2x4 * 4x6 multiplication:
    - Trivial: 48
    - Using 2x2 Strassen blocks: 42
    - Known lower bound: at least 2*4*6/2 = 24 (very loose)
    
    Let me try a different approach: use the rank-12 decomposition for 2x4*4x2.
    Then 2x4*4x6 requires three such decompositions, but the B side can be shared differently.
    
    Actually, the cleanest approach:
    2x4 * 4x6: think of it as 6 independent 2x4*4x1 multiplications (each column of C).
    Each 2x4*4x1 has rank at most 4 (trivial), so total rank 24.
    
    Or: 2x4*4x6 = (2x4*4x2) x 3 blocks. If 2x4*4x2 has rank 12, total is 36.
    
    Let me just use the 42-term Strassen approach which is straightforward.
    """
    n, m, p = 2, 4, 6
    U_str, V_str, W_str = _construct_strassen_block_2x2x2()
    
    total_rank = 42
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # A is 2x4: indices (i, j) with i in {0,1}, j in {0,1,2,3}
    # A1: j in {0,1}, A2: j in {2,3}
    # B is 4x6: indices (j, k) with j in {0,1,2,3}, k in {0,...,5}
    # B1: j in {0,1}, B2: j in {2,3}
    # C is 2x6: indices (i, k) with i in {0,1}, k in {0,...,5}
    # W index: k*n + i = k*2 + i
    
    # For each of the 6 blocks (A1 or A2) x (B1_block or B2_block):
    # Block (a, b) where a in {0,1} selects A1/A2, b in {0,1,2} selects k-range
    # A1: a=0, A2: a=1
    # B blocks: b=0 -> k in {0,1}, b=1 -> k in {2,3}, b=2 -> k in {4,5}
    # For a=0 (A1), B rows are 0,1 (B1)
    # For a=1 (A2), B rows are 2,3 (B2)
    
    for a in range(2):  # 0: A1, 1: A2
        for b in range(3):  # 0: k=0,1; 1: k=2,3; 2: k=4,5
            base = (a * 3 + b) * 7
            for r in range(7):
                col = base + r
                
                # U: A element (i, j) where j = a*2 + j_local
                # U_str[i_local, r] where i_local in {0,1}
                for i in range(2):
                    for j_local in range(2):
                        j = a * 2 + j_local
                        U[i * m + j, col] += U_str[i, r]
                
                # V: B element (j, k) where j = a*2 + j_local, k = b*2 + k_local
                # V_str[j_local, r] where j_local in {0,1}
                for j_local in range(2):
                    j = a * 2 + j_local
                    for k_local in range(2):
                        k = b * 2 + k_local
                        V[j * p + k, col] += V_str[j_local, r]
                
                # W: C element (i, k) where k = b*2 + k_local
                # W_str[i, r] where i in {0,1}
                for i in range(2):
                    for k_local in range(2):
                        k = b * 2 + k_local
                        W[k * n + i, col] += W_str[i, r]
    
    return U, V, W


def _construct_schoolbook():
    """Trivial rank-48 decomposition."""
    n, m, p = 2, 4, 6
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
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition (rank 42)
    U, V, W = _construct_2x4x6_strassen()
    T = _build_tensor(n, m, p)
    
    if _check_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (rank 48)
    U, V, W = _construct_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
