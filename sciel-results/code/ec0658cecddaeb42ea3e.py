"""Combined approach: block Strassen (2x2x2) with schoolbook fallback for 3x5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7) indexed by [i*2+a]
    V: (4, 7) indexed by [b*2+k]
    W: (4, 7) indexed by [k*2+i]
    
    Constraint: for all i,a,b,k in {0,1}:
      sum_c W[k*2+i, c] * U[i*2+a, c] * V[b*2+k, c] = delta[a,b]
    """
    # Strassen formulas for C = A*B where A, B are 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 - M3 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M6
    
    # U indexed by [i*2+a], V by [b*2+k], W by [k*2+i]
    # A indices: A[i,a] -> row i*2+a in U
    # B indices: B[b,k] -> row b*2+k in V
    # C indices: C[i,k] -> row k*2+i in W (trace convention W[k*n+i])
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0 (M0): (A00+A11)(B00+B11)
    # U: A00=1 (idx 0), A11=1 (idx 3)
    U[0, 0] = 1; U[3, 0] = 1
    # V: B00=1 (idx 0), B11=1 (idx 3)
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00 += 1 (idx 0), C11 += 1 (idx 3)
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1 (M1): (A10+A11)B00
    # U: A10=1 (idx 2), A11=1 (idx 3)
    U[2, 1] = 1; U[3, 1] = 1
    # V: B00=1 (idx 0)
    V[0, 1] = 1
    # W: C10 += 1 (idx k=0,i=1 -> 0*2+1=1... wait)
    # W indexed by [k*2+i]: C10 means i=1,k=0 -> idx 0*2+1 = 1
    W[1, 1] = 1
    
    # c=2 (M2): A00(B01-B11)
    # U: A00=1 (idx 0)
    U[0, 2] = 1
    # V: B01=1 (idx 1), B11=-1 (idx 3)
    V[1, 2] = 1; V[3, 2] = -1
    # W: C01 += 1 (i=0,k=1 -> idx 1*2+0=2), C11 -= 1 (i=1,k=1 -> idx 1*2+1=3)
    W[2, 2] = 1; W[3, 2] = -1
    
    # c=3 (M3): A11(B10-B00)
    # U: A11=1 (idx 3)
    U[3, 3] = 1
    # V: B10=1 (idx 2), B00=-1 (idx 0)
    V[2, 3] = 1; V[0, 3] = -1
    # W: C10 += 1 (i=1,k=0 -> idx 1), C00 -= 1 (i=0,k=0 -> idx 0)
    W[1, 3] = 1; W[0, 3] = -1
    
    # c=4 (M4): (A00+A01)B11
    # U: A00=1 (idx 0), A01=1 (idx 1)
    U[0, 4] = 1; U[1, 4] = 1
    # V: B11=1 (idx 3)
    V[3, 4] = 1
    # W: C01 += 1 (i=0,k=1 -> idx 2)
    W[2, 4] = 1
    
    # c=5 (M5): (A10-A00)(B00+B01)
    # U: A10=1 (idx 2), A00=-1 (idx 0)
    U[2, 5] = 1; U[0, 5] = -1
    # V: B00=1 (idx 0), B01=1 (idx 1)
    V[0, 5] = 1; V[1, 5] = 1
    # W: C00 += 1 (i=0,k=0 -> idx 0)
    W[0, 5] = 1
    
    # c=6 (M6): (A01-A11)(B10+B11)
    # U: A01=1 (idx 1), A11=-1 (idx 3)
    U[1, 6] = 1; U[3, 6] = -1
    # V: B10=1 (idx 2), B11=1 (idx 3)
    V[2, 6] = 1; V[3, 6] = 1
    # W: C11 += 1 (i=1,k=1 -> idx 3)
    W[3, 6] = 1
    
    return U, V, W


def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """
    Check that the factor matrices define the matrix multiplication tensor.
    Constraint: for all i in [0,n), a in [0,m), b in [0,m), k in [0,p):
      sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = (1 if a==b else 0)
    """
    rank = len(U[0])
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _try_block_decomposition(n, m, p):
    """
    Construct a decomposition using block Strassen for 2x2x2 sub-blocks
    and schoolbook for remaining blocks.
    
    Partition:
    - I (n=3): [2, 1]
    - J (m=5): [2, 2, 1]
    - K (p=7): [2, 2, 2, 1]
    
    For each (I_block, J_block, K_block):
    - If all sizes are 2: use Strassen (rank 7)
    - Otherwise: use schoolbook (rank = size_I * size_J * size_K)
    """
    # Block partitions: list of (start, size)
    I_blocks = [(0, 2), (2, 1)]
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    # Precompute Strassen factors
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Calculate total rank
    total_rank = 0
    block_info = []  # (i_start, i_size, j_start, j_size, k_start, k_size, is_strassen)
    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                is_strassen = (i_size == 2 and j_size == 2 and k_size == 2)
                if is_strassen:
                    total_rank += 7
                else:
                    total_rank += i_size * j_size * k_size
                block_info.append((i_start, i_size, j_start, j_size, k_start, k_size, is_strassen))
    
    # Build U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (i_start, i_size, j_start, j_size, k_start, k_size, is_strassen) in block_info:
        if is_strassen:
            # Place Strassen factors
            for local_i in range(2):
                for local_a in range(2):
                    gi = i_start + local_i
                    ga = j_start + local_a
                    u_idx = gi * m + ga
                    for c in range(7):
                        U[u_idx][col + c] = S_U[local_i * 2 + local_a][c]
            for local_b in range(2):
                for local_k in range(2):
                    gb = j_start + local_b
                    gk = k_start + local_k
                    v_idx = gb * p + gk
                    for c in range(7):
                        V[v_idx][col + c] = S_V[local_b * 2 + local_k][c]
            for local_k in range(2):
                for local_i in range(2):
                    gk = k_start + local_k
                    gi = i_start + local_i
                    w_idx = gk * n + gi
                    for c in range(7):
                        W[w_idx][col + c] = S_W[local_k * 2 + local_i][c]
            col += 7
        else:
            # Schoolbook for this block
            for li in range(i_size):
                for la in range(j_size):
                    for lk in range(k_size):
                        gi = i_start + li
                        ga = j_start + la
                        gk = k_start + lk
                        u_idx = gi * m + ga
                        v_idx = ga * p + gk
                        w_idx = gk * n + gi
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
    
    # Validate
    if not _validate_decomposition(U, V, W, n, m, p):
        return None
    
    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x7 matrix.
    
    Returns: {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = DIMENSIONS
    
    # Try block Strassen decomposition first
    result = _try_block_decomposition(n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
