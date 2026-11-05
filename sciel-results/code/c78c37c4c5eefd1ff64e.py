"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition with tolerance for floating point."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_true = _build_true_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_recon, T_true, atol=1e-8):
        return False
    return True

def _round_to_half_integer(arr):
    """Round array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _strassen_2x2x2():
    """
    Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists, each of shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)(e+f)
    # M2 = (c+d)(e)
    # M3 = (a)(f-b)
    # M4 = (d)(g-c)
    # M5 = (a+d)(h-g)
    # M6 = (b+c)(e+g)
    # M7 = (a-c)(f+h)
    # C = [[M1+M4-M5-M7, M3+M5],
    #      [M2+M7, M1-M2+M3+M6]]
    
    # U[i*2+j] for i in {0,1}, j in {0,1}
    # V[j*2+k] for j in {0,1}, k in {0,1}
    # W[k*2+i] for k in {0,1}, i in {0,1}
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+b)(e+f): U[0]=1,U[1]=1; V[0]=1,V[1]=1; W[0]=1,W[1]=1,W[2]=1,W[3]=1
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)(e): U[2]=1,U[3]=1; V[0]=1; W[0]=1,W[2]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1; W[2][1] = 1
    
    # M3 = (a)(f-b): U[0]=1; V[1]=1,V[0]=-1; W[1]=1,W[3]=1
    U[0][2] = 1
    V[1][2] = 1; V[0][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = (d)(g-c): U[3]=1; V[0]=1,V[2]=-1; W[0]=1,W[2]=1
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+d)(h-g): U[0]=1,U[3]=1; V[3]=1,V[2]=-1; W[1]=1,W[3]=1
    U[0][4] = 1; U[3][4] = 1
    V[3][4] = 1; V[2][4] = -1
    W[1][4] = 1; W[3][4] = 1
    
    # M6 = (b+c)(e+g): U[1]=1,U[2]=1; V[0]=1,V[2]=1; W[0]=1,W[1]=1
    U[1][5] = 1; U[2][5] = 1
    V[0][5] = 1; V[2][5] = 1
    W[0][5] = 1; W[1][5] = 1
    
    # M7 = (a-c)(f+h): U[0]=1,U[2]=-1; V[1]=1,V[3]=1; W[1]=1,W[3]=1
    U[0][6] = 1; U[2][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = 1
    
    return U, V, W

def _schoolbook_1x2x2():
    """
    Return schoolbook decomposition for 1x2x2 (rank 2).
    U shape (2, 2), V shape (4, 2), W shape (2, 2)
    """
    U = [[1, 0],
         [0, 1]]
    V = [[1, 0],
         [0, 0],
         [0, 1],
         [0, 0]]
    W = [[1, 0],
         [0, 1]]
    return U, V, W

def _structured_decomposition(n, m, p):
    """
    Build a structured decomposition for 3x4x8 using Strassen on 2x2x2 blocks.
    
    Strategy:
    - Split m=4 into two groups of 2: m_groups = [(0,1), (2,3)]
    - Split p=8 into four groups of 2: p_groups = [(0,1), (2,3), (4,5), (6,7)]
    - Split n=3 into: 2 rows + 1 row
    - For 2 A-rows x 2 B-rows x 2 C-cols: use Strassen (rank 7)
    - For 1 A-row x 2 B-rows x 2 C-cols: use schoolbook (rank 2)
    
    Total rank:
    - 2 A-rows: 1 x 2 B-groups x 4 C-groups x 7 = 56
    - 1 A-row: 1 x 2 B-groups x 4 C-groups x 2 = 16
    - Total: 72
    """
    n, m, p = 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n
    
    # Define groups
    # A rows: split into [0,1] and [2]
    a_row_groups = [(0, 1), (2,)]
    # B rows (middle dim): split into [0,1] and [2,3]
    b_row_groups = [(0, 1), (2, 3)]
    # C cols: split into four groups of 2
    c_col_groups = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    # Initialize matrices
    total_rank = 0
    for a_group in a_row_groups:
        for b_group in b_row_groups:
            for c_group in c_col_groups:
                if len(a_group) == 2 and len(b_group) == 2 and len(c_group) == 2:
                    total_rank += 7  # Strassen
                elif len(a_group) == 1 and len(b_group) == 2 and len(c_group) == 2:
                    total_rank += 2  # Schoolbook
                else:
                    total_rank += len(a_group) * len(b_group) * len(c_group)
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    col = 0
    for a_group in a_row_groups:
        for b_group in b_row_groups:
            for c_group in c_col_groups:
                a_indices = list(a_group)
                b_indices = list(b_group)
                c_indices = list(c_group)
                
                if len(a_group) == 2 and len(b_group) == 2 and len(c_group) == 2:
                    # Use Strassen 2x2x2
                    U_sub, V_sub, W_sub = _strassen_2x2x2()
                    for r in range(7):
                        for idx, i in enumerate(a_indices):
                            for j_idx, j in enumerate(b_indices):
                                U[i*m + j, col] = U_sub[i*2 + j_idx][r]
                        for j_idx, j in enumerate(b_indices):
                            for k_idx, k in enumerate(c_indices):
                                V[j*p + k, col] = V_sub[j_idx*2 + k_idx][r]
                        for k_idx, k in enumerate(c_indices):
                            for i_idx, i in enumerate(a_indices):
                                W[k*n + i, col] = W_sub[k_idx*2 + i_idx][r]
                        col += 1
                elif len(a_group) == 1 and len(b_group) == 2 and len(c_group) == 2:
                    # Use schoolbook 1x2x2 (rank 2)
                    i = a_indices[0]
                    for r in range(2):
                        for j_idx, j in enumerate(b_indices):
                            U[i*m + j, col] = 1 if j_idx == r else 0
                        for j_idx, j in enumerate(b_indices):
                            for k_idx, k in enumerate(c_indices):
                                V[j*p + k, col] = (1 if j_idx == r else 0) * (1 if k_idx == r else 0)
                        for k_idx, k in enumerate(c_indices):
                            W[k*n + i, col] = 1 if k_idx == r else 0
                        col += 1
                else:
                    # Fallback: schoolbook
                    for i in a_indices:
                        for j in b_indices:
                            for k in c_indices:
                                U[i*m + j, col] = 1
                                V[j*p + k, col] = 1
                                W[k*n + i, col] = 1
                                col += 1
    
    # Verify and return
    if col != total_rank:
        raise ValueError(f"Column mismatch: {col} vs {total_rank}")
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if not _verify_exact(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook if verification fails
        return _schoolbook(n, m, p)
    
    return {"U": U_list, "V": V_list, "W": W_list}

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

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 
    multiplying a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Use structured decomposition with Strassen blocks
    result = _structured_decomposition(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
