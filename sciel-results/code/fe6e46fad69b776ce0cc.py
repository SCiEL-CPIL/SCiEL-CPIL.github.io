import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank is sum of ranks of subproblems
    # Subproblems:
    # (0,0,0): 2x2x2 -> Strassen (rank 7)
    # (0,1,0): 2x3x2 -> Schoolbook (rank 12)
    # (0,0,1): 2x2x3 -> Schoolbook (rank 12)
    # (0,1,1): 2x3x3 -> Schoolbook (rank 18)
    # (1,0,0): 2x2x2 -> Strassen (rank 7)
    # (1,1,0): 2x3x2 -> Schoolbook (rank 12)
    # (1,0,1): 2x2x3 -> Schoolbook (rank 12)
    # (1,1,1): 2x3x3 -> Schoolbook (rank 18)
    
    # Total rank = 2*7 + 2*12 + 2*12 + 2*18 = 14 + 24 + 24 + 36 = 98
    rank = 98
    
    # Initialize global matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Define subproblem ranges
    # i ranges: [0,2), [2,4)
    # j ranges: [0,2), [2,5)
    # k ranges: [0,2), [2,5)
    
    i_ranges = [(0, 2), (2, 4)]
    j_ranges = [(0, 2), (2, 5)]
    k_ranges = [(0, 2), (2, 5)]
    
    for i_idx in range(2):
        i_start, i_end = i_ranges[i_idx]
        n_sub = i_end - i_start
        
        for j_idx in range(2):
            j_start, j_end = j_ranges[j_idx]
            m_sub = j_end - j_start
            
            for k_idx in range(2):
                k_start, k_end = k_ranges[k_idx]
                p_sub = k_end - k_start
                
                # Determine if this is a 2x2x2 subproblem
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Use Strassen
                    local_U, local_V, local_W, local_rank = _strassen_2x2x2()
                else:
                    # Use Schoolbook
                    local_U, local_V, local_W, local_rank = _schoolbook(n_sub, m_sub, p_sub)
                
                # Embed local matrices into global matrices
                for r in range(local_rank):
                    # U: row index is i_global * m + j_global
                    for i_local in range(n_sub):
                        for j_local in range(m_sub):
                            i_global = i_start + i_local
                            j_global = j_start + j_local
                            row_u = i_global * m + j_global
                            U[row_u][current_rank + r] += local_U[i_local * m_sub + j_local][r]
                    
                    # V: row index is j_global * p + k_global
                    for j_local in range(m_sub):
                        for k_local in range(p_sub):
                            j_global = j_start + j_local
                            k_global = k_start + k_local
                            row_v = j_global * p + k_global
                            V[row_v][current_rank + r] += local_V[j_local * p_sub + k_local][r]
                    
                    # W: row index is k_global * n + i_global
                    for k_local in range(p_sub):
                        for i_local in range(n_sub):
                            k_global = k_start + k_local
                            i_global = i_start + i_local
                            row_w = k_global * n + i_global
                            W[row_w][current_rank + r] += local_W[k_local * n_sub + i_local][r]
                
                current_rank += local_rank
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Returns local U, V, W matrices for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: A coefficients
    # V: B coefficients
    # W: C coefficients
    
    # M1
    U[0][0] = 1; U[3][0] = 1 # A00 + A11
    V[0][0] = 1; V[3][0] = 1 # B00 + B11
    W[0][0] = 1; W[3][0] = 1 # C00, C11
    
    # M2
    U[2][1] = 1; U[3][1] = 1 # A10 + A11
    V[0][1] = 1 # B00
    W[2][1] = 1; W[3][1] = -1 # C10, C11
    
    # M3
    U[0][2] = 1 # A00
    V[1][2] = 1; V[3][2] = -1 # B01 - B11
    W[1][2] = 1; W[3][2] = 1 # C01, C11
    
    # M4
    U[3][3] = 1 # A11
    V[2][3] = 1; V[0][3] = -1 # B10 - B00
    W[0][3] = 1; W[2][3] = 1 # C00, C10
    
    # M5
    U[0][4] = 1; U[1][4] = 1 # A00 + A01
    V[3][4] = 1 # B11
    W[0][4] = -1; W[1][4] = 1 # C00, C01
    
    # M6
    U[2][5] = 1; U[0][5] = -1 # A10 - A00
    V[0][5] = 1; V[1][5] = 1 # B00 + B01
    W[3][5] = 1 # C11
    
    # M7
    U[1][6] = 1; U[3][6] = -1 # A01 - A11
    V[2][6] = 1; V[3][6] = 1 # B10 + B11
    W[0][6] = 1 # C00
    
    return U, V, W, rank

def _schoolbook(n, m, p):
    """
    Returns local U, V, W matrices for schoolbook decomposition of nxm by mxp.
    """
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
    
    return U, V, W, rank
# EVOLVE-BLOCK-END
