"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2 x 2x2 matrix multiplication using Strassen.
    Local indices:
      U: row = i*2 + j, i,j in [0,1]
      V: row = j*2 + k, j,k in [0,1]
      W: row = k*2 + i, k,i in [0,1]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][3] = 1
    
    # M2 = (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00 * (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    """Validate the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Partition: n=[2], m=[2,2,1], p=[2,2,1]
    n_parts = [2]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Compute offsets
    n_offsets = [0]
    for i in range(1, len(n_parts)):
        n_offsets.append(n_offsets[-1] + n_parts[i-1])
    m_offsets = [0]
    for i in range(1, len(m_parts)):
        m_offsets.append(m_offsets[-1] + m_parts[i-1])
    p_offsets = [0]
    for i in range(1, len(p_parts)):
        p_offsets.append(p_offsets[-1] + p_parts[i-1])
    
    U_entries = []  # (global_row, value, col)
    V_entries = []
    W_entries = []
    col_offset = 0
    
    for block_idx, (ni, nj, nk) in enumerate(zip(n_parts, m_parts, p_parts)):
        n_off = n_offsets[block_idx]
        m_off = m_offsets[block_idx]
        p_off = p_offsets[block_idx]
        
        if ni == 2 and nj == 2 and nk == 2:
            # Use Strassen
            Us, Vs, Ws = _strassen_2x2x2()
            for i_loc in range(2):
                for j_loc in range(2):
                    u_row = i_loc * 2 + j_loc
                    for c in range(7):
                        if Us[u_row][c] != 0:
                            global_row = (n_off + i_loc) * m + (m_off + j_loc)
                            U_entries.append((global_row, Us[u_row][c], col_offset + c))
            for j_loc in range(2):
                for k_loc in range(2):
                    v_row = j_loc * 2 + k_loc
                    for c in range(7):
                        if Vs[v_row][c] != 0:
                            global_row = (m_off + j_loc) * p + (p_off + k_loc)
                            V_entries.append((global_row, Vs[v_row][c], col_offset + c))
            for k_loc in range(2):
                for i_loc in range(2):
                    w_row = k_loc * 2 + i_loc
                    for c in range(7):
                        if Ws[w_row][c] != 0:
                            global_row = (p_off + k_loc) * n + (n_off + i_loc)
                            W_entries.append((global_row, Ws[w_row][c], col_offset + c))
            col_offset += 7
        else:
            # Schoolbook
            for i_loc in range(ni):
                for j_loc in range(nj):
                    for k_loc in range(nk):
                        u_row = (n_off + i_loc) * m + (m_off + j_loc)
                        v_row = (m_off + j_loc) * p + (p_off + k_loc)
                        w_row = (p_off + k_loc) * n + (n_off + i_loc)
                        U_entries.append((u_row, 1, col_offset))
                        V_entries.append((v_row, 1, col_offset))
                        W_entries.append((w_row, 1, col_offset))
                        col_offset += 1
    
    total_rank = col_offset
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for row, val, col in U_entries:
        U[row][col] = val
    for row, val, col in V_entries:
        V[row][col] = val
    for row, val, col in W_entries:
        W[row][col] = val
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
