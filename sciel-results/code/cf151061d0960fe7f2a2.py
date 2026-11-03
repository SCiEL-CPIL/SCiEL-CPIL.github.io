"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return (U, V, W) for 2x2x2 Strassen multiplication.
    
    Indices:
    U rows: i*m + j -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V rows: j*p + k -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W rows: k*n + i -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return (U, V, W) for schoolbook multiplication of n x m by m x p."""
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
    return U, V, W

def _embed_block(global_U, global_V, global_W, 
                 local_U, local_V, local_W,
                 i_offset, j_offset, k_offset,
                 n_block, m_block, p_block,
                 col_offset):
    """Embed a local block decomposition into global factors.
    
    local_U rows: (i_local, j_local) where i_local in [0, n_block), j_local in [0, m_block)
    local_V rows: (j_local, k_local) where j_local in [0, m_block), k_local in [0, p_block)
    local_W rows: (k_local, i_local) where k_local in [0, p_block), i_local in [0, n_block)
    
    Global U rows: (i, j) where i in [0, n), j in [0, m)
    Global V rows: (j, k) where j in [0, m), k in [0, p)
    Global W rows: (k, i) where k in [0, p), i in [0, n)
    """
    rank_local = len(local_U[0]) if local_U else 0
    
    for r in range(rank_local):
        col = col_offset + r
        
        # Embed U
        for i_local in range(n_block):
            for j_local in range(m_block):
                val = local_U[i_local * m_block + j_local][r]
                if val != 0:
                    global_U[(i_offset + i_local) * 5 + (j_offset + j_local)][col] += val
        
        # Embed V
        for j_local in range(m_block):
            for k_local in range(p_block):
                val = local_V[j_local * p_block + k_local][r]
                if val != 0:
                    global_V[(j_offset + j_local) * 5 + (k_offset + k_local)][col] += val
        
        # Embed W
        for k_local in range(p_block):
            for i_local in range(n_block):
                val = local_W[k_local * n_block + i_local][r]
                if val != 0:
                    global_W[(k_offset + k_local) * 3 + (i_offset + i_local)][col] += val
    
    return col_offset + rank_local

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Use Strassen for the 2x2x2 sub-blocks where possible.
    # Partition the 5x5 middle dimension into 2+2+1.
    # Partition the 3x5 A matrix rows into 2+1.
    # Partition the 5x5 B matrix cols into 2+2+1.
    
    # Block sizes:
    # A: rows 0-1 (2), row 2 (1); cols 0-1 (2), 2-3 (2), 4 (1)
    # B: rows 0-1 (2), 2-3 (2), 4 (1); cols 0-1 (2), 2-3 (2), 4 (1)
    # C: rows 0-1 (2), row 2 (1); cols 0-1 (2), 2-3 (2), 4 (1)
    
    # We will decompose each (n_block, m_block, p_block) subproblem.
    # If n_block=2, m_block=2, p_block=2, use Strassen (rank 7).
    # Otherwise, use schoolbook (rank n_block * m_block * p_block).
    
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 1]
    
    # Calculate total rank
    total_rank = 0
    for n_b in n_blocks:
        for m_b in m_blocks:
            for p_b in p_blocks:
                if n_b == 2 and m_b == 2 and p_b == 2:
                    total_rank += 7
                else:
                    total_rank += n_b * m_b * p_b
    
    # Initialize global factors
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    i_offset = 0
    for n_b in n_blocks:
        j_offset = 0
        for m_b in m_blocks:
            k_offset = 0
            for p_b in p_blocks:
                if n_b == 2 and m_b == 2 and p_b == 2:
                    local_U, local_V, local_W = _strassen_222()
                else:
                    local_U, local_V, local_W = _schoolbook_block(n_b, m_b, p_b)
                
                col_offset = _embed_block(U, V, W, 
                                          local_U, local_V, local_W,
                                          i_offset, j_offset, k_offset,
                                          n_b, m_b, p_b,
                                          col_offset)
                k_offset += p_b
            j_offset += m_b
        i_offset += n_b
    
    # Verification (optional, for debugging)
    # U_np = np.array(U)
    # V_np = np.array(V)
    # W_np = np.array(W)
    # T_reconstructed = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    # T_expected = np.zeros((n, m, p))
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             T_expected[i, j, k] = 1
    # if not np.allclose(T_reconstructed, T_expected):
    #     print("Verification failed!")
    #     # Find mismatch
    #     diff = T_reconstructed - T_expected
    #     mismatch = np.argwhere(diff != 0)
    #     print("Mismatches at:", mismatch)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
