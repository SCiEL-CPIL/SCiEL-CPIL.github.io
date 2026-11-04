"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank 84 decomposition using Strassen for 2x2x2 blocks.
    # Split n=3 into 2, 1
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate total rank
    # For each block (ni, mj, pk):
    # If ni=2, mj=2, pk=2: Rank 7
    # If ni=2, mj=1, pk=2: Rank 4 (Schoolbook 2*1*2)
    # If ni=1, mj=2, pk=2: Rank 4 (Schoolbook 1*2*2)
    # If ni=1, mj=1, pk=2: Rank 2 (Schoolbook 1*1*2)
    
    rank = 0
    for ni in n_splits:
        for mj in m_splits:
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    rank += 7
                else:
                    rank += ni * mj * pk
                    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    n_offset = 0
    for ni in n_splits:
        m_offset = 0
        for mj in m_splits:
            p_offset = 0
            for pk in p_splits:
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen 2x2x2
                    # Indices in U: row i*m+j. i in [0, ni), j in [0, mj)
                    # Indices in V: row j*p+k. j in [0, mj), k in [0, pk)
                    # Indices in W: row k*n+i. k in [0, pk), i in [0, ni)
                    
                    # Strassen terms P1..P7
                    # P1 = (A11+A22)(B11+B22)
                    # A11 is U[0,0] relative to block, A22 is U[1,1]
                    # B11 is V[0,0], B22 is V[1,1]
                    # W11 gets P1, W22 gets P1
                    
                    # We need to map global indices to local block indices.
                    # U row: (n_offset + i) * m + (m_offset + j)
                    # V row: (m_offset + j) * p + (p_offset + k)
                    # W row: (p_offset + k) * n + (n_offset + i)
                    
                    def add_strassen(col_offset):
                        nonlocal col
                        # P1
                        U[n_offset * m + m_offset][col] = 1
                        U[(n_offset + 1) * m + (m_offset + 1)][col] = 1
                        V[m_offset * p + p_offset][col] = 1
                        V[(m_offset + 1) * p + (p_offset + 1)][col] = 1
                        W[p_offset * n + n_offset][col] = 1
                        W[(p_offset + 1) * n + (n_offset + 1)][col] = 1
                        col += 1
                        
                        # P2 = (A21+A22)B11
                        U[(n_offset + 1) * m + m_offset][col] = 1
                        U[(n_offset + 1) * m + (m_offset + 1)][col] = 1
                        V[m_offset * p + p_offset][col] = 1
                        W[p_offset * n + (n_offset + 1)][col] = 1
                        W[(p_offset + 1) * n + (n_offset + 1)][col] = 1
                        col += 1
                        
                        # P3 = A11(B12-B22)
                        U[n_offset * m + m_offset][col] = 1
                        V[m_offset * p + (p_offset + 1)][col] = 1
                        V[(m_offset + 1) * p + (p_offset + 1)][col] = -1
                        W[(p_offset + 1) * n + n_offset][col] = 1
                        W[(p_offset + 1) * n + (n_offset + 1)][col] = 1
                        col += 1
                        
                        # P4 = A22(B21-B11)
                        U[(n_offset + 1) * m + (m_offset + 1)][col] = 1
                        V[(m_offset + 1) * p + p_offset][col] = 1
                        V[m_offset * p + p_offset][col] = -1
                        W[p_offset * n + (n_offset + 1)][col] = 1
                        W[(p_offset + 1) * n + (n_offset + 1)][col] = -1
                        col += 1
                        
                        # P5 = (A11+A12)B22
                        U[n_offset * m + m_offset][col] = 1
                        U[n_offset * m + (m_offset + 1)][col] = 1
                        V[(m_offset + 1) * p + (p_offset + 1)][col] = 1
                        W[(p_offset + 1) * n + n_offset][col] = 1
                        W[(p_offset + 1) * n + (n_offset + 1)][col] = -1
                        col += 1
                        
                        # P6 = (A21-A11)(B11+B12)
                        U[(n_offset + 1) * m + m_offset][col] = 1
                        U[n_offset * m + m_offset][col] = -1
                        V[m_offset * p + p_offset][col] = 1
                        V[m_offset * p + (p_offset + 1)][col] = 1
                        W[p_offset * n + (n_offset + 1)][col] = 1
                        W[(p_offset + 1) * n + (n_offset + 1)][col] = 1
                        col += 1
                        
                        # P7 = (A12-A22)(B21+B22)
                        U[n_offset * m + (m_offset + 1)][col] = 1
                        U[(n_offset + 1) * m + (m_offset + 1)][col] = -1
                        V[(m_offset + 1) * p + p_offset][col] = 1
                        V[(m_offset + 1) * p + (p_offset + 1)][col] = 1
                        W[p_offset * n + n_offset][col] = 1
                        W[(p_offset + 1) * n + n_offset][col] = -1
                        col += 1
                        
                    add_strassen(col)
                else:
                    # Schoolbook
                    for i in range(ni):
                        for j in range(mj):
                            for k in range(pk):
                                u_idx = (n_offset + i) * m + (m_offset + j)
                                v_idx = (m_offset + j) * p + (p_offset + k)
                                w_idx = (p_offset + k) * n + (n_offset + i)
                                U[u_idx][col] = 1
                                V[v_idx][col] = 1
                                W[w_idx][col] = 1
                                col += 1
                                
                p_offset += pk
            m_offset += mj
        n_offset += ni
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
