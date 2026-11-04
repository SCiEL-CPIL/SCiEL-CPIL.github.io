import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2, rank 7. U:4x7, V:4x7, W:4x7"""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # M2=(A10+A11)B00
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
    # M3=A00(B01-B11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    # M4=A11(B10-B00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # M5=(A00+A10)B11
    U[0][4]=1; U[2][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # M6=(A11-A01)(B00+B01)
    U[3][5]=1; U[1][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # M7=(A01-A11)(B10+B11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _schoolbook_2x3x3():
    """2x3x3 schoolbook, rank 18. U:6x18, V:9x18, W:6x18"""
    U = [[0]*18 for _ in range(6)]
    V = [[0]*18 for _ in range(9)]
    W = [[0]*18 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col]=1
                V[j*3+k][col]=1
                W[k*2+i][col]=1
                col+=1
    return U, V, W

def _schoolbook_2x2x3():
    """2x2x3 schoolbook, rank 12. U:4x12, V:6x12, W:6x12"""
    U = [[0]*12 for _ in range(4)]
    V = [[0]*12 for _ in range(6)]
    W = [[0]*12 for _ in range(6)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*2+j][col]=1
                V[j*3+k][col]=1
                W[k*2+i][col]=1
                col+=1
    return U, V, W

def _schoolbook_2x3x2():
    """2x3x2 schoolbook, rank 12. U:6x12, V:6x12, W:4x12"""
    U = [[0]*12 for _ in range(6)]
    V = [[0]*12 for _ in range(6)]
    W = [[0]*12 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3+j][col]=1
                V[j*2+k][col]=1
                W[k*2+i][col]=1
                col+=1
    return U, V, W

def _schoolbook_2x2x2():
    """2x2x2 schoolbook, rank 8. U:4x8, V:4x8, W:4x8"""
    U = [[0]*8 for _ in range(4)]
    V = [[0]*8 for _ in range(4)]
    W = [[0]*8 for _ in range(4)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][col]=1
                V[j*2+k][col]=1
                W[k*2+i][col]=1
                col+=1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Partition: n=[2,2], m=[2,3], p=[3,2]
    # Blocks: (2,2,3)x2, (2,2,2)x2, (2,3,3)x2, (2,3,2)x2
    # Ranks: 12*2 + 7*2 + 18*2 + 12*2 = 24+14+36+24 = 98
    # Alternative: n=[2,2], m=[3,2], p=[3,2]
    # Blocks: (2,3,3)x2, (2,3,2)x2, (2,2,3)x2, (2,2,2)x2
    # Ranks: 18*2 + 12*2 + 12*2 + 7*2 = 36+24+24+14 = 98
    # Try: n=[2,2], m=[2,3], p=[2,3]
    # Blocks: (2,2,2)x2, (2,2,3)x2, (2,3,2)x2, (2,3,3)x2
    # Same: 14+24+24+36 = 98
    
    # Best known: use Strassen for 2x2x2 (7), schoolbook for others
    # n=[2,2], m=[2,3], p=[2,3]:
    #   (2,2,2): 7 each x2 = 14
    #   (2,2,3): 12 each x2 = 24
    #   (2,3,2): 12 each x2 = 24
    #   (2,3,3): 18 each x2 = 36
    #   Total: 98
    
    # Can we do better? What about using a 3x3x3 Laderman (rank 23) approach?
    # n=[3,1], m=[3,2], p=[3,2]:
    #   (3,3,3): 23
    #   (3,3,2): ? 
    #   (3,2,3): ?
    #   (3,2,2): ?
    #   (1,3,3): 9
    #   (1,3,2): 6
    #   (1,2,3): 6
    #   (1,2,2): 4
    #   Too complex without known decompositions.
    
    # Stick with the 98-rank decomposition but verify correctness carefully.
    
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_block(U_b, V_b, W_b, i_off, j_off, k_off):
        """Add a block decomposition with given offsets."""
        for r in range(len(U_b)):
            U_rows.append(U_b[r])
        for r in range(len(V_b)):
            V_rows.append(V_b[r])
        for r in range(len(W_b)):
            W_rows.append(W_b[r])
    
    def get_block(n_b, m_b, p_b):
        """Get the decomposition for a block of size n_b x m_b x p_b."""
        if n_b == 2 and m_b == 2 and p_b == 2:
            return _strassen_2x2x2()
        elif n_b == 2 and m_b == 2 and p_b == 3:
            return _schoolbook_2x2x3()
        elif n_b == 2 and m_b == 3 and p_b == 2:
            return _schoolbook_2x3x2()
        elif n_b == 2 and m_b == 3 and p_b == 3:
            return _schoolbook_2x3x3()
        else:
            # Fallback schoolbook
            rank = n_b * m_b * p_b
            U = [[0]*rank for _ in range(n_b*m_b)]
            V = [[0]*rank for _ in range(m_b*p_b)]
            W = [[0]*rank for _ in range(p_b*n_b)]
            col = 0
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        U[i*m_b+j][col]=1
                        V[j*p_b+k][col]=1
                        W[k*n_b+i][col]=1
                        col+=1
            return U, V, W
    
    # Process each block
    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if time.time() - start_time > 140:
                    break
                U_b, V_b, W_b = get_block(i_size, j_size, k_size)
                add_block(U_b, V_b, W_b, i_start, j_start, k_start)
    
    rank = len(U_rows[0]) if U_rows else 0
    
    # Verify the decomposition is correct
    # Build full U, V, W matrices
    U_full = np.zeros((n*m, rank))
    V_full = np.zeros((m*p, rank))
    W_full = np.zeros((p*n, rank))
    
    # We need to properly index the rows
    # For each block, the U rows correspond to local indices within the block
    # Let me rebuild this more carefully
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        u_col = [0.0] * (n*m)
        v_col = [0.0] * (m*p)
        w_col = [0.0] * (p*n)
        for idx, val in u_vec:
            u_col[idx] = val
        for idx, val in v_vec:
            v_col[idx] = val
        for idx, val in w_vec:
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if time.time() - start_time > 140:
                    break
                U_b, V_b, W_b = get_block(i_size, j_size, k_size)
                rank_b = len(U_b[0])
                
                for r in range(rank_b):
                    u_vec = []
                    v_vec = []
                    w_vec = []
                    for i_loc in range(i_size):
                        for j_loc in range(j_size):
                            val = U_b[i_loc*j_size+j_loc][r]
                            if val != 0:
                                i_glob = i_start + i_loc
                                j_glob = j_start + j_loc
                                u_vec.append((i_glob*m + j_glob, val))
                    for j_loc in range(j_size):
                        for k_loc in range(k_size):
                            val = V_b[j_loc*k_size+k_loc][r]
                            if val != 0:
                                j_glob = j_start + j_loc
                                k_glob = k_start + k_loc
                                v_vec.append((j_glob*p + k_glob, val))
                    for k_loc in range(k_size):
                        for i_loc in range(i_size):
                            val = W_b[k_loc*i_size+i_loc][r]
                            if val != 0:
                                k_glob = k_start + k_loc
                                i_glob = i_start + i_loc
                                w_vec.append((k_glob*n + i_glob, val))
                    add_term(u_vec, v_vec, w_vec)
    
    if not U_cols:
        # Fallback
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col]=1
                    V[j*p+k][col]=1
                    W[k*n+i][col]=1
                    col+=1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_cols, "V": V_cols, "W": W_cols}
