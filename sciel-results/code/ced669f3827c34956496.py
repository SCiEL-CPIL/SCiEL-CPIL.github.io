import time
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 135.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 84
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Helper to generate Strassen 2x2x2 decomposition
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # U: A coefficients
        # M1: a11+a22
        U[0, 0] = 1.0; U[3, 0] = 1.0
        # M2: a21+a22
        U[2, 1] = 1.0; U[3, 1] = 1.0
        # M3: a11
        U[0, 2] = 1.0
        # M4: a22
        U[3, 3] = 1.0
        # M5: a11+a12
        U[0, 4] = 1.0; U[1, 4] = 1.0
        # M6: a21-a11
        U[2, 5] = 1.0; U[0, 5] = -1.0
        # M7: a12-a22
        U[1, 6] = 1.0; U[3, 6] = -1.0
        
        # V: B coefficients
        # M1: b11+b22
        V[0, 0] = 1.0; V[3, 0] = 1.0
        # M2: b11
        V[0, 1] = 1.0
        # M3: b12-b22
        V[1, 2] = 1.0; V[3, 2] = -1.0
        # M4: b21-b11
        V[2, 3] = 1.0; V[0, 3] = -1.0
        # M5: b22
        V[3, 4] = 1.0
        # M6: b11+b12
        V[0, 5] = 1.0; V[1, 5] = 1.0
        # M7: b21+b22
        V[2, 6] = 1.0; V[3, 6] = 1.0
        
        # W: C coefficients
        # C11 = M1 + M4 - M5 + M7
        W[0, 0] = 1.0; W[0, 3] = 1.0; W[0, 4] = -1.0; W[0, 6] = 1.0
        # C12 = M3 + M5
        W[1, 2] = 1.0; W[1, 4] = 1.0
        # C21 = M2 + M4
        W[2, 1] = 1.0; W[2, 3] = 1.0
        # C22 = M1 - M2 + M3 + M6
        W[3, 0] = 1.0; W[3, 1] = -1.0; W[3, 2] = 1.0; W[3, 5] = 1.0
        
        return U, V, W

    # Helper to generate M_{1,m,p} schoolbook
    def get_1mp(m, p):
        rank = m * p
        U = np.zeros((m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p, rank))
        for j in range(m):
            for k in range(p):
                r = j * p + k
                U[j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k, r] = 1.0
        return U, V, W

    # Helper to generate M_{2,2,1}
    def get_221():
        rank = 4
        U = np.zeros((4, rank))
        V = np.zeros((2, rank))
        W = np.zeros((2, rank))
        for i in range(2):
            for j in range(2):
                r = i * 2 + j
                U[i*2 + j, r] = 1.0
                V[j, r] = 1.0
                W[i, r] = 1.0
        return U, V, W

    # Construct M_{2,2,3} using M_{2,2,1} + M_{2,2,2}
    def get_223():
        U_221, V_221, W_221 = get_221()
        U_222, V_222, W_222 = get_strassen_222()
        
        rank = 4 + 7
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        U[:, 0:4] = U_221
        U[:, 4:11] = U_222
        
        # V rows: j=0,1 for k=0 (from 221), j=0,1 for k=1 (from 222), j=0,1 for k=2 (from 222)
        V[0:2, 0:4] = V_221
        V[2:4, 4:11] = V_222[0:2, :]
        V[4:6, 4:11] = V_222[2:4, :]
        
        # W rows: i=0,1 for k=0 (from 221), i=0,1 for k=1 (from 222), i=0,1 for k=2 (from 222)
        W[0:2, 0:4] = W_221
        W[2:4, 4:11] = W_222[0:2, :]
        W[4:6, 4:11] = W_222[2:4, :]
        
        return U, V, W

    # Construct M_{2,2,4} using M_{2,2,2} + M_{2,2,2}
    def get_224():
        U_222, V_222, W_222 = get_strassen_222()
        
        rank = 14
        U = np.zeros((4, rank))
        V = np.zeros((8, rank))
        W = np.zeros((8, rank))
        
        U[:, 0:7] = U_222
        U[:, 7:14] = U_222
        
        # V rows: j=0,1 for k=0,1 (first 222), j=0,1 for k=2,3 (second 222)
        V[0:2, 0:7] = V_222[0:2, :]
        V[2:4, 0:7] = V_222[2:4, :]
        V[4:6, 7:14] = V_222[0:2, :]
        V[6:8, 7:14] = V_222[2:4, :]
        
        # W rows: i=0,1 for k=0,1 (first 222), i=0,1 for k=2,3 (second 222)
        W[0:2, 0:7] = W_222[0:2, :]
        W[2:4, 0:7] = W_222[2:4, :]
        W[4:6, 7:14] = W_222[0:2, :]
        W[6:8, 7:14] = W_222[2:4, :]
        
        return U, V, W

    # Splits
    n_splits = [1, 2]
    m_splits = [2, 2]
    p_splits = [3, 4]
    
    # Get decompositions for each block type
    # Block (1, 2, 3)
    U_123, V_123, W_123 = get_1mp(2, 3)
    # Block (1, 2, 4)
    U_124, V_124, W_124 = get_1mp(2, 4)
    # Block (2, 2, 3)
    U_223, V_223, W_223 = get_223()
    # Block (2, 2, 4)
    U_224, V_224, W_224 = get_224()
    
    total_rank = 0
    blocks = []
    
    # Iterate through all combinations of splits
    for ni, nb_n in enumerate(n_splits):
        for mi, nb_m in enumerate(m_splits):
            for pi, nb_p in enumerate(p_splits):
                if nb_n == 1 and nb_m == 2 and nb_p == 3:
                    U_b, V_b, W_b = U_123, V_123, W_123
                elif nb_n == 1 and nb_m == 2 and nb_p == 4:
                    U_b, V_b, W_b = U_124, V_124, W_124
                elif nb_n == 2 and nb_m == 2 and nb_p == 3:
                    U_b, V_b, W_b = U_223, V_223, W_223
                elif nb_n == 2 and nb_m == 2 and nb_p == 4:
                    U_b, V_b, W_b = U_224, V_224, W_224
                else:
                    continue
                
                rank_b = U_b.shape[1]
                total_rank += rank_b
                blocks.append((ni, mi, pi, nb_n, nb_m, nb_p, U_b, V_b, W_b, rank_b))
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for (ni, mi, pi, nb_n, nb_m, nb_p, U_b, V_b, W_b, rank_b) in blocks:
        # Calculate global row offsets
        # i offset
        i_off = sum(n_splits[:ni])
        # j offset
        j_off = sum(m_splits[:mi])
        # k offset
        k_off = sum(p_splits[:pi])
        
        # U rows: i in [i_off, i_off+nb_n), j in [j_off, j_off+nb_m)
        # Global index: i*m + j
        u_rows = []
        for i in range(i_off, i_off + nb_n):
            for j in range(j_off, j_off + nb_m):
                u_rows.append(i*m + j)
        
        # V rows: j in [j_off, j_off+nb_m), k in [k_off, k_off+nb_p)
        # Global index: j*p + k
        v_rows = []
        for j in range(j_off, j_off + nb_m):
            for k in range(k_off, k_off + nb_p):
                v_rows.append(j*p + k)
        
        # W rows: k in [k_off, k_off+nb_p), i in [i_off, i_off+nb_n)
        # Global index: k*n + i
        w_rows = []
        for k in range(k_off, k_off + nb_p):
            for i in range(i_off, i_off + nb_n):
                w_rows.append(k*n + i)
        
        # Fill U
        for r in range(rank_b):
            for local_row, global_row in enumerate(u_rows):
                U_full[global_row, col_offset + r] = U_b[local_row, r]
        
        # Fill V
        for r in range(rank_b):
            for local_row, global_row in enumerate(v_rows):
                V_full[global_row, col_offset + r] = V_b[local_row, r]
        
        # Fill W
        for r in range(rank_b):
            for local_row, global_row in enumerate(w_rows):
                W_full[global_row, col_offset + r] = W_b[local_row, r]
        
        col_offset += rank_b
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid and total_rank < best_rank:
        best_rank = total_rank
        best_U = U_full.copy()
        best_V = V_full.copy()
        best_W = W_full.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    construct()
