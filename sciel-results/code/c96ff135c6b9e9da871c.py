"""Strassen-based decomposition for 2x4 * 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

def construct():
    """
    Decompose 2x4 * 4x8 matrix multiplication using block Strassen.
    
    We split the 4x4 middle dimension into two 2x2 blocks:
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x8) = [B1; B2] where B1, B2 are 2x8
    
    Then C = A1*B1 + A2*B2
    
    For each 2x2 * 2x8 multiplication, we split the 8 columns into four 2x2 blocks:
    2x8 = [C1 | C2 | C3 | C4] where each Ci is 2x2
    
    So we have 8 separate 2x2 * 2x2 multiplications.
    
    Each 2x2 * 2x2 uses Strassen with rank 7.
    Total rank = 8 * 7 = 56.
    
    Actually, let me reconsider the structure more carefully.
    
    A is 2x4, B is 4x8, C is 2x8.
    
    Split A into A_left (2x2) and A_right (2x2): A = [A_left | A_right]
    Split B into B_top (2x8) and B_bottom (2x8): B = [B_top; B_bottom]
    
    C = A_left * B_top + A_right * B_bottom
    
    Each product is 2x8. We can further split into 2x2 blocks:
    A_left * B_top = [A_left * B_top_0 | A_left * B_top_1 | A_left * B_top_2 | A_left * B_top_3]
    where each B_top_i is 2x2.
    
    So we have 8 separate 2x2*2x2 multiplications, each done via Strassen (rank 7).
    Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (32, rank)  
    # W has shape (p*n, rank) = (16, rank)
    
    # Strassen for 2x2 * 2x2:
    # Input X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Output Z = X*Y
    # Strassen factors:
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y01+y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    def strassen_decomposition():
        """Return (U_str, V_str, W_str) for one 2x2*2x2 Strassen decomposition.
        
        U maps the 4 input elements of X to the 7 intermediate products.
        V maps the 4 input elements of Y to the 7 intermediate products.
        W maps the 7 intermediate products to the 4 output elements of Z.
        
        U[i, c]: coefficient of X element i in M_c
        V[j, c]: coefficient of Y element j in M_c
        W[k, c]: coefficient of M_c in Z element k
        
        Indexing: X elements: 0=x00, 1=x01, 2=x10, 3=x11
                  Y elements: 0=y00, 1=y01, 2=y10, 3=y11
                  Z elements: 0=z00, 1=z01, 2=z10, 3=z11
        """
        U_str = [[0]*7 for _ in range(4)]
        V_str = [[0]*7 for _ in range(4)]
        W_str = [[0]*7 for _ in range(4)]
        
        # M1 = (x00+x11)*(y00+y11)
        U_str[0][0] = 1  # x00
        U_str[3][0] = 1  # x11
        V_str[0][0] = 1  # y00
        V_str[3][0] = 1  # y11
        
        # M2 = (x10+x11)*y00
        U_str[2][1] = 1  # x10
        U_str[3][1] = 1  # x11
        V_str[0][1] = 1  # y00
        
        # M3 = x00*(y01-y11)
        U_str[0][2] = 1  # x00
        V_str[1][2] = 1  # y01
        V_str[3][2] = -1  # y11
        
        # M4 = x11*(y10-y00)
        U_str[3][3] = 1  # x11
        V_str[2][3] = 1  # y10
        V_str[0][3] = -1  # y00
        
        # M5 = (x00+x01)*y11
        U_str[0][4] = 1  # x00
        U_str[1][4] = 1  # x01
        V_str[3][4] = 1  # y11
        
        # M6 = (x10-x00)*(y00+y10)
        U_str[2][5] = 1  # x10
        U_str[0][5] = -1  # x00
        V_str[0][5] = 1  # y00
        V_str[2][5] = 1  # y10
        
        # M7 = (x01-x11)*(y01+y11)
        U_str[1][6] = 1  # x01
        U_str[3][6] = -1  # x11
        V_str[1][6] = 1  # y01
        V_str[3][6] = 1  # y11
        
        # W: Z = sum_c W[k,c] * M_c
        # Z00 = M1 + M4 - M5 + M7
        W_str[0][0] = 1   # M1
        W_str[0][3] = 1   # M4
        W_str[0][4] = -1  # M5
        W_str[0][6] = 1   # M7
        
        # Z01 = M3 + M5
        W_str[1][2] = 1   # M3
        W_str[1][4] = 1   # M5
        
        # Z10 = M2 + M4
        W_str[2][1] = 1   # M2
        W_str[2][3] = 1   # M4
        
        # Z11 = M1 - M2 + M3 + M6
        W_str[3][0] = 1   # M1
        W_str[3][1] = -1  # M2
        W_str[3][2] = 1   # M3
        W_str[3][5] = 1   # M6
        
        return U_str, V_str, W_str
    
    U_str, V_str, W_str = strassen_decomposition()
    
    # Now build the full decomposition.
    # 
    # A (2x4) indexed as A[i][j] where i in {0,1}, j in {0,1,2,3}
    # B (4x8) indexed as B[j][k] where j in {0,1,2,3}, k in {0,...,7}
    # C (2x8) indexed as C[i][k] where i in {0,1}, k in {0,...,7}
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split: j=0,1 are "top", j=2,3 are "bottom"
    # A_top = A[:, 0:2] (2x2), A_bot = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x8), B_bot = B[2:4, :] (2x8)
    #
    # C = A_top * B_top + A_bot * B_bot
    #
    # Further split B_top into 4 blocks of 2x2: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    # Similarly for B_bot.
    #
    # So we have 8 separate 2x2*2x2 multiplications:
    # For s in {0,1} (top/bottom) and t in {0,1,2,3} (column block):
    #   X = A_s (2x2), Y = B_s[:, 2*t : 2*t+2] (2x2)
    #   Z_s,t = X * Y contributes to C[:, 2*t : 2*t+2]
    #
    # C[:, 2*t:2*t+2] = sum_s Z_s,t
    #
    # Total rank = 8 * 7 = 56
    
    rank = 8 * 7
    U = [[0] * rank for _ in range(8)]   # n*m = 8
    V = [[0] * rank for _ in range(32)]  # m*p = 32
    W = [[0] * rank for _ in range(16)]  # p*n = 16
    
    # Indexing for U: row = i*m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    #   A[i,j] corresponds to U row i*4+j
    #   For top (j=0,1): A_top[i, j] = A[i, j], so row = i*4 + j
    #   For bottom (j=2,3): A_bot[i, j-2] = A[i, j], so row = i*4 + j
    #
    # Indexing for V: row = j*p + k = j*8 + k, where j in {0,...,3}, k in {0,...,7}
    #   B[j,k] corresponds to V row j*8 + k
    #   For top (j=0,1): B_top[j, k] = B[j, k], row = j*8 + k
    #   For bottom (j=2,3): B_bot[j-2, k] = B[j, k], row = j*8 + k
    #
    # Indexing for W: row = k*n + i = k*2 + i, where k in {0,...,7}, i in {0,1}
    #   C[i,k] corresponds to W row k*2 + i
    #
    # For each (s, t) pair where s in {0,1} (top=0, bottom=1) and t in {0,1,2,3}:
    #   base_col = (s*4 + t) * 7
    #   
    #   The X matrix elements (2x2) map to U rows:
    #     X[0,0] = A[s*2 + 0, s*2 + 0] -> U row (s*2+0)*4 + (s*2+0)
    #     X[0,1] = A[s*2 + 0, s*2 + 1] -> U row (s*2+0)*4 + (s*2+1)
    #     X[1,0] = A[s*2 + 1, s*2 + 0] -> U row (s*2+1)*4 + (s*2+0)
    #     X[1,1] = A[s*2 + 1, s*2 + 1] -> U row (s*2+1)*4 + (s*2+1)
    #
    #   Wait, let me re-index. For s=0 (top): j ranges 0,1. For s=1 (bottom): j ranges 2,3.
    #   X[i][j_local] where j_local in {0,1} corresponds to A[i][s*2 + j_local]
    #   U row = i * 4 + (s*2 + j_local)
    #
    #   The Y matrix elements (2x2) map to V rows:
    #     Y[j_local][k_local] where j_local in {0,1}, k_local in {0,1}
    #     corresponds to B[s*2 + j_local][2*t + k_local]
    #     V row = (s*2 + j_local) * 8 + (2*t + k_local)
    #
    #   The Z matrix elements (2x2) map to W rows:
    #     Z[i][k_local] where i in {0,1}, k_local in {0,1}
    #     corresponds to C[i][2*t + k_local]
    #     W row = (2*t + k_local) * 2 + i
    
    for s in range(2):  # 0=top, 1=bottom
        for t in range(4):  # column block
            base_col = (s * 4 + t) * 7
            for c in range(7):  # Strassen factor index
                col = base_col + c
                # U contributions
                for xi in range(4):  # X element index: 0=X[0,0], 1=X[0,1], 2=X[1,0], 3=X[1,1]
                    if U_str[xi][c] != 0:
                        i_x = xi // 2  # row in X
                        j_x = xi % 2   # col in X
                        u_row = i_x * 4 + (s * 2 + j_x)
                        U[u_row][col] += U_str[xi][c]
                
                # V contributions
                for yi in range(4):  # Y element index: 0=Y[0,0], 1=Y[0,1], 2=Y[1,0], 3=Y[1,1]
                    if V_str[yi][c] != 0:
                        j_y = yi // 2  # row in Y
                        k_y = yi % 2   # col in Y
                        v_row = (s * 2 + j_y) * 8 + (2 * t + k_y)
                        V[v_row][col] += V_str[yi][c]
                
                # W contributions
                for zi in range(4):  # Z element index: 0=Z[0,0], 1=Z[0,1], 2=Z[1,0], 3=Z[1,1]
                    if W_str[zi][c] != 0:
                        i_z = zi // 2  # row in Z
                        k_z = zi % 2   # col in Z
                        w_row = (2 * t + k_z) * 2 + i_z
                        W[w_row][col] += W_str[zi][c]
    
    return {"U": U, "V": V, "W": W}
