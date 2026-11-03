import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition for the remainder
    # Strassen decomposition for 2x2x2 blocks
    
    def get_strassen_factors():
        # Returns U, V, W for 2x2x2
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=1: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=2: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1] = 1
        
        # r=3: M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = -1
        
        # r=4: M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = -1
        
        # r=5: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = 1; W[3][4] = 1
        
        # r=6: M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[0][5] = 1; W[2][5] = 1
        
        # r=7: M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[2][6] = 1
        
        return U, V, W

    U_str, V_str, W_str = get_strassen_factors()
    
    # Define blocks
    # i in {0,1}, j in {0,1}, {2,3}, k in {0,1}, {2,3}, {4,5}
    blocks = []
    for i_start in [0]:
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                blocks.append((i_start, j_start, k_start))
                
    # Initialize full matrices
    # Rank = 6 * 7 + (105 - 48) = 42 + 57 = 99
    rank = 99
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col_offset = 0
    
    # Add Strassen blocks
    for i_s, j_s, k_s in blocks:
        for r in range(7):
            for i_l in range(2):
                for j_l in range(2):
                    if U_str[i_l*2+j_l][r] != 0:
                        i_g = i_s + i_l
                        j_g = j_s + j_l
                        U[i_g*m + j_g][col_offset] = U_str[i_l*2+j_l][r]
                        
            for j_l in range(2):
                for k_l in range(2):
                    if V_str[j_l*2+k_l][r] != 0:
                        j_g = j_s + j_l
                        k_g = k_s + k_l
                        V[j_g*p + k_g][col_offset] = V_str[j_l*2+k_l][r]
                        
            for k_l in range(2):
                for i_l in range(2):
                    if W_str[k_l*2+i_l][r] != 0:
                        k_g = k_s + k_l
                        i_g = i_s + i_l
                        W[k_g*n + i_g][col_offset] = W_str[k_l*2+i_l][r]
                        
            col_offset += 1
            
    # Add schoolbook for the rest
    # The rest are indices not in any block
    # Block covers i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Rest:
    # 1. i=2, any j, any k
    # 2. i in {0,1}, j=4, any k
    # 3. i in {0,1}, j in {0,1,2,3}, k=6
    
    def in_block(i, j, k):
        if i >= 2: return False
        if j >= 4: return False
        if k >= 6: return False
        return True

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not in_block(i, j, k):
                    U[i*m + j][col_offset] = 1
                    V[j*p + k][col_offset] = 1
                    W[k*n + i][col_offset] = 1
                    col_offset += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
