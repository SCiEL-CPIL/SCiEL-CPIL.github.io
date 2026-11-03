"""Rank-50 decomposition for 2x4x7 matrix multiplication using Strassen blocks."""

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Strassen 2x2x2, rank 7. U,V,W each shape (4,7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # M2 = (A10+A11)B00
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    W[2][1]=1
    # M3 = A00(B01-B11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1
    W[1][2]=1
    # M4 = A11(B10-B00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # M5 = (A00+A10)B11
    U[0][4]=1; U[2][4]=1; V[3][4]=1
    W[1][4]=1; W[0][4]=-1
    # M6 = (A11-A01)(B00+B01)
    U[3][5]=1; U[1][5]=-1; V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7 = (A01-A10)(B10+B11)
    U[1][6]=1; U[2][6]=-1; V[2][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W

def _schoolbook_2x2x1():
    """Schoolbook 2x2x1, rank 4. U shape (4,4), V shape (2,4), W shape (2,4).
    U[i*2+j,c], V[j,c], W[i,c]."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    c = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j][c]=1
            V[j][c]=1
            W[i][c]=1
            c+=1
    return U, V, W

def construct():
    """Rank-50 decomposition for 2x4x7."""
    n, m, p = 2, 4, 7
    total_rank = 50
    U = [[0]*total_rank for _ in range(n*m)]  # (8, 50)
    V = [[0]*total_rank for _ in range(m*p)]  # (28, 50)
    W = [[0]*total_rank for _ in range(p*n)]  # (14, 50)
    
    col = 0
    for col_group in range(4):
        if col_group < 3:
            # 2x4x2: output cols 2*col_group, 2*col_group+1
            # A0*B_top + A1*B_bot, each 2x2x2 Strassen
            U_s, V_s, W_s = _strassen_2x2x2()
            # A0: rows 0-3 of U
            for i in range(4):
                for c in range(7):
                    U[i][col+c] = U_s[i][c]
            # B_top: rows 0-3 of V
            for j in range(4):
                for c in range(7):
                    V[j][col+c] = V_s[j][c]
            # W: global col = 2*col_group + k_local, row = i_local
            for k_local in range(2):
                for i_local in range(2):
                    for c in range(7):
                        W[(2*col_group + k_local)*n + i_local][col+c] = W_s[k_local*n + i_local][c]
            col += 7
            
            U_s, V_s, W_s = _strassen_2x2x2()
            # A1: rows 4-7 of U
            for i in range(4):
                for c in range(7):
                    U[4+i][col+c] = U_s[i][c]
            # B_bot: rows 4-7 of V
            for j in range(4):
                for c in range(7):
                    V[4+j][col+c] = V_s[j][c]
            # W: same output
            for k_local in range(2):
                for i_local in range(2):
                    for c in range(7):
                        W[(2*col_group + k_local)*n + i_local][col+c] = W_s[k_local*n + i_local][c]
            col += 7
        else:
            # 2x4x1: output col 6
            # A0*B_top + A1*B_bot, each 2x2x1 schoolbook
            U_s, V_s, W_s = _schoolbook_2x2x1()
            for i in range(4):
                for c in range(4):
                    U[i][col+c] = U_s[i][c]
            for j in range(2):
                for c in range(4):
                    V[j][col+c] = V_s[j][c]
            for i in range(2):
                for c in range(4):
                    W[6*n + i][col+c] = W_s[i][c]
            col += 4
            
            U_s, V_s, W_s = _schoolbook_2x2x1()
            for i in range(4):
                for c in range(4):
                    U[4+i][col+c] = U_s[i][c]
            for j in range(2):
                for c in range(4):
                    V[4+j][col+c] = V_s[j][c]
            for i in range(2):
                for c in range(4):
                    W[6*n + i][col+c] = W_s[i][c]
            col += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
