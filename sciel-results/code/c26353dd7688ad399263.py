import time
import random

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 100
    
    # Rank-30 decomposition using Strassen on 2x2 sub-blocks
    # A (3x3), B (3x4), C (3x4)
    # Split: A = [A_tl(2x2) | A_tr(2x1); A_bl(1x2) | A_br(1x1)]
    #        B = [B_top(2x4); B_bot(1x4)]
    #        C = [C_top(2x4); C_bot(1x4)]
    # C_top = A_tl*B_top + A_tr*B_bot
    # C_bot = A_bl*B_top + A_br*B_bot
    # A_tl*B_top split into A_tl*B_tl(2x2*2x2) + A_tl*B_tr(2x2*2x2)
    # Each 2x2*2x2 uses Strassen rank 7
    # A_tr*B_bot: rank 8
    # A_bl*B_top: rank 4
    # A_br*B_bot: rank 4
    # Total: 7+7+8+4+4 = 30
    
    rank = 30
    U = [[0.0]*rank for _ in range(n*m)]  # (9, 30)
    V = [[0.0]*rank for _ in range(m*p)]  # (12, 30)
    W = [[0.0]*rank for _ in range(p*n)]  # (12, 30)
    
    # U indices: A[i*m+j] = A[i*3+j]
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # A[2,0]=6, A[2,1]=7, A[2,2]=8
    
    # V indices: B[j*p+k] = B[j*4+k]
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3
    # B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
    
    # W indices: W[k*n+i] = W[k*3+i]
    # C[0,0]=W[0], C[0,1]=W[3], C[0,2]=W[6], C[0,3]=W[9]
    # C[1,0]=W[1], C[1,1]=W[4], C[1,2]=W[7], C[1,3]=W[10]
    # C[2,0]=W[2], C[2,1]=W[5], C[2,2]=W[8], C[2,3]=W[11]
    
    col = 0
    # Strassen for 2x2*2x2: A_tl * B_tl
    # A_tl = [[A00, A01],[A10, A11]] -> U indices 0,1,3,4
    # B_tl = [[B00, B01],[B10, B11]] -> V indices 0,1,4,5
    # C_top_left = [[C00, C01],[C10, C11]] -> W indices 0,3,1,4
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[5][col] += 1
    W[0][col] += 1; W[4][col] += 1  # C00, C11
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[1][col] += 1  # C10
    W[4][col] += -1  # C11
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] += 1
    V[4][col] += 1; V[5][col] += -1
    W[3][col] += 1  # C01
    W[4][col] += 1  # C11
    col += 1
    
    # M4 = A11(B01-B00)
    U[4][col] += 1
    V[1][col] += 1; V[0][col] += -1
    W[0][col] += 1  # C00
    W[1][col] += 1  # C10
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] += 1; U[1][col] += 1
    V[5][col] += 1
    W[0][col] += -1  # C00
    W[3][col] += 1  # C01
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] += 1; U[0][col] += -1
    V[0][col] += 1; V[1][col] += 1
    W[4][col] += 1  # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] += 1; U[4][col] += -1
    V[4][col] += 1; V[5][col] += 1
    W[0][col] += 1  # C00
    col += 1
    
    # Strassen for 2x2*2x2: A_tl * B_tr
    # B_tr = [[B02, B03],[B12, B13]] -> V indices 2,3,6,7
    # C_top_right = [[C02, C03],[C12, C13]] -> W indices 6,9,7,10
    
    # M8 = (A00+A11)(B02+B12)
    U[0][col] += 1; U[4][col] += 1
    V[2][col] += 1; V[6][col] += 1
    W[6][col] += 1; W[10][col] += 1  # C02, C13
    col += 1
    
    # M9 = (A10+A11)B02
    U[3][col] += 1; U[4][col] += 1
    V[2][col] += 1
    W[7][col] += 1  # C12
    W[10][col] += -1  # C13
    col += 1
    
    # M10 = A00(B12-B13)
    U[0][col] += 1
    V[6][col] += 1; V[7][col] += -1
    W[9][col] += 1  # C03
    W[10][col] += 1  # C13
    col += 1
    
    # M11 = A11(B03-B02)
    U[4][col] += 1
    V[3][col] += 1; V[2][col] += -1
    W[6][col] += 1  # C02
    W[7][col] += 1  # C12
    col += 1
    
    # M12 = (A00+A01)B13
    U[0][col] += 1; U[1][col] += 1
    V[7][col] += 1
    W[6][col] += -1  # C02
    W[9][col] += 1  # C03
    col += 1
    
    # M13 = (A10-A00)(B02+B03)
    U[3][col] += 1; U[0][col] += -1
    V[2][col] += 1; V[3][col] += 1
    W[10][col] += 1  # C13
    col += 1
    
    # M14 = (A01-A11)(B12+B13)
    U[1][col] += 1; U[4][col] += -1
    V[6][col] += 1; V[7][col] += 1
    W[6][col] += 1  # C02
    col += 1
    
    # A_tr * B_bot: 2x1 * 1x4, rank 8
    # A_tr = [A02, A12] -> U indices 2, 5
    # B_bot = [B20, B21, B22, B23] -> V indices 8,9,10,11
    # C_top contribution: C0k += A02*B2k, C1k += A12*B2k
    # C00=W[0], C01=W[3], C02=W[6], C03=W[9]
    # C10=W[1], C11=W[4], C12=W[7], C13=W[10]
    
    for k in range(4):
        # A02 * B2k
        U[2][col] += 1
        V[8+k][col] += 1
        W[3*k][col] += 1  # C[0,k]
        col += 1
        # A12 * B2k
        U[5][col] += 1
        V[8+k][col] += 1
        W[3*k+1][col] += 1  # C[1,k]
        col += 1
    
    # A_bl * B_top: 1x2 * 2x4, rank 4
    # A_bl = [A20, A21] -> U indices 6, 7
    # B_top = [B00..B03, B10..B13] -> V indices 0,1,2,3,4,5,6,7
    # C_bot[k] = A20*B0k + A21*B1k
    # C20=W[2], C21=W[5], C22=W[8], C23=W[11]
    
    for k in range(4):
        # A20 * B0k
        U[6][col] += 1
        V[k][col] += 1
        W[3*k+2][col] += 1  # C[2,k]
        # A21 * B1k
        U[7][col] += 1
        V[4+k][col] += 1
        W[3*k+2][col] += 1  # C[2,k]
        col += 1
    
    # A_br * B_bot: 1x1 * 1x4, rank 4
    # A_br = A22 -> U index 8
    # B_bot = [B20, B21, B22, B23] -> V indices 8,9,10,11
    # C_bot[k] += A22*B2k
    # C20=W[2], C21=W[5], C22=W[8], C23=W[11]
    
    for k in range(4):
        U[8][col] += 1
        V[8+k][col] += 1
        W[3*k+2][col] += 1  # C[2,k]
        col += 1
    
    # Verify
    if time.time() < deadline:
        random.seed(42)
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_dec = [[0.0]*p for _ in range(n)]
        for r in range(rank):
            u_val = sum(U[i*m+j][r] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j*p+k][r] * B[j][k] for j in range(m) for k in range(p))
            prod = u_val * v_val
            for k in range(p):
                for i in range(n):
                    C_dec[i][k] += W[k*n+i][r] * prod
        
        valid = True
        for i in range(n):
            for k in range(p):
                if abs(C_dec[i][k] - C[i][k]) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank_sb = n * m * p
            U_sb = [[0.0]*rank_sb for _ in range(n*m)]
            V_sb = [[0.0]*rank_sb for _ in range(m*p)]
            W_sb = [[0.0]*rank_sb for _ in range(p*n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j][c] = 1.0
                        V_sb[j*p+k][c] = 1.0
                        W_sb[k*n+i][c] = 1.0
                        c += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
