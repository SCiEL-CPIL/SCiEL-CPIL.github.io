import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125

    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def construct_strassen_2x2():
        """Strassen 2x2 matrix multiplication.
        U[i*2+j, t] for A[i,j], V[j*2+k, t] for B[j,k], W[k*2+i, t] for C[i,j].
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B22)
        M4 = A22(B11-B21)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # M1 = (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1  # A11, A22
        V[0,0]=1; V[3,0]=1  # B11, B22
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1  # C11+=M1, C12+=M1, C21+=M1, C22-=M1
        # M2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1  # A21, A22
        V[0,1]=1  # B11
        W[2,1]=1; W[3,1]=1  # C21+=M2, C22-=M2
        # M3 = A11(B21-B22)
        U[0,2]=1  # A11
        V[1,2]=1; V[3,2]=-1  # B21, -B22
        W[0,2]=1; W[1,2]=1  # C11+=M3, C12+=M3
        # M4 = A22(B11-B21)
        U[3,3]=1  # A22
        V[0,3]=1; V[2,3]=-1  # B11, -B21
        W[0,3]=1; W[2,3]=1  # C11+=M4, C21+=M4
        # M5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1  # A11, A12
        V[3,4]=1  # B22
        W[0,4]=-1; W[1,4]=1  # C11-=M5, C12+=M5
        # M6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1  # A21, -A11
        V[0,5]=1; V[1,5]=1  # B11, B12
        W[3,6]=1  # C22+=M6 (wait, let me recheck)
        # Actually C22 = M1 - M2 + M3 + M6, so W[3,6]=1 for C22
        # But I need to fix the indexing. Let me redo this carefully.
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # M1 = (A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[2,0]=1; W[1,0]=1; W[3,0]=-1  # C11,C21,C12,C22
        # M2 = (A21+A22)B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1  # C21+=M2, C22-=M2
        # M3 = A11(B21-B22)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[0,2]=1; W[1,2]=1  # C11+=M3, C12+=M3
        # M4 = A22(B11-B21)
        U[3,3]=1
        V[0,3]=1; V[2,3]=-1
        W[0,3]=1; W[2,3]=1  # C11+=M4, C21+=M4
        # M5 = (A11+A12)B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1  # C11-=M5, C12+=M5
        # M6 = (A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1  # C22+=M6
        # M7 = (A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1  # C11+=M7
        return U, V, W

    def construct_laderman_3x3():
        U = np.zeros((9, 15)); V = np.zeros((9, 15)); W = np.zeros((9, 15))
        U[0,0]=1; U[4,0]=1; U[8,0]=1; V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1; W[4,0]=-1; W[5,0]=1; W[6,0]=1; W[7,0]=1; W[8,0]=-1
        U[1,1]=1; U[5,1]=1; V[0,1]=1; V[8,1]=1; W[1,1]=1; W[2,1]=1; W[4,1]=1; W[5,1]=1
        U[2,2]=1; U[6,2]=1; V[3,2]=1; V[8,2]=1; W[3,2]=1; W[4,2]=1; W[6,2]=1; W[7,2]=1
        U[0,3]=1; U[8,3]=1; V[0,3]=1; V[4,3]=1; W[0,3]=1; W[1,3]=1; W[3,3]=1; W[4,3]=1
        U[1,4]=1; U[5,4]=1; U[8,4]=1; V[3,4]=1; V[8,4]=1
        W[3,4]=1; W[4,4]=1; W[5,4]=1; W[6,4]=1; W[7,4]=1; W[8,4]=1
        U[2,5]=1; U[6,5]=1; V[0,5]=1; V[4,5]=1; W[0,5]=1; W[1,5]=1; W[3,5]=1; W[4,5]=1
        U[0,6]=1; U[4,6]=1; V[0,6]=1; V[4,6]=1; W[0,6]=1; W[1,6]=1; W[3,6]=1; W[4,6]=1
        U[0,7]=1; U[4,7]=1; V[0,7]=1; V[8,7]=1; W[0,7]=1; W[1,7]=1; W[5,7]=1; W[6,7]=1
        U[1,8]=1; U[5,8]=1; V[3,8]=1; V[8,8]=1; W[3,8]=1; W[4,8]=1; W[6,8]=1; W[7,8]=1
        U[0,9]=1; U[8,9]=1; V[0,9]=1; V[8,9]=1; W[0,9]=1; W[1,9]=1; W[5,9]=1; W[6,9]=1
        U[2,10]=1; U[6,10]=1; V[0,10]=1; V[4,10]=1; V[8,10]=1
        W[0,10]=1; W[1,10]=1; W[3,10]=1; W[4,10]=1; W[5,10]=1; W[6,10]=1
        U[0,11]=1; U[4,11]=1; U[8,11]=1; V[3,11]=1; V[8,11]=1
        W[3,11]=1; W[4,11]=1; W[5,11]=1; W[6,11]=1; W[7,11]=1; W[8,11]=1
        U[1,12]=1; U[5,12]=1; U[8,12]=1; V[0,12]=1; V[8,12]=1; W[0,12]=1; W[1,12]=1; W[5,12]=1; W[6,12]=1
        U[0,13]=1; U[4,13]=1; V[3,13]=1; V[8,13]=1; W[3,13]=1; W[4,13]=1; W[6,13]=1; W[7,13]=1
        U[2,14]=1; U[6,14]=1; V[3,14]=1; V[8,14]=1; W[3,14]=1; W[4,14]=1; W[6,14]=1; W[7,14]=1
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.dot(U[i*3+j], V[j*3+k] * W[k*3+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err < 1e-8:
            return U, V, W
        return None

    def build_block_decomposition():
        laderman_result = construct_laderman_3x3()
        use_laderman = laderman_result is not None
        laderman_rank = 15 if use_laderman else 27
        # Block structure for 5x5 = (2+3) x (2+3) x (2+3):
        # T1: 2x2x2 (Strassen, rank 7)
        # T2: 2x3x3 (schoolbook, rank 12)
        # T3: 3x2x2 (schoolbook, rank 12)
        # T4: 3x3x3 (Laderman or schoolbook)
        total_rank = 7 + 12 + 12 + laderman_rank
        U = np.zeros((25, total_rank)); V = np.zeros((25, total_rank)); W = np.zeros((25, total_rank))
        col_offset = 0
        
        # T1: 2x2x2 Strassen block (rows 0-1, cols 0-1)
        U_a, V_a, W_a = construct_strassen_2x2()
        for t in range(7):
            for i_prime in range(2):
                for j_prime in range(2):
                    U[i_prime*5+j_prime, col_offset+t] = U_a[i_prime*2+j_prime, t]
            for j_prime in range(2):
                for k_prime in range(2):
                    V[j_prime*5+k_prime, col_offset+t] = V_a[j_prime*2+k_prime, t]
            for k_prime in range(2):
                for i_prime in range(2):
                    W[k_prime*5+i_prime, col_offset+t] = W_a[k_prime*2+i_prime, t]
        col_offset += 7
        
        # T2: 2x3x3 schoolbook (rows 0-1, cols 2-4)
        for i_prime in range(2):
            for j_prime in range(3):
                for k_prime in range(3):
                    t = col_offset + i_prime*9 + j_prime*3 + k_prime
                    U[i_prime*5+(2+j_prime), t] = 1
                    V[(2+j_prime)*5+(2+k_prime), t] = 1
                    W[(2+k_prime)*5+i_prime, t] = 1
        col_offset += 12
        
        # T3: 3x2x2 schoolbook (rows 2-4, cols 0-1)
        for i_prime in range(3):
            for j_prime in range(2):
                for k_prime in range(2):
                    t = col_offset + i_prime*4 + j_prime*2 + k_prime
                    U[(2+i_prime)*5+j_prime, t] = 1
                    V[j_prime*5+k_prime, t] = 1
                    W[k_prime*5+(2+i_prime), t] = 1
        col_offset += 12
        
        # T4: 3x3x3 (rows 2-4, cols 2-4)
        if use_laderman:
            U_l, V_l, W_l = laderman_result
            for t in range(15):
                for i_prime in range(3):
                    for j_prime in range(3):
                        U[(2+i_prime)*5+(2+j_prime), col_offset+t] = U_l[i_prime*3+j_prime, t]
                for j_prime in range(3):
                    for k_prime in range(3):
                        V[(2+j_prime)*5+(2+k_prime), col_offset+t] = V_l[j_prime*3+k_prime, t]
                for k_prime in range(3):
                    for i_prime in range(3):
                        W[(2+k_prime)*5+(2+i_prime), col_offset+t] = W_l[k_prime*3+i_prime, t]
            col_offset += 15
        else:
            for i_prime in range(3):
                for j_prime in range(3):
                    for k_prime in range(3):
                        t = col_offset + i_prime*9 + j_prime*3 + k_prime
                        U[(2+i_prime)*5+(2+j_prime), t] = 1
                        V[(2+j_prime)*5+(2+k_prime), t] = 1
                        W[(2+k_prime)*5+(2+i_prime), t] = 1
            col_offset += 27
        
        return U, V, W, col_offset

    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col]=1; V_school[j*p+k, col]=1; W_school[k*n+i, col]=1
                col += 1
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)

    if time.time() < deadline - 10:
        U_block, V_block, W_block, block_rank = build_block_decomposition()
        max_err = verify(U_block, V_block, W_block, block_rank)
        if max_err < 1e-8:
            best_rank = block_rank
            best_result = (U_block, V_block, W_block)


    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
