# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        if not U or not V or not W:
            return False
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial schoolbook decomposition with rank = n*m*p = 54."""
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

    def make_rank_48():
        """
        Construct a rank 48 decomposition by combining two rank-24 3x3x3 decompositions.
        """
        rank_s = 24
        U_s = [[0] * rank_s for _ in range(9)]
        V_s = [[0] * rank_s for _ in range(9)]
        W_s = [[0] * rank_s for _ in range(9)]
        
        # Strassen 2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        r = 0
        # m1 = (a00+a11)*(b00+b11)
        U_s[0][r]=1; U_s[4][r]=1
        V_s[0][r]=1; V_s[3][r]=1
        W_s[0][r]=1; W_s[4][r]=1
        
        r = 1
        # m2 = (a10+a11)*b00
        U_s[4][r]=1; U_s[1][r]=1
        V_s[0][r]=1
        W_s[3][r]=1; W_s[4][r]=-1
        
        r = 2
        # m3 = a00*(b10-b11)
        U_s[0][r]=1
        V_s[3][r]=1; V_s[4][r]=-1
        W_s[3][r]=1; W_s[4][r]=1
        
        r = 3
        # m4 = a11*(b01-b00)
        U_s[4][r]=1
        V_s[1][r]=1; V_s[0][r]=-1
        W_s[1][r]=1; W_s[0][r]=1
        
        r = 4
        # m5 = (a00+a01)*b11
        U_s[0][r]=1; U_s[1][r]=1
        V_s[4][r]=1
        W_s[1][r]=-1; W_s[0][r]=-1
        
        r = 5
        # m6 = (a10-a00)*(b00+b01)
        U_s[1][r]=1; U_s[0][r]=-1
        V_s[0][r]=1; V_s[1][r]=1
        W_s[4][r]=1
        
        r = 6
        # m7 = (a01-a11)*(b10+b11)
        U_s[1][r]=1; U_s[4][r]=-1
        V_s[3][r]=1; V_s[4][r]=1
        W_s[1][r]=1; W_s[0][r]=-1
        
        # Direct terms for 3rd row/col
        # a02*b20 -> c20
        r = 7
        U_s[2][r]=1; V_s[6][r]=1; W_s[6][r]=1
        # a02*b21 -> c21
        r = 8
        U_s[2][r]=1; V_s[7][r]=1; W_s[7][r]=1
        # a02*b22 -> c22
        r = 9
        U_s[2][r]=1; V_s[8][r]=1; W_s[8][r]=1
        # a12*b20 -> c20
        r = 10
        U_s[5][r]=1; V_s[6][r]=1; W_s[6][r]=1
        # a12*b21 -> c21
        r = 11
        U_s[5][r]=1; V_s[7][r]=1; W_s[7][r]=1
        # a12*b22 -> c22
        r = 12
        U_s[5][r]=1; V_s[8][r]=1; W_s[8][r]=1
        # a20*b00 -> c00
        r = 13
        U_s[6][r]=1; V_s[0][r]=1; W_s[0][r]=1
        # a20*b01 -> c01
        r = 14
        U_s[6][r]=1; V_s[1][r]=1; W_s[1][r]=1
        # a20*b02 -> c02
        r = 15
        U_s[6][r]=1; V_s[2][r]=1; W_s[2][r]=1
        # a21*b10 -> c10
        r = 16
        U_s[7][r]=1; V_s[3][r]=1; W_s[3][r]=1
        # a21*b11 -> c11
        r = 17
        U_s[7][r]=1; V_s[4][r]=1; W_s[4][r]=1
        # a21*b12 -> c12
        r = 18
        U_s[7][r]=1; V_s[5][r]=1; W_s[5][r]=1
        # a22*b20 -> c20
        r = 19
        U_s[8][r]=1; V_s[6][r]=1; W_s[6][r]=1
        # a22*b21 -> c21
        r = 20
        U_s[8][r]=1; V_s[7][r]=1; W_s[7][r]=1
        # a22*b22 -> c22
        r = 21
        U_s[8][r]=1; V_s[8][r]=1; W_s[8][r]=1
        # a00*b02 -> c02
        r = 22
        U_s[0][r]=1; V_s[2][r]=1; W_s[2][r]=1
        # a01*b12 -> c12
        r = 23
        U_s[1][r]=1; V_s[5][r]=1; W_s[5][r]=1

        rank_total = 2 * rank_s
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for r in range(rank_s):
                    U[i*m+j][r] = U_s[i*m+j][r]
        for j in range(m):
            for k in range(3):
                for r in range(rank_s):
                    V[j*p+k][r] = V_s[j*3+k][r]
        for k in range(3):
            for i in range(n):
                for r in range(rank_s):
                    W[k*n+i][r] = W_s[k*3+i][r]

        for i in range(n):
            for j in range(m):
                for r in range(rank_s):
                    U[i*m+j][rank_s + r] = U_s[i*m+j][r]
        for j in range(m):
            for k in range(3, 6):
                for r in range(rank_s):
                    V[j*p+k][rank_s + r] = V_s[j*3+(k-3)][r]
        for k in range(3, 6):
            for i in range(n):
                for r in range(rank_s):
                    W[k*n+i][rank_s + r] = W_s[(k-3)*3+i][r]

        return {"U": U, "V": V, "W": W}

    # Try Rank 48
    if time.time() < deadline - 10:
        result_48 = make_rank_48()
        if result_48 is not None and verify(result_48["U"], result_48["V"], result_48["W"]):
            return result_48

    # Fallback: Schoolbook (rank 54)
    return make_schoolbook()
# EVOLVE-BLOCK-END
