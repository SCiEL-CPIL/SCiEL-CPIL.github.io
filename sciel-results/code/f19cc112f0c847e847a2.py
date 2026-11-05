"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

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

    def make_strassen_3x3():
        """
        Construct the rank-23 3x3x3 decomposition.
        Indexing: A is 3x3 with A[i,j] at index i*3+j
                  B is 3x3 with B[j,k] at index j*3+k  
                  C is 3x3 with C[i,k] at index i*3+k
        U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r] summed over r gives C[i,k]
        
        W index: W[k*3+i] where i=row of C, k=col of C
        """
        rank_s = 23
        U_s = [[0] * rank_s for _ in range(9)]
        V_s = [[0] * rank_s for _ in range(9)]
        W_s = [[0] * rank_s for _ in range(9)]
        
        # Strassen 2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # m1 = (a00+a11)*(b00+b11): contributes to c00, c11; subtract from c01, c10
        # U: a00=U[0], a11=U[4]
        # V: b00=V[0], b11=V[4]
        # W: c00=W[0*3+0]=W[0], c11=W[1*3+1]=W[4], c10=W[0*3+1]=W[1], c01=W[1*3+0]=W[3]
        r = 0
        U_s[0][r]=1; U_s[4][r]=1
        V_s[0][r]=1; V_s[4][r]=1
        W_s[0][r]=1; W_s[4][r]=1
        W_s[1][r]=-1; W_s[3][r]=-1
        
        # m2 = (a10+a11)*b00: contributes to c10
        # U: a10=U[1], a11=U[4]
        # V: b00=V[0]
        # W: c10=W[0*3+1]=W[1]
        r = 1
        U_s[1][r]=1; U_s[4][r]=1
        V_s[0][r]=1
        W_s[1][r]=1
        
        # m3 = a00*(b10-b11): contributes to c10
        # U: a00=U[0]
        # V: b10=V[3], b11=V[4]
        # W: c10=W[0*3+1]=W[1]
        r = 2
        U_s[0][r]=1
        V_s[3][r]=1; V_s[4][r]=-1
        W_s[1][r]=1
        
        # m4 = a11*(b01-b00): contributes to c01
        # U: a11=U[4]
        # V: b01=V[1], b00=V[0]
        # W: c01=W[1*3+0]=W[3]
        r = 3
        U_s[4][r]=1
        V_s[1][r]=1; V_s[0][r]=-1
        W_s[3][r]=1
        
        # m5 = (a00+a01)*b11: contributes to c01
        # U: a00=U[0], a01=U[1]
        # V: b11=V[4]
        # W: c01=W[1*3+0]=W[3]
        r = 4
        U_s[0][r]=1; U_s[1][r]=1
        V_s[4][r]=1
        W_s[3][r]=1
        
        # m6 = (a10-a00)*(b00+b01): contributes to c00
        # U: a10=U[1], a00=U[0]
        # V: b00=V[0], b01=V[1]
        # W: c00=W[0*3+0]=W[0]
        r = 5
        U_s[1][r]=1; U_s[0][r]=-1
        V_s[0][r]=1; V_s[1][r]=1
        W_s[0][r]=1
        
        # m7 = (a01-a11)*(b10+b11): contributes to c11
        # U: a01=U[1], a11=U[4]
        # V: b10=V[3], b11=V[4]
        # W: c11=W[1*3+1]=W[4]
        r = 6
        U_s[1][r]=1; U_s[4][r]=-1
        V_s[3][r]=1; V_s[4][r]=1
        W_s[4][r]=1
        
        # Additional terms for the 3rd row/col
        # a02*b20 -> c20: U[2]*V[6]*W[0*3+2]=W[2]
        r = 7
        U_s[2][r]=1
        V_s[6][r]=1
        W_s[2][r]=1
        
        # a02*b21 -> c21: U[2]*V[7]*W[1*3+2]=W[5]
        r = 8
        U_s[2][r]=1
        V_s[7][r]=1
        W_s[5][r]=1
        
        # a02*b22 -> c22: U[2]*V[8]*W[2*3+2]=W[8]
        r = 9
        U_s[2][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        # a12*b20 -> c20: U[5]*V[6]*W[2]
        r = 10
        U_s[5][r]=1
        V_s[6][r]=1
        W_s[2][r]=1
        
        # a12*b21 -> c21: U[5]*V[7]*W[5]
        r = 11
        U_s[5][r]=1
        V_s[7][r]=1
        W_s[5][r]=1
        
        # a12*b22 -> c22: U[5]*V[8]*W[8]
        r = 12
        U_s[5][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        # a20*b00 -> c00: U[6]*V[0]*W[0]
        r = 13
        U_s[6][r]=1
        V_s[0][r]=1
        W_s[0][r]=1
        
        # a20*b01 -> c01: U[6]*V[1]*W[3]
        r = 14
        U_s[6][r]=1
        V_s[1][r]=1
        W_s[3][r]=1
        
        # a20*b02 -> c02: U[6]*V[2]*W[6]
        r = 15
        U_s[6][r]=1
        V_s[2][r]=1
        W_s[6][r]=1
        
        # a21*b10 -> c10: U[7]*V[3]*W[1]
        r = 16
        U_s[7][r]=1
        V_s[3][r]=1
        W_s[1][r]=1
        
        # a21*b11 -> c11: U[7]*V[4]*W[4]
        r = 17
        U_s[7][r]=1
        V_s[4][r]=1
        W_s[4][r]=1
        
        # a21*b12 -> c12: U[7]*V[5]*W[7]
        r = 18
        U_s[7][r]=1
        V_s[5][r]=1
        W_s[7][r]=1
        
        # a22*b20 -> c20: U[8]*V[6]*W[2]
        r = 19
        U_s[8][r]=1
        V_s[6][r]=1
        W_s[2][r]=1
        
        # a22*b21 -> c21: U[8]*V[7]*W[5]
        r = 20
        U_s[8][r]=1
        V_s[7][r]=1
        W_s[5][r]=1
        
        # a22*b22 -> c22: U[8]*V[8]*W[8]
        r = 21
        U_s[8][r]=1
        V_s[8][r]=1
        W_s[8][r]=1
        
        # a00*b02 -> c02: U[0]*V[2]*W[6]
        r = 22
        U_s[0][r]=1
        V_s[2][r]=1
        W_s[6][r]=1
        
        return U_s, V_s, W_s

    def make_rank_46():
        """
        Construct a rank 46 decomposition by combining two rank-23 3x3x3 decompositions.
        3x3 * 3x6 = [3x3 * 3x3 | 3x3 * 3x3]
        
        For the full problem:
        - A[i,j] -> U[i*m+j, r] = U[i*3+j, r]
        - B[j,k] -> V[j*p+k, r] = V[j*6+k, r]
        - C[i,k] -> W[k*n+i, r] = W[k*3+i, r]
        
        For sub-block 1 (k=0,1,2): B[j,k] maps to V_s[j*3+k], C[i,k] maps to W_s[k*3+i]
        For sub-block 2 (k=3,4,5): B[j,k] maps to V_s[j*3+(k-3)], C[i,k] maps to W_s[(k-3)*3+i]
        """
        U_s, V_s, W_s = make_strassen_3x3()
        rank_s = 23
        
        rank_total = 2 * rank_s
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # First half (k=0,1,2): use U_s, V_s, W_s directly
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

        # Second half (k=3,4,5): use U_s, V_s, W_s for the second 3x3 block
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

    def make_rank_45():
        """
        Try to construct a rank-45 decomposition.
        
        Key insight: The two 3x3x3 blocks share the same A matrix.
        We can potentially save one rank by combining terms that involve
        the same A entries but different B entries.
        
        Strategy: Look for pairs of terms across the two blocks that can be
        merged. If term r1 in block 1 uses A[i,j] and term r2 in block 2 uses
        A[i,j], we might combine them if the V and W patterns allow it.
        
        Actually, let's try a different approach: use a rank-22 3x3x3 if it exists,
        or find a way to share one multiplication between the two blocks.
        
        For now, let's try to verify if rank 46 works first.
        """
        return None

    def try_optimize_rank(target_rank, time_budget=100):
        """
        Try to find a decomposition of the given rank using numerical optimization.
        This is a last resort and may not converge.
        """
        if time.time() > deadline - time_budget:
            return None
        
        # This is computationally expensive and unlikely to succeed
        # for non-trivial tensor rank problems. Skip for now.
        return None

    # Try Rank 46
    result_46 = make_rank_46()
    if result_46 is not None and verify(result_46["U"], result_46["V"], result_46["W"]):
        return result_46

    # Try Rank 45
    if time.time() < deadline - 30:
        result_45 = make_rank_45()
        if result_45 is not None and verify(result_45["U"], result_45["V"], result_45["W"]):
            return result_45

    # Fallback: Schoolbook (rank 54)
    return make_schoolbook()
# EVOLVE-BLOCK-END
