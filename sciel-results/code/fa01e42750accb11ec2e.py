import time
import math

DIMENSIONS = [3, 4, 7]

def _build_strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: W[k*n + i, c] where n=2, m=2, p=2
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0]*rank for _ in range(n*m)]  # 4x7
    V = [[0.0]*rank for _ in range(m*p)]  # 4x7
    W = [[0.0]*rank for _ in range(p*n)]  # 4x7

    # Strassen for 2x2x2:
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = [[c11,c12],[c21,c22]]
    #
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # U rows: a11, a12, a21, a22 (index i*m+j)
    # V rows: b11, b12, b21, b22 (index j*p+k)
    # W rows: c11, c12, c21, c22 (index k*n+i)

    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Construct a tensor decomposition for 3x4x7 matrix multiplication.
    Returns dict with U, V, W.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def _decompose_recursive(nn, mm, pp):
        """Return (U, V, W) for nn x mm x pp multiplication."""
        # Base cases
        if nn == 1 and mm == 1 and pp == 1:
            return [[1]], [[1]], [[1]]
        if nn == 2 and mm == 2 and pp == 2:
            return _build_strassen_2x2x2()
        
        # If any dimension is 1, schoolbook is often optimal or near-optimal for small sizes
        # but we can still try to split if size > 1 to potentially use Strassen on padded parts?
        # For simplicity and correctness, if any dim is 1, we can just use schoolbook
        # UNLESS we can split the other dims to hit 2x2x2.
        # e.g., 1 x 2 x 2. Can we do better than 4?
        # 1x2x2: C1 = a1b11 + a2b21, C2 = a1b12 + a2b22.
        # Rank is 4? 
        # Let's stick to schoolbook for dims with 1 to avoid complex padding logic which might introduce errors.
        if nn == 1 or mm == 1 or pp == 1:
            return _schoolbook(nn, mm, pp)
        
        best_rank = nn * mm * pp
        best_result = _schoolbook(nn, mm, pp)
        
        # Try splitting the middle dimension (mm)
        for mm1 in range(1, mm):
            mm2 = mm - mm1
            # Check time
            if time.time() > deadline:
                break
            U1, V1, W1 = _decompose_recursive(nn, mm1, pp)
            U2, V2, W2 = _decompose_recursive(nn, mm2, pp)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            if total_rank < best_rank:
                best_rank = total_rank
                U = [[0]*total_rank for _ in range(nn*mm)]
                V = [[0]*total_rank for _ in range(mm*pp)]
                W = [[0]*total_rank for _ in range(pp*nn)]
                
                # U: A is nn x mm. Split into A1 (nn x mm1) and A2 (nn x mm2).
                # Rows of U1 correspond to A1. Rows of U2 correspond to A2.
                for i in range(nn*mm1):
                    for c in range(rank1):
                        U[i][c] = U1[i][c]
                for i in range(nn*mm2):
                    for c in range(rank2):
                        U[nn*mm1+i][rank1+c] = U2[i][c]
                
                # V: B is mm x pp. Split into B1 (mm1 x pp) and B2 (mm2 x pp).
                # Rows of V1 correspond to B1. Rows of V2 correspond to B2.
                for j in range(mm1*pp):
                    for c in range(rank1):
                        V[j][c] = V1[j][c]
                for j in range(mm2*pp):
                    for c in range(rank2):
                        V[mm1*pp+j][rank1+c] = V2[j][c]
                
                # W: C is pp x nn. Same for both blocks.
                # W1 and W2 have the same row structure (pp x nn).
                for k in range(pp*nn):
                    for c in range(rank1):
                        W[k][c] = W1[k][c]
                    for c in range(rank2):
                        W[k][rank1+c] = W2[k][c]
                
                best_result = (U, V, W)
        
        # Try splitting the first dimension (nn)
        for nn1 in range(1, nn):
            nn2 = nn - nn1
            if time.time() > deadline:
                break
            U1, V1, W1 = _decompose_recursive(nn1, mm, pp)
            U2, V2, W2 = _decompose_recursive(nn2, mm, pp)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            if total_rank < best_rank:
                best_rank = total_rank
                U = [[0]*total_rank for _ in range(nn*mm)]
                V = [[0]*total_rank for _ in range(mm*pp)]
                W = [[0]*total_rank for _ in range(pp*nn)]
                
                # U: A split into A1 (nn1 x mm) and A2 (nn2 x mm).
                for i in range(nn1*mm):
                    for c in range(rank1):
                        U[i][c] = U1[i][c]
                for i in range(nn2*mm):
                    for c in range(rank2):
                        U[nn1*mm+i][rank1+c] = U2[i][c]
                
                # V: B is same. Concatenate rank cols.
                for j in range(mm*pp):
                    for c in range(rank1):
                        V[j][c] = V1[j][c]
                    for c in range(rank2):
                        V[j][rank1+c] = V2[j][c]
                
                # W: C split into C1 (pp x nn1) and C2 (pp x nn2).
                # W1 rows: index k*nn1 + i1 (0<=i1<nn1)
                # W2 rows: index k*nn2 + i2 (0<=i2<nn2)
                # W_full rows: index k*nn + i (0<=i<nn)
                # For C1: i = i1. W_full[k*nn + i1] gets W1[k*nn1 + i1].
                # For C2: i = nn1 + i2. W_full[k*nn + nn1 + i2] gets W2[k*nn2 + i2].
                
                for k in range(pp):
                    for i1 in range(nn1):
                        idx1 = k*nn1 + i1
                        idx_full = k*nn + i1
                        for c in range(rank1):
                            W[idx_full][c] = W1[idx1][c]
                    for i2 in range(nn2):
                        idx2 = k*nn2 + i2
                        idx_full = k*nn + nn1 + i2
                        for c in range(rank2):
                            W[idx_full][rank1+c] = W2[idx2][c]
                
                best_result = (U, V, W)
        
        # Try splitting the third dimension (pp)
        for pp1 in range(1, pp):
            pp2 = pp - pp1
            if time.time() > deadline:
                break
            U1, V1, W1 = _decompose_recursive(nn, mm, pp1)
            U2, V2, W2 = _decompose_recursive(nn, mm, pp2)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            if total_rank < best_rank:
                best_rank = total_rank
                U = [[0]*total_rank for _ in range(nn*mm)]
                V = [[0]*total_rank for _ in range(mm*pp)]
                W = [[0]*total_rank for _ in range(pp*nn)]
                
                # U: A is same. Concatenate rank cols.
                for i in range(nn*mm):
                    for c in range(rank1):
                        U[i][c] = U1[i][c]
                    for c in range(rank2):
                        U[i][rank1+c] = U2[i][c]
                
                # V: B split into B1 (mm x pp1) and B2 (mm x pp2).
                for j in range(mm*pp1):
                    for c in range(rank1):
                        V[j][c] = V1[j][c]
                for j in range(mm*pp2):
                    for c in range(rank2):
                        V[mm*pp1+j][rank1+c] = V2[j][c]
                
                # W: C split into C1 (pp1 x nn) and C2 (pp2 x nn).
                # W1 rows: index k1*nn + i (0<=k1<pp1)
                # W2 rows: index k2*nn + i (0<=k2<pp2)
                # W_full rows: index k*nn + i (0<=k<pp)
                # For C1: k = k1. W_full[k1*nn + i] gets W1[k1*nn + i].
                # For C2: k = pp1 + k2. W_full[(pp1+k2)*nn + i] gets W2[k2*nn + i].
                
                for k1 in range(pp1):
                    for i in range(nn):
                        idx1 = k1*nn + i
                        idx_full = k1*nn + i
                        for c in range(rank1):
                            W[idx_full][c] = W1[idx1][c]
                for k2 in range(pp2):
                    for i in range(nn):
                        idx2 = k2*nn + i
                        idx_full = (pp1+k2)*nn + i
                        for c in range(rank2):
                            W[idx_full][rank1+c] = W2[idx2][c]
                
                best_result = (U, V, W)
        
        return best_result

    # Execute recursive decomposition
    try:
        U, V, W = _decompose_recursive(n, m, p)
    except Exception as e:
        # Fallback to schoolbook if recursion fails
        U, V, W = _schoolbook(n, m, p)

    # Ensure entries are half-integers
    def _convert_to_halfint(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                h = round(x * 2) / 2
                if abs(x - h) < 1e-9:
                    new_row.append(h)
                else:
                    new_row.append(x)
            result.append(new_row)
        return result

    U = _convert_to_halfint(U)
    V = _convert_to_halfint(V)
    W = _convert_to_halfint(W)

    return {"U": U, "V": V, "W": W}
