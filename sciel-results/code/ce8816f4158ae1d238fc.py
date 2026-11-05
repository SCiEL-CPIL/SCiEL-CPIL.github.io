"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition search."""
DIMENSIONS = [4, 5, 5]

import sys

def construct():
    n, m, p = DIMENSIONS
    
    def trivial_decomp(n, m, p):
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
        return U, V, W

    def strassen_2x2x2():
        # Standard Strassen decomposition for 2x2x2
        # Returns U (4x7), V (4x7), W (4x7)
        # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = -1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = 1
        
        # M3 = A11(B21-B22)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[2][2] = -1
        
        # M4 = A22(B12-B11)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[1][3] = -1; W[3][3] = 1
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[2][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[1][5] = 1; W[0][5] = -1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[2][6] = 1; W[3][6] = -1
        
        return U, V, W

    def combine_decomps(n1, m1, p1, n2, m2, p2, d1, d2):
        # d1 = (U1, V1, W1) for n1 x m1 x p1
        # d2 = (U2, V2, W2) for n2 x m2 x p2
        # We are computing (n1+n2) x (m1+m2) x (p1+p2)? No, we split one dimension.
        # This helper is for splitting p: n x m x (p1+p2)
        # U is n*m x (rank1+rank2)
        # V is m*(p1+p2) x (rank1+rank2)
        # W is (p1+p2)*n x (rank1+rank2)
        
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        
        rank = len(U1[0]) + len(U2[0])
        U = [[0]*rank for _ in range(n1*m1)] # Note: n1 should be n, m1 should be m
        V = [[0]*rank for _ in range(m1*(p1+p2))]
        W = [[0]*rank for _ in range((p1+p2)*n1)]
        
        # Fill U: same for both parts in terms of rows? No.
        # U depends on i, j. i in [0,n), j in [0,m).
        # Since we split p, U is identical for both parts? No, U is just U.
        # Wait, the decomposition is additive: T = T1 + T2.
        # T1 uses U1, V1, W1. T2 uses U2, V2, W2.
        # U is n*m x rank. For the first rank1 cols, it's U1. For next rank2 cols, it's U2.
        # But U1 and U2 have the same shape (n*m x rank1/2).
        # So U[:, :rank1] = U1, U[:, rank1:] = U2.
        
        for i in range(n1*m1):
            for r in range(len(U1[0])):
                U[i][r] = U1[i][r]
            for r in range(len(U2[0])):
                U[i][rank1 + r] = U2[i][r]
        
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        
        # V is m x (p1+p2). 
        # Rows 0..p1-1 correspond to V1. Rows p1..p1+p2-1 correspond to V2.
        # V[:, :rank1] for first p1 rows is V1.
        # V[:, rank1:] for first p1 rows is 0.
        # V[:, :rank1] for last p2 rows is 0.
        # V[:, rank1:] for last p2 rows is V2.
        
        for j in range(m1):
            for k in range(p1):
                idx_v = j * (p1+p2) + k
                for r in range(rank1):
                    V[idx_v][r] = V1[j*p1+k][r]
            for k in range(p2):
                idx_v = j * (p1+p2) + p1 + k
                for r in range(rank2):
                    V[idx_v][rank1+r] = V2[j*p2+k][r]
        
        # W is (p1+p2) x n.
        # Rows 0..p1-1 correspond to W1. Rows p1..p1+p2-1 correspond to W2.
        # W[:, :rank1] for first p1 rows is W1.
        # W[:, rank1:] for first p1 rows is 0.
        # W[:, :rank1] for last p2 rows is 0.
        # W[:, rank1:] for last p2 rows is W2.
        
        for k in range(p1):
            for i in range(n1):
                idx_w = k * n1 + i
                for r in range(rank1):
                    W[idx_w][r] = W1[k*n1+i][r]
        for k in range(p2):
            for i in range(n1):
                idx_w = (p1+k) * n1 + i
                for r in range(rank2):
                    W[idx_w][rank1+r] = W2[k*n2+i][r] # Note: n2 should be n1 here if we split p
            
        return U, V, W

    def get_decomp(n, m, p):
        if n == 1 and m == 1 and p == 1:
            U = [[1]]; V = [[1]]; W = [[1]]
            return U, V, W
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        # Heuristic: Split the largest dimension.
        # Prefer splitting p, then m, then n.
        if p >= m and p >= n:
            if p == 1:
                return trivial_decomp(n, m, p)
            p1 = p // 2
            p2 = p - p1
            d1 = get_decomp(n, m, p1)
            d2 = get_decomp(n, m, p2)
            return combine_decomps(n, m, p1, n, m, p2, d1, d2)
        elif m >= p and m >= n:
            if m == 1:
                return trivial_decomp(n, m, p)
            m1 = m // 2
            m2 = m - m1
            # Split m: n x (m1+m2) x p
            # U: n x (m1+m2). V: (m1+m2) x p. W: p x n.
            # T = T1 + T2.
            # U1: n x m1. U2: n x m2.
            # U: n x (m1+m2). U[:, :m1] = U1? No, U is n*m x rank.
            # This logic is getting complicated to generalize combine_decomps.
            # Let's just stick to splitting p for simplicity, or use trivial if m/n split needed.
            # For 4,5,5, we split p=5 into 2,3.
            pass
        
        # Fallback to trivial for other splits to ensure correctness
        return trivial_decomp(n, m, p)
        
    # For 4,5,5, we can specifically construct:
    # Split p=5 into 2 and 3.
    # d1 = get_decomp(4, 5, 2)
    # d2 = get_decomp(4, 5, 3)
    
    d1 = get_decomp(4, 5, 2)
    d2 = get_decomp(4, 5, 3)
    
    # We need a specific combine function for splitting p
    def combine_split_p(n, m, p1, p2, d1, d2):
        U1, V1, W1 = d1
        U2, V2, W2 = d2
        rank1 = len(U1[0])
        rank2 = len(U2[0])
        rank = rank1 + rank2
        p = p1 + p2
        
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for i in range(n*m):
            for r in range(rank1):
                U[i][r] = U1[i][r]
            for r in range(rank2):
                U[i][rank1+r] = U2[i][r]
                
        for j in range(m):
            for k in range(p1):
                idx_v = j*p + k
                for r in range(rank1):
                    V[idx_v][r] = V1[j*p1+k][r]
            for k in range(p2):
                idx_v = j*p + p1 + k
                for r in range(rank2):
                    V[idx_v][rank1+r] = V2[j*p2+k][r]
                    
        for k in range(p1):
            for i in range(n):
                idx_w = k*n + i
                for r in range(rank1):
                    W[idx_w][r] = W1[k*n+i][r]
        for k in range(p2):
            for i in range(n):
                idx_w = (p1+k)*n + i
                for r in range(rank2):
                    W[idx_w][rank1+r] = W2[k*n+i][r]
                    
        return U, V, W

    # Recursive helper that actually works for p-splitting
    def get_decomp_p_split(n, m, p):
        if p == 1:
            return trivial_decomp(n, m, 1)
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        p1 = p // 2
        p2 = p - p1
        d1 = get_decomp_p_split(n, m, p1)
        d2 = get_decomp_p_split(n, m, p2)
        return combine_split_p(n, m, p1, p2, d1, d2)

    U, V, W = get_decomp_p_split(n, m, p)
    return {"U": U, "V": V, "W": W}
