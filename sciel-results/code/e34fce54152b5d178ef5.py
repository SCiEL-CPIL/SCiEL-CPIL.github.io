import time
import sys

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Memoization for get_decomp
    memo = {}
    
    def get_decomp(n, m, p):
        if (n, m, p) in memo:
            return memo[(n, m, p)]
            
        # Base cases
        if n == 2 and m == 2 and p == 2:
            res = strassen_2x2x2()
            memo[(n, m, p)] = res
            return res
            
        if n == 1 or m == 1 or p == 1:
            res = schoolbook(n, m, p)
            memo[(n, m, p)] = res
            return res
            
        best_rank = float('inf')
        best_decomp = None
        
        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                r1 = len(U1[0])
                r2 = len(U2[0])
                rank = r1 + r2
                if rank < best_rank:
                    best_rank = rank
                    # Combine
                    U, V, W = combine_split_n(n, m, p, n1, U1, V1, W1, U2, V2, W2)
                    best_decomp = (U, V, W)
                    
        # Try splitting m
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                r1 = len(U1[0])
                r2 = len(U2[0])
                rank = r1 + r2
                if rank < best_rank:
                    best_rank = rank
                    U, V, W = combine_split_m(n, m, p, m1, U1, V1, W1, U2, V2, W2)
                    best_decomp = (U, V, W)
                    
        # Try splitting p
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                r1 = len(U1[0])
                r2 = len(U2[0])
                rank = r1 + r2
                if rank < best_rank:
                    best_rank = rank
                    U, V, W = combine_split_p(n, m, p, p1, U1, V1, W1, U2, V2, W2)
                    best_decomp = (U, V, W)
                    
        if best_decomp is None:
            # Fallback to schoolbook if no split worked (should not happen for n,m,p>1)
            best_decomp = schoolbook(n, m, p)
            
        memo[(n, m, p)] = best_decomp
        return best_decomp

    def combine_split_n(n, m, p, n1, U1, V1, W1, U2, V2, W2):
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # U1 is (n1*m, r1), U2 is (n2*m, r2)
        # Global U index: i*m + j
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1+i)*m + j][r1 + c] = U2[i*m + j][c]
                    
        # V1 is (m*p, r1), V2 is (m*p, r2)
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
                for c in range(r2):
                    V[j*p + k][r1 + c] = V2[j*p + k][c]
                    
        # W1 is (p*n1, r1), W2 is (p*n2, r2)
        # Global W index: k*n + i
        for k in range(p):
            for i in range(n1):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n1 + i][c]
            for i in range(n2):
                for c in range(r2):
                    W[k*n + (n1+i)][r1 + c] = W2[k*n2 + i][c]
                    
        return U, V, W

    def combine_split_m(n, m, p, m1, U1, V1, W1, U2, V2, W2):
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # U1 is (n*m1, r1), U2 is (n*m2, r2)
        for i in range(n):
            for j in range(m1):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m1 + j][c]
            for j in range(m2):
                for c in range(r2):
                    U[i*m + (m1+j)][r1 + c] = U2[i*m2 + j][c]
                    
        # V1 is (m1*p, r1), V2 is (m2*p, r2)
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p1 + k][c] # Wait, V1 index is j*p1+k? No, V1 is for m1.
                    # V1 shape (m1*p, r1). Index j*p + k? No, local index j*p1 + k?
                    # V1 is for M_{n, m1, p}. So V1 row index is j*p + k?
                    # No, V1 corresponds to B which is m1 x p.
                    # So V1 row index is j*p + k where j in [0, m1).
                    # Yes.
                    # But wait, in get_decomp(n, m1, p), the V matrix has shape (m1*p, rank).
                    # So V1[j*p + k] is correct?
                    # No, V1 is the factor for the sub-problem.
                    # The sub-problem B is m1 x p.
                    # So V1 rows are indexed by j*p + k?
                    # Yes, because the sub-problem has dimension p.
                    
        # Correction: In combine_split_m, V1 is from get_decomp(n, m1, p).
        # V1 shape is (m1*p, r1).
        # Global V shape is (m*p, total_rank).
        # Global row index for block 1 is j*p + k?
        # No, global B is m x p.
        # Block 1 B is m1 x p.
        # So global row index is j*p + k?
        # Yes, because j ranges 0..m1-1, and k ranges 0..p-1.
        # The mapping is direct for j < m1.
        
        # Let's re-verify V indexing.
        # V1[j*p + k] -- Wait.
        # If m1=2, p=3. V1 rows: 0,1,2,3,4,5.
        # j=0, k=0 -> 0. j=0, k=1 -> 1. j=0, k=2 -> 2.
        # j=1, k=0 -> 3.
        # Global V rows for m=5, p=3.
        # Block 1 (m1=2): rows 0..5.
        # j=0, k=0 -> 0*3+0=0.
        # j=1, k=2 -> 1*3+2=5.
        # This matches V1[j*p + k].
        
        for j in range(m1):
            for k in range(p):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p + k][c]
        for j in range(m2):
            for k in range(p):
                for c in range(r2):
                    V[(m1+j)*p + k][r1 + c] = V2[j*p + k][c]
                    
        # W1 is (p*n, r1), W2 is (p*n, r2)
        for k in range(p):
            for i in range(n):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
                for c in range(r2):
                    W[k*n + i][r1 + c] = W2[k*n + i][c]
                    
        return U, V, W

    def combine_split_p(n, m, p, p1, U1, V1, W1, U2, V2, W2):
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        # U1 is (n*m, r1), U2 is (n*m, r2)
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i*m + j][c] = U1[i*m + j][c]
                for c in range(r2):
                    U[i*m + j][r1 + c] = U2[i*m + j][c]
                    
        # V1 is (m*p1, r1), V2 is (m*p2, r2)
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j*p + k][c] = V1[j*p1 + k][c]
            for k in range(p2):
                for c in range(r2):
                    V[j*p + (p1+k)][r1 + c] = V2[j*p2 + k][c]
                    
        # W1 is (p1*n, r1), W2 is (p2*n, r2)
        for k in range(p1):
            for i in range(n):
                for c in range(r1):
                    W[k*n + i][c] = W1[k*n + i][c]
        for k in range(p2):
            for i in range(n):
                for c in range(r2):
                    W[(p1+k)*n + i][r1 + c] = W2[k*n + i][c]
                    
        return U, V, W

    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
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
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][c] = 1
                    V[j*p + k][c] = 1
                    W[k*n + i][c] = 1
                    c += 1
        return U, V, W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
