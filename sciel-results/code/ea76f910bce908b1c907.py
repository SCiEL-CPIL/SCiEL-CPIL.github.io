"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin

    def schoolbook(n, m, p):
        """Return U, V, W for n x m * m x p using schoolbook (rank = n*m*p)."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    def strassen_222():
        """Return U, V, W for 2 x 2 * 2 x 2 using Strassen (rank = 7)."""
        # A is 2x2: indices A[i,j] -> row i*2+j in U
        # B is 2x2: indices B[j,k] -> row j*2+k in V
        # C is 2x2: indices C[i,k] -> row k*2+i in W (trace convention)
        #
        # Strassen products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A11+A01)(B00+B10)
        # M7 = (A01+A11)(B01+B11)
        #
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M7
        #
        # U rows: A00=0, A01=1, A10=2, A11=3
        # V rows: B00=0, B01=1, B10=2, B11=3
        # W rows: C00=0, C10=1, C01=2, C11=3  (k*2+i: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3)
        
        rank = 7
        U = [
            [1, 0, 1, 0, 1, 0, 0],  # A00: in M1, M3, M5
            [0, 0, 0, 0, 0, 1, 1],  # A01: in M6, M7
            [0, 1, 0, 0, 1, 0, 0],  # A10: in M2, M5
            [1, 1, 0, 1, 0, 1, 1],  # A11: in M1, M2, M4, M6, M7
        ]
        V = [
            [1, 1, 0, 1, 0, 1, 0],  # B00: in M1, M2, M4, M6
            [0, 0, 0, 1, 0, 0, 1],  # B01: in M4, M7
            [0, 0, 1, 0, 0, 1, 0],  # B10: in M3, M6
            [1, 0, 1, 0, 1, 0, 1],  # B11: in M1, M3, M5, M7
        ]
        # W rows: C00, C10, C01, C11
        # C00 = M1 - M4 + M6 - M7
        # C10 = M2 + M5
        # C01 = M3 + M4
        # C11 = M1 - M2 + M3 + M7
        W = [
            [1, 0, 0, -1, 0, 1, -1],  # C00
            [0, 1, 0, 0, 1, 0, 0],    # C10
            [0, 0, 1, 1, 0, 0, 0],    # C01
            [1, -1, 1, 0, 0, 0, 1],   # C11
        ]
        return U, V, W

    def combine_split_n(n1, n2, m, p, U1, V1, W1, U2, V2, W2):
        """Combine two decompositions split along n dimension."""
        rank1 = len(U1[0]) if U1 else 0
        rank2 = len(U2[0]) if U2 else 0
        new_rank = rank1 + rank2
        n = n1 + n2
        
        new_U = [[0.0] * new_rank for _ in range(n * m)]
        for i in range(n1):
            for j in range(m):
                new_U[i * m + j] = [x + 0.0 for x in U1[i * m + j]] + [0.0] * rank2
        for i in range(n1, n):
            for j in range(m):
                new_U[i * m + j] = [0.0] * rank1 + [x + 0.0 for x in U2[(i - n1) * m + j]]
        
        new_V = [[0.0] * new_rank for _ in range(m * p)]
        for j in range(m):
            for k in range(p):
                new_V[j * p + k] = [x + 0.0 for x in V1[j * p + k]] + [0.0] * rank2
        
        new_W = [[0.0] * new_rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n1):
                new_W[k * n + i] = [x + 0.0 for x in W1[k * n1 + i]] + [0.0] * rank2
            for i in range(n1, n):
                new_W[k * n + i] = [0.0] * rank1 + [x + 0.0 for x in W2[k * n2 + (i - n1)]]
        
        return new_U, new_V, new_W

    def combine_split_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2):
        """Combine two decompositions split along m dimension."""
        rank1 = len(U1[0]) if U1 else 0
        rank2 = len(U2[0]) if U2 else 0
        new_rank = rank1 + rank2
        m = m1 + m2
        
        new_U = [[0.0] * new_rank for _ in range(n * m)]
        for i in range(n):
            for j in range(m1):
                new_U[i * m + j] = [x + 0.0 for x in U1[i * m1 + j]] + [0.0] * rank2
            for j in range(m1, m):
                new_U[i * m + j] = [0.0] * rank1 + [x + 0.0 for x in U2[i * m2 + (j - m1)]]
        
        new_V = [[0.0] * new_rank for _ in range(m * p)]
        for j in range(m1):
            for k in range(p):
                new_V[j * p + k] = [x + 0.0 for x in V1[j * p + k]] + [0.0] * rank2
        for j in range(m1, m):
            for k in range(p):
                new_V[j * p + k] = [0.0] * rank1 + [x + 0.0 for x in V2[(j - m1) * p + k]]
        
        new_W = [[0.0] * new_rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                new_W[k * n + i] = [x + 0.0 for x in W1[k * n + i]] + [0.0] * rank2
        
        return new_U, new_V, new_W

    def combine_split_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
        """Combine two decompositions split along p dimension."""
        rank1 = len(U1[0]) if U1 else 0
        rank2 = len(U2[0]) if U2 else 0
        new_rank = rank1 + rank2
        p = p1 + p2
        
        new_U = [[0.0] * new_rank for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                new_U[i * m + j] = [x + 0.0 for x in U1[i * m + j]] + [0.0] * rank2
        
        new_V = [[0.0] * new_rank for _ in range(m * p)]
        for j in range(m):
            for k in range(p1):
                new_V[j * p + k] = [x + 0.0 for x in V1[j * p1 + k]] + [0.0] * rank2
            for k in range(p1, p):
                new_V[j * p + k] = [0.0] * rank1 + [x + 0.0 for x in V2[j * p2 + (k - p1)]]
        
        new_W = [[0.0] * new_rank for _ in range(p * n)]
        for k in range(p1):
            for i in range(n):
                new_W[k * n + i] = [x + 0.0 for x in W1[k * n + i]] + [0.0] * rank2
        for k in range(p1, p):
            for i in range(n):
                new_W[k * n + i] = [0.0] * rank1 + [x + 0.0 for x in W2[(k - p1) * n + i]]
        
        return new_U, new_V, new_W

    memo = {}

    def decompose(n, m, p):
        if (n, m, p) in memo:
            return memo[(n, m, p)]
        
        # Base cases
        if n == 1 and m == 1 and p == 1:
            result = schoolbook(1, 1, 1)
            memo[(1, 1, 1)] = result
            return result
        
        if n == 2 and m == 2 and p == 2:
            result = strassen_222()
            memo[(2, 2, 2)] = result
            return result
        
        # If any dimension is 1, just use schoolbook
        if n == 1 or m == 1 or p == 1:
            result = schoolbook(n, m, p)
            memo[(n, m, p)] = result
            return result
        
        best_rank = n * m * p
        best_U, best_V, best_W = None, None, None
        
        # Try splitting n
        if n >= 2 and time.time() < deadline:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = decompose(n1, m, p)
                U2, V2, W2 = decompose(n2, m, p)
                
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                new_rank = rank1 + rank2
                
                if new_rank < best_rank:
                    new_U, new_V, new_W = combine_split_n(n1, n2, m, p, U1, V1, W1, U2, V2, W2)
                    best_rank = new_rank
                    best_U, best_V, new_W = new_U, new_V, new_W
        
        # Try splitting m
        if m >= 2 and time.time() < deadline:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = decompose(n, m1, p)
                U2, V2, W2 = decompose(n, m2, p)
                
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                new_rank = rank1 + rank2
                
                if new_rank < best_rank:
                    new_U, new_V, new_W = combine_split_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2)
                    best_rank = new_rank
                    best_U, new_V, new_W = new_U, new_V, new_W
        
        # Try splitting p
        if p >= 2 and time.time() < deadline:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = decompose(n, m, p1)
                U2, V2, W2 = decompose(n, m, p2)
                
                rank1 = len(U1[0]) if U1 else 0
                rank2 = len(U2[0]) if U2 else 0
                new_rank = rank1 + rank2
                
                if new_rank < best_rank:
                    new_U, new_V, new_W = combine_split_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2)
                    best_rank = new_rank
                    best_U, new_V, new_W = new_U, new_V, new_W
        
        # If no split improved, use schoolbook
        if best_U is None:
            result = schoolbook(n, m, p)
            memo[(n, m, p)] = result
            return result
        
        result = (best_U, best_V, best_W)
        memo[(n, m, p)] = result
        return result

    U, V, W = decompose(n, m, p)
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Check tensor identity: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # The multiplication tensor T should have T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k]
        # Actually, the tensor for matrix multiplication has entries:
        # T[i,j,k] = delta_{j,j} = 1 for all valid (i,j,k)
        # Wait, no. The matrix multiplication tensor is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index)
        # Let me re-check: the tensor is defined such that
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k
        # because the (i,j,k) entry of the multiplication tensor is 1.
        
        # Actually, let's just check with a random example
        import random
        random.seed(42)
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # True product
        C_true = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_true[i][k] += A[i][j] * B[j][k]
        
        # Product from decomposition
        C_dec = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += 0  # We'll compute this differently
                # C_dec[i][k] = sum_j sum_r A[i,j] U[i*m+j, r] * (sum_j' B[j',k] V[j'*p+k, r]) * W[k*n+i, r]
                # This isn't quite right. Let me think more carefully.
                # The decomposition says: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j' B[j',k] V[j'*p+k, r]) * W[k*n+i, r]
                # Wait, that's not right either. The standard form is:
                # C[i,k] = sum_r U[i*m+*, r] * ... 
                # Actually, the tensor decomposition is:
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # And T[i,j,k] = 1 for all valid (i,j,k).
                # So the check is: for all i, j, k:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
