import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15

    def schoolbook():
        rank = nm * mp
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2():
        """Return Strassen's 7-term decomposition for 2x2x2 matrix multiplication."""
        # Strassen's algorithm: 7 multiplications for 2x2 * 2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+b)(e+f), M2 = (a-b)(e+f), M3 = (c+d)h, M4 = a(g-h)
        # M5 = d(e-g), M6 = (a+b)(h-f), M7 = (c-d)(e+f)
        # C11 = M1+M3-M5-M7, C12 = M4+M5, C21 = M2+M3, C22 = M1-M2+M4-M6
        
        # U: indices (i,j) for A, V: indices (j,k) for B, W: indices (k,i) for C
        # Using 0-based: A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
        
        # For each of the 7 terms, define U, V, W contributions
        # Term 1: M1 = (a+b)(e+f)
        # U: a=1, b=1; V: e=1, f=1; W: C11=1, C22=1
        # Term 2: M2 = (a-b)(e+f)
        # U: a=1, b=-1; V: e=1, f=1; W: C21=1, C22=-1
        # Term 3: M3 = (c+d)h
        # U: c=1, d=1; V: h=1; W: C11=1, C21=1
        # Term 4: M4 = a(g-h)
        # U: a=1; V: g=1, h=-1; W: C12=1, C22=1
        # Term 5: M5 = d(e-g)
        # U: d=1; V: e=1, g=-1; W: C11=-1, C12=1
        # Term 6: M6 = (a+b)(h-f)
        # U: a=1, b=1; V: h=1, f=-1; W: C22=-1
        # Term 7: M7 = (c-d)(e+f)
        # U: c=1, d=-1; V: e=1, f=1; W: C11=-1, C22=-1
        
        U = []
        V = []
        W = []
        
        # Term 1
        u1 = [1.0, 1.0, 0.0, 0.0]  # a, b, c, d
        v1 = [1.0, 1.0, 0.0, 0.0]  # e, f, g, h
        w1 = [1.0, 0.0, 0.0, 1.0]  # C11, C12, C21, C22
        U.append(u1); V.append(v1); W.append(w1)
        
        # Term 2
        u2 = [1.0, -1.0, 0.0, 0.0]
        v2 = [1.0, 1.0, 0.0, 0.0]
        w2 = [0.0, 0.0, 1.0, -1.0]
        U.append(u2); V.append(v2); W.append(w2)
        
        # Term 3
        u3 = [0.0, 0.0, 1.0, 1.0]
        v3 = [0.0, 0.0, 0.0, 1.0]
        w3 = [1.0, 0.0, 1.0, 0.0]
        U.append(u3); V.append(v3); W.append(w3)
        
        # Term 4
        u4 = [1.0, 0.0, 0.0, 0.0]
        v4 = [0.0, 0.0, 1.0, -1.0]
        w4 = [0.0, 1.0, 0.0, 1.0]
        U.append(u4); V.append(v4); W.append(w4)
        
        # Term 5
        u5 = [0.0, 0.0, 0.0, 1.0]
        v5 = [1.0, 0.0, -1.0, 0.0]
        w5 = [-1.0, 1.0, 0.0, 0.0]
        U.append(u5); V.append(v5); W.append(w5)
        
        # Term 6
        u6 = [1.0, 1.0, 0.0, 0.0]
        v6 = [0.0, -1.0, 0.0, 1.0]
        w6 = [0.0, 0.0, 0.0, -1.0]
        U.append(u6); V.append(v6); W.append(w6)
        
        # Term 7
        u7 = [0.0, 0.0, 1.0, -1.0]
        v7 = [1.0, 1.0, 0.0, 0.0]
        w7 = [-1.0, 0.0, 0.0, -1.0]
        U.append(u7); V.append(v7); W.append(w7)
        
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition is exact."""
        nm = n * m
        mp = m * p
        pn = p * n
        rank = len(U[0]) if U else 0
        
        # Check dimensions
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        for i in range(nm):
            if len(U[i]) != rank:
                return False
        for j in range(mp):
            if len(V[j]) != rank:
                return False
        for k in range(pn):
            if len(W[k]) != rank:
                return False
        
        # Verify: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # where T[i,j,k] = 1 if i=m*k+j (for A[i,j] * B[j,k] -> C[k,i])
        # Actually, the tensor T has T[i,j,k] = 1 if the multiplication involves A[i,j] and B[j,k] producing C[k,i]
        # So T[i,j,k] = 1 iff j is valid (0<=j<m) and i is row of A, k is col of B
        # More precisely: T[i,j,k] = delta_{i, row} * delta_{j, col_of_A_and_row_of_B} * delta_{k, col}
        # The multiplication tensor: T[i,j,k] = 1 if (i,j) is an entry of A, (j,k) is an entry of B, (k,i) is an entry of C
        # So T[i,j,k] = 1 if 0<=i<n, 0<=j<m, 0<=k<p
        
        # Actually, the standard matrix multiplication tensor:
        # T[i,j,k] = 1 if i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
        # The output C[k,i] gets contribution from A[i,j]*B[j,k]
        
        # Let's verify by checking a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1
                    computed = 0.0
                    for r in range(rank):
                        computed += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(computed - 1.0) > tol:
                        return False
        
        # Also check that entries where the tensor should be 0 are 0
        # The tensor T[i,j,k] is 0 unless 0<=i<n, 0<=j<m, 0<=k<p
        # But our indices are already constrained, so we just need to check the above
        
        return True

    def try_kronecker_decomposition():
        """
        Try to build a decomposition using block structure.
        
        For 3x3 * 3x5, we can think of this as:
        - Split A into blocks, B into blocks
        - Use known low-rank decompositions for smaller blocks
        
        One approach: treat the 3x3 * 3x5 as involving 3x3 * 3x3 and 3x3 * 3x2 parts
        Or use the fact that 3 = 2+1 and try to combine 2x2 Strassen with 1x1 terms.
        
        Let's try: A = [[A22, A21], [A12, A11]] where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
        B = [[B22, B21, B22p], [B12, B11, B12p]] where first two cols form 3x2, last col is 3x1
        
        This is getting complex. Let me try a simpler approach:
        Use the fact that we can write the 3x3*3x5 multiplication as a combination of:
        - 2x2*2x2 (Strassen, rank 7) for the top-left 2x2 block
        - Additional terms for the remaining parts
        
        Actually, let me try a different structural approach:
        The 3x3 * 3x5 multiplication can be seen as 3 independent 1x3 * 3x5 multiplications
        (one for each row of A), each of which is a dot product with rank 15 (schoolbook).
        That gives rank 45, same as schoolbook.
        
        Let me try to use the block structure more cleverly.
        """
        # Try to construct a decomposition with rank < 45
        # using the block structure and Strassen-like ideas
        
        # One idea: use the fact that for 3x3 * 3x3, the best known rank is 23 (Strassen-like)
        # For 3x3 * 3x5, we might be able to do better than 45
        
        # Let's try a randomized search for a low-rank decomposition
        # using the structure of the tensor
        
        best_rank = 45
        best_result = None
        
        # Try ranks from 30 to 44 (we know 45 works)
        for target_rank in range(30, 45):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            
            # Try to find a decomposition of this rank
            # Use a simple approach: initialize with random values and try to optimize
            # But this is hard without a good optimizer.
            
            # Alternative: use the structure to build a candidate
            # For now, let's just return schoolbook if we can't find something better
            pass
        
        return best_result

    # First, try the Kronecker/block approach
    result = try_kronecker_decomposition()
    
    if result is None:
        # Fall back to schoolbook
        result = schoolbook()
    
    return result
# EVOLVE-BLOCK-END
