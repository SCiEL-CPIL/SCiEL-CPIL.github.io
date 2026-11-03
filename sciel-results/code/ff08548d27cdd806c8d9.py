import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s budget out of 120s limit

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # This is the tensor we need to decompose.
    # A rank-r decomposition means: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook rank = n*m*p = 63
    # Let's try to find a lower rank decomposition.
    
    # Approach: Use the fact that M(3,3,7) can be decomposed using 
    # Kronecker products and known results.
    
    # Known: M(2,2,2) has rank 7 (Strassen)
    # M(3,3,3) has rank at most 23 (Landsberg) or known constructions
    # M(3,3,7) = M(3,3,3) * M(1,1,7) + ... no, that's not how it works directly.
    
    # Better approach: Direct construction using bilinear forms.
    # The matrix product (AB)_{ik} = sum_j A_{ij} B_{jk}
    # We need to decompose this into rank-1 terms: u_{ij} * v_{jk} * w_{ik}
    
    # Let's use a systematic approach:
    # 1. Start with schoolbook (rank 63)
    # 2. Try to use Strassen-like recurrences for sub-blocks
    
    # Strategy: Split 3x3x7 into blocks
    # A is 3x3, B is 3x7
    # We can think of this as:
    # Row 0 of A (1x3) * B (3x7) -> 1x7, rank at most 7
    # Row 1 of A (1x3) * B (3x7) -> 1x7, rank at most 7  
    # Row 2 of A (1x3) * B (3x7) -> 1x7, rank at most 7
    # Total: 21 (this is actually better than schoolbook 63!)
    
    # Wait, that's just the row-by-row computation. Let me think more carefully.
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # For each pair (i,k), we need to compute sum_j A_{ij} B_{jk}
    # This is an inner product of row i of A and column k of B.
    
    # The schoolbook decomposition has one term for each (i,j,k) triple, giving rank 63.
    # But we can do better by noting that for fixed i and k, the sum over j can be
    # computed more efficiently.
    
    # Actually, the tensor M(3,3,7) has been studied. Known upper bounds:
    # - Schoolbook: 63
    # - Using fast matrix multiplication ideas: potentially lower
    
    # Let me try a direct optimization approach.
    # We want to find U (9 x r), V (21 x r), W (21 x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if and only if the 
    # indices match the multiplication pattern, i.e., for each (i,j,k), 
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to 1 over c for the 
    # correct (i,j,k) and 0 otherwise.
    
    # Actually, the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other combinations of indices (i', j', k') where the pattern doesn't match:
    #   sum_c U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c] = 0
    
    # Wait, that's not right either. The tensor T has entries T[a, b, c] where:
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a, b, c] = 1 if j == j' and k == k' and i == i', else 0.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # T[a, b, c] = 0 for all other (a,b,c)
    
    # The rank-r decomposition requires:
    # T[a,b,c] = sum_{r=1}^R U[a,r] * V[b,r] * W[c,r]
    
    # Let me try to find a small rank decomposition using numerical optimization.
    
    def try_rank(R, max_time=80):
        """Try to find a rank-R decomposition using random initialization and gradient descent."""
        start_time = time.time()
        np.random.seed(42)
        
        # Initialize U, V, W with small random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # Precompute the target tensor entries
        # For each (i,j,k), we need T[i*m+j, j*p+k, k*n+i] = 1
        target_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_indices.append((i*m+j, j*p+k, k*n+i))
        
        # Loss function: sum over target entries of (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
        #              + sum over non-target entries of (sum_r U[a,r]*V[b,r]*W[c,r])^2
        
        # For efficiency, we only compute the target entries explicitly and 
        # assume non-target entries are 0 (which is most of the tensor)
        
        # Actually, the non-target entries are the vast majority. We need to handle them.
        # Total tensor size: 9 * 21 * 21 = 3969
        # Target entries: 63
        # Non-target: 3906
        
        # This is too large for direct computation. Let's use a smarter approach.
        
        # Alternative: Use the fact that the tensor has a specific structure.
        # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        
        # Let me try a different approach: use known results for small matrix multiplication.
        
        # For M(3,3,7), one can use the following construction:
        # Decompose B into 3 blocks of 3x7? No, B is 3x7.
        
        # Let me try: split A into rows, and use the fact that each row of A times B
        # gives a 1x7 vector, which can be computed with rank at most 7 (schoolbook for 1x3 times 3x7).
        # But that's 3*7 = 21, which is already better than 63.
        
        # Wait, that's not a valid tensor decomposition in the usual sense because 
        # the "multiplications" would be A_{ij} * B_{jk} which are rank-1, but we need
        # to express the sum as a product of three vectors.
        
        # Let me think about this differently. The rank of M(n,m,p) is the minimum R such that
        # there exist U in R^{n*m x R}, V in R^{m*p x R}, W in R^{p*n x R} with
        # T = U vee V vee W (tensor product decomposition).
        
        # For M(1,m,p): T[i*m+j, j*p+k, k*1+i] = delta_{i,0} * delta_{j,j'} * delta_{k,k'}
        # Actually for n=1: T[j, j*p+k, k] = 1 for all j,k
        # Rank of M(1,m,p) = m*p (schoolbook, and this is optimal for n=1)
        
        # So M(1,3,7) has rank 21.
        # M(3,3,7) >= M(1,3,7) in some sense? Not directly.
        
        # Known result: M(3,3,3) has rank at most 23 (Landsberg 1983), and it's known that
        # M(3,3,3) has rank at least 19.
        
        # For M(3,3,7), I don't know the exact rank. Let me try to construct a decomposition.
        
        # Approach: Use the "border rank" or explicit constructions.
        
        # Let me try a simple block decomposition:
        # A = [A11 A12 A13]  (3 rows, each 1x3)
        # B = [B1; B2; B3]   (3 blocks, each 3x7)
        
        # Row i of AB = sum_j A_{ij} * B_j (where B_j is row j of B, 1x7)
        # So (AB)_{ik} = sum_j A_{ij} B_{jk}
        
        # This is just the standard formula. The schoolbook decomposition has one term per (i,j,k).
        
        # To reduce rank, we need to find linear combinations that compute multiple 
        # products simultaneously.
        
        # For 2x2 matrix multiplication, Strassen's algorithm reduces 8 to 7 multiplications.
        # Can we apply Strassen-like ideas to 3x3 x 3x7?
        
        # Let me try to construct an explicit decomposition of rank less than 63.
        
        # Idea: Group by the j index. For each j, we have n*p = 21 terms of the form
        # A_{ij} * B_{jk}. If we could compute these more efficiently...
        
        # Actually, let me just try to return a known good decomposition.
        # The schoolbook is rank 63. Let me see if I can find something better.
        
        # One approach: use the fact that M(3,3,7) can be embedded in M(4,4,8) or similar,
        # but that doesn't directly help.
        
        # Let me try a numerical search for a small rank.
        
        # Given time constraints, let me try rank 21 first (the obvious "row-wise" bound).
        
        # Actually, I realize that the row-wise argument gives rank 21 but it's not a 
        # valid tensor decomposition in the rank-1 sense. Each "row computation" involves
        # 3 multiplications A_{ij} * B_{jk} for j=0,1,2, and these are separate rank-1 terms.
        # So the total is still 3*3*7 = 63 rank-1 terms.
        
        # To get below 63, we need genuine sharing.
        
        # Let me try to use Strassen's algorithm on a 2x2 subblock.
        # Consider the top-left 2x2 block of A and the corresponding 2x7 block of B.
        # The product is 2x7, which can be computed as:
        # [A00 A01; A10 A11] * [B00..B06; B10..B16]
        # = [A00*B00..B06 + A01*B10..B16; A10*B00..B06 + A11*B10..B16]
        
        # Using Strassen for the 2x2 part, we need 7 "scalar" multiplications for the 2x2 case,
        # but here the second factor is 2x7, so each "scalar" multiplication becomes a vector.
        
        # Specifically:
        # M1 = (A00+A11) * (B00..B06 + B10..B16)  -- this is a 1x7 vector
        # M2 = (A10+A11) * (B00..B06)              -- 1x7
        # M3 = A00 * (B10..B16 - B00..B06)         -- 1x7
        # M4 = A11 * (B00..B06 - B10..B16)         -- 1x7
        # M5 = (A00+A01) * (B10..B16)              -- 1x7
        # M6 = (A10-A00) * (B00..B06 + B01..B07)   -- wait, B is 2x7, so B00..B06 is row 0
        # M7 = (A01-A11) * (B10..B16 + B00..B06)   
        
        # Hmm, this is getting complicated. Each "multiplication" is actually a dot product
        # of a scalar with a vector, which is 7 scalar multiplications.
        
        # So 7 Strassen steps * 7 = 49 scalar multiplications for the 2x2 x 2x7 block,
        # compared to 8 * 7 = 56 for schoolbook.
        
        # Then we add the remaining parts:
        # Row 2 of A (1x3) * B (3x7): 3*7 = 21
        # Column 2 of A... wait, let me be more careful.
        
        # A is 3x3, B is 3x7.
        # Split A into [A_top; A_bot] where A_top is 2x3, A_bot is 1x3.
        # Split B into [B_left; B_right] where B_left is 2x7, B_right is 1x7.
        
        # AB = A_top * B_left + A_top * B_right + A_bot * B_left + A_bot * B_right
        #    = (2x7) + (2x7) + (1x7) + (1x7)
        
        # A_top * B_left: 2x3 * 3x7, use Strassen: 7 * 7 = 49
        # A_top * B_right: 2x3 * 3x1, use Strassen: 7 * 1 = 7
        # A_bot * B_left: 1x3 * 3x7, schoolbook: 3 * 7 = 21
        # A_bot * B_right: 1x3 * 3x1, schoolbook: 3 * 1 = 3
        # Total: 49 + 7 + 21 + 3 = 80
        
        # That's worse than 63! Strassen doesn't help when one dimension is 1.
        
        # Let me try a different split.
        
        # Actually, for M(2,3,7):
        # Schoolbook: 2*3*7 = 42
        # With Strassen on the 2x2 part: 
        #   A is 2x3, B is 3x7
        #   Split A = [A11 A12; A21 A22] where A11,A21 are 1x1, A12,A22 are 1x1, and there's a third column
        #   This is messy.
        
        # Let me just try to numerically search for a low-rank decomposition.
        
        # Given the time constraint, let me try a few ranks and see if any work.
        
        def verify(U, V, W):
            """Verify that U, V, W form a valid decomposition."""
            R = U.shape[1]
            # Check all target entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val - 1) > 1e-8:
                            return False
            # Check some non-target entries
            # T[a,b,c] should be 0 for non-matching indices
            # a = i*m+j, b = j'*p+k, c = k'*n+i'
            # Non-target when j != j' or k != k' or i != i'
            for _ in range(1000):
                i = np.random.randint(n)
                j = np.random.randint(m)
                kp = np.random.randint(p)
                i2 = np.random.randint(n)
                j2 = np.random.randint(m)
                k2 = np.random.randint(p)
                if i == i2 and j == j2 and kp == k2:
                    continue
                a = i * m + j
                b = j2 * p + kp
                c = k2 * n + i2
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val) > 1e-6:
                    return False
            return True
        
        # Try various ranks
        for R in range(63, 20, -1):
            if time.time() > deadline:
                break
            # Try a few random initializations
            for attempt in range(5):
                if time.time() > deadline:
                    break
                np.random.seed(R * 100 + attempt)
                U = np.random.randn(n*m, R) * 0.5
                V = np.random.randn(m*p, R) * 0.5
                W = np.random.randn(p*n, R) * 0.5
                
                # Simple gradient descent
                lr = 0.01
                for step in range(500):
                    if time.time() > deadline:
                        break
                    # Compute gradients
                    # dL/dU[a] = sum over (b,c) in support: 2*(pred-1)*V[b]*W[c]
                    # This is expensive. Let's just check if the current solution is close.
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.sum(U[a] * V[b] * W[c])
                                err += (val - 1) ** 2
                    if err < 1e-6:
                        break
                    # Simple SGD on target entries
                    i = np.random.randint(n)
                    j = np.random.randint(m)
                    k = np.random.randint(p)
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    grad = (val - 1)
                    U[a] -= lr * grad * V[b] * W[c]
                    V[b] -= lr * grad * U[a] * W[c]
                    W[c] -= lr * grad * U[a] * V[b]
                
                if verify(U, V, W):
                    # Snap to rationals
                    U_frac = [[Fraction(int(round(x * 10000)), 10000) for x in U[r]] for r in range(U.shape[0])]
                    V_frac = [[Fraction(int(round(x * 10000)), 10000) for x in V[r]] for r in range(V.shape[0])]
                    W_frac = [[Fraction(int(round(x * 10000)), 10000) for x in W[r]] for r in range(W.shape[0])]
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    result = try_rank(63)
    return result
# EVOLVE-BLOCK-END
