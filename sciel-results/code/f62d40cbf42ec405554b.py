"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s guard for safety
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: Use rank-5 decomposition for 3x3 matrix multiplication (Strassen-like)
    # The 2x3x9 multiplication can be viewed as:
    # - 2 "rows" of the first matrix (n=2)
    # - 3 "columns" of first / 3 "rows" of second (m=3)  
    # - 9 "columns" of second (p=9)
    # 
    # We can decompose the 9 columns into 3 groups of 3, giving us 3 independent 3x3 multiplications
    # each with rank 5 (using a rank-5 algorithm for 3x3)
    # Total rank: 2 * 3 * 5 = 30 (vs schoolbook 54)
    #
    # Actually, let me think more carefully.
    # Matrix A is 2x3, Matrix B is 3x9.
    # C = A * B is 2x9.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # If we split B into 3 blocks of 3 columns: B = [B1, B2, B3] where each Bq is 3x3
    # Then C = [A*B1, A*B2, A*B3]
    # Each A*Bq is a 2x3 times 3x3 = 2x3 multiplication
    #
    # For 2x3 * 3x3: we can use a rank-5 decomposition for the 3x3 core
    # But actually, 2x3 * 3x3 has tensor dimension (2,3,3) with schoolbook rank 18
    # 
    # Let me use a different approach: directly construct a low-rank decomposition
    # using the known rank-5 for 3x3 and outer products for the n=2 dimension.
    
    # Rank-5 decomposition for 3x3 matrix multiplication (Strassen-like)
    # This uses 5 multiplications instead of 9
    # The decomposition is:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # That's 7 for Strassen. For rank 5, we need a different scheme.
    
    # Actually, the minimum known rank for 3x3 is 19 (not 5). Let me reconsider.
    # The rank of 3x3 matrix multiplication is at least 19 (lower bound) and at most 23 (upper bound).
    # So rank-5 is not possible for 3x3.
    
    # Let me use a different strategy: exploit the specific dimensions.
    # For 2x3 * 3x9, we can potentially use a lower rank than 54.
    
    # Let me try a direct construction using bilinear decomposition.
    # We need to find matrices U (6 x r), V (27 x r), W (18 x r) such that
    # for all i in [0,2), j in [0,3), k in [0,9):
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #   = sum_{r} U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r]
    
    # Start with schoolbook rank 54 and try to reduce.
    # Use a greedy/iterative approach to find a lower rank decomposition.
    
    # First, let's construct the schoolbook as baseline
    rank_schoolbook = n * m * p  # 54
    
    # Try to find a lower rank using numerical optimization
    # We'll use the tensor representation and try to reduce rank
    
    # Create the tensor T with T[i,j,k] = 1 if this corresponds to a valid multiplication term
    # Actually, the matrix multiplication tensor has T[i,j,k] = delta_{j,j} (always 1 for valid indices)
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the middle dimension)
    
    # Let me use a different approach: start from schoolbook and try to merge terms
    
    # Schoolbook: each term (i,j,k) gives U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # Can we combine multiple terms into one rank-1 component?
    
    # A rank-1 component (u, v, w) contributes u[i*m+j] * v[j*p+k] * w[k*n+i] to T[i,j,k]
    # For this to match the identity in j, we need:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = delta_{j,j} = 1 (for all i,k)
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 (it's just 1 for all valid indices, since C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Actually, the tensor T satisfies: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
    # where T[i,j,k,r] is the r-th component. For schoolbook, T[i,j,k,r] = 1 if r corresponds to (i,j,k).
    
    # Let me just try to verify if we can do better than 54.
    # The lower bound for (2,3,9) matrix multiplication rank is not well-known.
    # Let me try a numerical search for a lower rank decomposition.
    
    # Use a simple approach: start with random initialization and try to optimize
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition exactly represents matrix multiplication."""
        r = len(U[0]) if U else 0
        # Check: for all i, j, k, the sum should give the correct coefficient
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Our decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
        # This is not quite right. Let me think again.
        
        # The standard tensor decomposition for matrix multiplication:
        # We have tensors U (n*m x r), V (m*p x r), W (p*n x r)
        # The decomposition satisfies: for all i, j, k:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # (since the "tensor" for matmul is all-ones in the appropriate sense)
        
        # Wait, that's not the right formulation either. Let me be precise.
        # The matrix multiplication tensor T is a 3-tensor with T[i,j,k] = 1 for all i,j,k.
        # A rank-r decomposition is: T[i,j,k] = sum_{r=1}^R U_r[i,j] * V_r[j,k] * W_r[k,i]
        # where U_r is n x m, V_r is m x p, W_r is p x n.
        
        # In our flattened form:
        # U is (n*m) x R, V is (m*p) x R, W is (p*n) x R
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For matrix multiplication, T[i,j,k] should be 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(r):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a decomposition of rank < 54
    # Use a structured approach based on the dimension factors
    
    # Key insight: p=9=3*3, m=3, n=2
    # We can view this as 2 independent (1,3,9) multiplications, or
    # as 3 independent (2,3,3) multiplications, etc.
    
    # Let me try: view B as a 3x3 block matrix (3 rows, 3 blocks of 3 columns)
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]] where each Bxy is 3x3
    # A is 2x3: A = [[a00, a01, a02], [a10, a11, a12]]
    # C = A*B is 2x9
    # C[i, 3*q + k] = sum_j A[i,j] * B[j, 3*q+k]
    
    # For fixed q (0,1,2), C[:, 3*q:3*q+3] = A * B[:, 3*q:3*q+3]
    # This is a (2,3,3) multiplication.
    
    # So we need 3 independent (2,3,3) decompositions.
    # The rank of (2,3,3) matrix multiplication: schoolbook is 18.
    # Can we do better? The lower bound is not tight, but let's try.
    
    # For (2,3,3): A is 2x3, B is 3x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can use the fact that n=2 to potentially reduce rank.
    
    # Let me try a direct construction for (2,3,3) with rank < 18.
    
    # Actually, let me just try to use a numerical method to find a lower rank decomposition
    # for the full (2,3,9) problem.
    
    # Approach: Use alternating least squares or a similar method to minimize the rank
    # subject to the constraint that the decomposition is exact.
    
    # Since we need an EXACT decomposition (integer/half-integer entries), let's try
    # a constructive approach.
    
    # For (2,3,9), let me try to exploit the structure more directly.
    # 
    # Idea: Use the fact that we can share computation across the 9 output columns.
    # 
    # Let me try a rank-30 construction (2 * 3 * 5, using rank-5 for 3x3 blocks)
    # But as noted, rank-5 for 3x3 doesn't exist.
    
    # Let me try a different decomposition strategy.
    # 
    # For (2,3,9): 
    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # 
    # We can write:
    # C[0,k] = (A[0,0]+A[0,1])*B[0,k] + (A[0,1]+A[0,2])*B[1,k] + (A[0,2]+A[0,0])*B[2,k]
    #          - A[0,1]*B[0,k] - A[0,2]*B[1,k] - A[0,0]*B[2,k]
    # This doesn't obviously help.
    
    # Let me try a systematic search for a lower rank using integer/half-integer coefficients.
    
    # Given the time constraint, let me try a few structured decompositions.
    
    # Attempt 1: Rank 36 (2/3 of schoolbook)
    # Try to pair up some of the schoolbook terms.
    
    # Actually, let me just return the schoolbook for now if I can't find something better
    # in the time available. But first, let me try one more idea.
    
    # Idea: Use the fact that n=2 is small.
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    # 
    # Let D[j,k] = B[j,k] (shared)
    # Let E[j] = (A[0,j], A[1,j]) for j=0,1,2
    # 
    # We can write:
    # [C[0,k], C[1,k]]^T = sum_j E[j] * B[j,k]
    # 
    # This is a (2,3,9) problem. The rank is at most 2*3*9 = 54 (schoolbook).
    # 
    # Can we do better? Let's try to use a bilinear decomposition.
    
    # For now, let me try a simple improvement: check if we can reduce rank by 1 or 2
    # using a perturbation of the schoolbook.
    
    # Given the complexity, let me just return the schoolbook decomposition.
    # The rank is 54.
    
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
# EVOLVE-BLOCK-END
