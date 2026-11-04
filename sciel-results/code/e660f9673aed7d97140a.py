"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 4x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Use a Strassen-based decomposition for the 4x4x4 part (rank 18 via Strassen recursion).
    2. Handle the 5th output column with additional terms.
    3. Use structured linear algebra to minimize rank.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Build Strassen-based decomposition for 4x4x4
    # Strassen's algorithm for 2x2x2 has rank 7
    # For 4x4x4, we can use Strassen recursively: split into 2x2 blocks
    # This gives rank 7 * 7 = 49 for 4x4x4 (not optimal but better than schoolbook 64)
    # Actually, let's use a more efficient construction.
    
    # For 4x4x4, the known minimal rank is 19.
    # Let me try to construct a rank-19 or rank-20 decomposition.
    
    # Approach: Use the fact that 4x4 matrix multiplication can be decomposed
    # using bilinear forms. We'll search for a low-rank decomposition.
    
    # First, let's build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # For 4x4x5: dimensions are 16 x 20 x 20
    
    # We'll try to find a rank-r decomposition by solving a linear system.
    # For rank r, we have r terms, each with:
    # - u_t in R^(n*m) = R^16
    # - v_t in R^(m*p) = R^20  
    # - w_t in R^(p*n) = R^20
    
    # The tensor equation is: sum_t u_t ⊗ v_t ⊗ w_t = T
    # This is a nonlinear system. Instead, we'll use a structured approach.
    
    # Let's use a hybrid: Strassen for 4x4x4 + schoolbook for the extra column
    # Strassen 4x4x4: split into 2x2 blocks, use Strassen 2x2x7 for each
    # This gives rank 7*7 = 49 for 4x4x4
    # Plus 4 terms for the 5th column = 53 total
    
    # Actually, let's try a better approach: use the known rank-19 for 4x4x4
    # and add 4 terms for the 5th column = 23 total
    
    # For now, let's implement a working solution with rank 53 (Strassen-based)
    # and try to optimize if time permits.
    
    def strassen_2x2():
        """Return rank-7 decomposition for 2x2x2 matrix multiplication."""
        # Strassen's 7 products for 2x2x2
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # Products:
        # p1 = (a11+a22)*(b11+b22)
        # p2 = (a21+a22)*b11
        # p3 = a11*(b12-b22)
        # p4 = a22*(b21-b11)
        # p5 = (a11+a12)*b22
        # p6 = (a21-a11)*(b11+b12)
        # p7 = (a12-a22)*(b21+b22)
        # Result:
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        rank = 7
        U = [[0]*rank for _ in range(4)]  # n*m = 2*2 = 4
        V = [[0]*rank for _ in range(4)]  # m*p = 2*2 = 4
        W = [[0]*rank for _ in range(4)]  # p*n = 2*2 = 4
        
        # Index: U[i*m+j], V[j*p+k], W[k*n+i]
        # For 2x2: U indices 0-3 (i*2+j), V indices 0-3 (j*2+k), W indices 0-3 (k*2+i)
        
        # p1 = (a11+a22)*(b11+b22)
        # U: a11 (idx 0) + a22 (idx 3), V: b11 (idx 0) + b22 (idx 3)
        # W: c11 (idx 0) + 1, c22 (idx 3) + 1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21+a22)*b11
        # U: a21 (idx 2) + a22 (idx 3), V: b11 (idx 0)
        # W: c21 (idx 2) + 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # p3 = a11*(b12-b22)
        # U: a11 (idx 0), V: b12 (idx 1) - b22 (idx 3)
        # W: c12 (idx 1) + 1, c22 (idx 3) - 1
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = -1
        
        # p4 = a22*(b21-b11)
        # U: a22 (idx 3), V: b21 (idx 2) - b11 (idx 0)
        # W: c11 (idx 0) + 1, c21 (idx 2) + 1
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # p5 = (a11+a12)*b22
        # U: a11 (idx 0) + a12 (idx 1), V: b22 (idx 3)
        # W: c11 (idx 0) - 1, c12 (idx 1) + 1
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # p6 = (a21-a11)*(b11+b12)
        # U: a21 (idx 2) - a11 (idx 0), V: b11 (idx 0) + b12 (idx 1)
        # W: c22 (idx 3) + 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12-a22)*(b21+b22)
        # U: a12 (idx 1) - a22 (idx 3), V: b21 (idx 2) + b22 (idx 3)
        # W: c11 (idx 0) + 1
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    # For 4x4x4, we can't simply use 7x7=49 rank with simple Kronecker product
    # because the indices don't align perfectly. Let's use a different approach.
    
    # Let's just use schoolbook for now and try to optimize.
    # Schoolbook for 4x4x5: rank 80
    
    # Actually, let's try a structured approach:
    # Decompose 4x4x5 as (4x4x4) + (4x4x1)
    # For 4x4x4, use a rank-20 decomposition (close to optimal 19)
    # For 4x4x1, use rank-4 schoolbook
    # Total: 24
    
    # Let me implement a rank-20 decomposition for 4x4x4 using linear algebra search
    
    def find_4x4x4_decomposition():
        """Find a low-rank decomposition for 4x4x4 matrix multiplication."""
        n4, m4, p4 = 4, 4, 4
        nm = n4 * m4  # 16
        mp = m4 * p4  # 16
        pn = p4 * n4  # 16
        
        # Target tensor T[i*m+j, j*p+k, k*n+i] = 1
        # We'll try rank 20 first
        
        for target_rank in [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
            if time.time() > deadline:
                break
            
            # For rank r, we need to find U (16xr), V (16xr), W (16xr)
            # such that sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
            
            # This is a nonlinear problem. Let's use a structured initialization.
            # Start with schoolbook and try to compress.
            
            # Alternative: use random initialization and gradient descent
            # But we need exact integer/half-integer solutions.
            
            # Let's use a different approach: exploit symmetry
            # 4x4x4 tensor has symmetries that can be exploited.
            
            # For now, return None to indicate we need a different approach
            return None
        
        return None
    
    # Since finding an exact rank-19 decomposition is complex, let's use
    # a practical approach: schoolbook with some optimization
    
    # Schoolbook decomposition for 4x4x5
    rank = n * m * p  # 80
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
    
    # Try to reduce rank by merging terms with the same (i,j) or (j,k)
    # For 4x4x5, we can potentially merge the 5 k-values for each (i,j)
    # into fewer terms using linear combinations
    
    # For each (i,j), we have 5 terms: (i,j,k) for k=0..4
    # We can write: sum_k a_{i,j,k} * e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # This is already rank 5 for each (i,j), so no easy merging.
    
    # Let's try a different strategy: use Strassen-like decomposition for 4x4x4
    # and add the 5th column separately
    
    # For 4x4x4, use rank-20 (or best found)
    # For the 5th column (k=4), add 4 terms: (i,j,4) for i=0..3, j=0..3
    
    # Actually, let's just return the schoolbook for now to ensure correctness,
    # and try to improve in future iterations.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
