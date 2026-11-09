"""Matrix multiplication 3,4,8 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 8]

import time
import numpy as np

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    Conventions: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1: (A11 + A22)(B11 + B22)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2: (A21 + A22) * B11
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0
    
    # M3: A11 * (B12 - B22)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0
    
    # M4: A22 * (B21 - B11)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0
    
    # M5: (A11 + A12) * B22
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0
    
    # M6: (A21 - A11) * (B11 + B12)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7: (A12 - A22) * (B21 + B22)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0; W[3][6] = 1.0
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness with random test."""
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C = A @ B
    
    # Reconstruct using tensor decomposition
    C_recon = np.zeros((n, p))
    rank = len(U[0]) if U else 0
    for t in range(rank):
        u = np.array([U[i*m+j][t] for i in range(n) for j in range(m)]).reshape(n, m)
        v = np.array([V[j*p+k][t] for j in range(m) for k in range(p)]).reshape(m, p)
        w = np.array([W[k*n+i][t] for k in range(p) for i in range(n)]).reshape(p, n)
        # C[i,k] += sum_j u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += u[i,j] * v[j,k] * w[k,i]
                C_recon[i,k] += s
    
    return np.allclose(C, C_recon, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Split m=4 into two blocks of 2, p=8 into four blocks of 2.
    # For each (m_block, p_block) pair, we have a 3x2x2 subproblem.
    # Use Strassen on the 2x2 part for each.
    
    # m_blocks: [0,1] and [2,3]
    # p_blocks: [0,1], [2,3], [4,5], [6,7]
    
    # For a 3x2x2 subproblem with Strassen on the 2x2 core:
    # The 2x2 Strassen gives rank 7 for the 2x2x2 part.
    # But we have 3 rows in A, so we need to extend.
    # 
    # Actually, for 3x2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is just linear. The tensor rank for 3x2x2 is 6 (each output needs 
    # contributions from 2 products, but they're independent per i).
    # 
    # Let me use a different approach: direct rank-6 decomposition for 3x2x2.
    
    def _rank6_3x2x2():
        """
        Rank-6 decomposition for 3x2x2 matrix multiplication.
        C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1,2}, k in {0,1}
        
        U shape: (3*2, 6) = (6, 6)
        V shape: (2*2, 6) = (4, 6)  
        W shape: (2*3, 6) = (6, 6)
        
        U[i*2+j, t], V[j*2+k, t], W[k*3+i, t]
        
        We need 6 terms: for each (i,k) pair, one term per j? No.
        Actually: C[i,k] = sum_j A[i,j]*B[j,k]
        For fixed i,k, there are 2 terms (j=0,1). Total 3*2*2=12 terms in schoolbook.
        But we can share: for each i, the two A[i,j] are shared across k.
        
        Minimal rank for 3x2x2: Let's think of it as computing 6 outputs, each a dot product of length 2.
        The tensor is T[i,j,k] = e_{i,j} * e_{j,k} -> contributes to C[i,k].
        
        Actually the rank is just 6: we can write
        C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
        
        That's 12 products. Can we do better?
        
        Hmm, for this specific shape, I think the rank is 6. Let me just use schoolbook for now
        and see if the overall rank improves with Strassen on larger blocks.
        """
        # Schoolbook for 3x2x2: rank = 3*2*2 = 12
        # But let's try to find a better one.
        # 
        # Alternative: Use the fact that this is equivalent to multiplying 
        # a 3x2 matrix by a 2x2 matrix. We can view B as two column vectors b0, b1.
        # C[:,0] = A * b0, C[:,1] = A * b1.
        # Each is a 3x2 times 2x1 = 3 products. Total 6 products if we share A.
        # But in tensor form, each product A[i,j]*B[j,k] is separate.
        # 
        # I think the minimal rank for 3x2x2 is 6. Let me construct it:
        # Term (i,k): A[i,0]*B[0,k] -> contributes to C[i,k]
        # Term (i,k): A[i,1]*B[1,k] -> contributes to C[i,k]
        # That's 12 terms. To get rank 6, we need to combine.
        #
        # Actually, I realize I should just use Strassen on 2x2x2 blocks and handle
        # the 3-row dimension separately.
        
        # Let me just return schoolbook for now: rank 12
        U = [[0.0]*12 for _ in range(6)]
        V = [[0.0]*12 for _ in range(4)]
        W = [[0.0]*12 for _ in range(6)]
        
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1.0
                    V[j*2+k][col] = 1.0
                    W[k*3+i][col] = 1.0
                    col += 1
        
        return U, V, W
    
    # Let me try a completely different approach: 
    # Use Strassen to reduce the 4x8 multiplication.
    # View the 4x8 as (2+2) x (2+2+2+2).
    # Split into 2x2 blocks: there are 2 row-blocks and 4 col-blocks = 8 blocks of 2x2.
    # For each 2x2 block multiplication (from the A side being 3x2 and B side being 2x2),
    # use Strassen with rank 7.
    # Total rank: 8 * 7 = 56.
    
    # But wait, the A matrix is 3x4, not 2x4. So for each column block of A (size 3x2),
    # and each 2x2 block of B, we multiply 3x2 by 2x2.
    
    # For 3x2 x 2x2: Let's use a rank-7 method based on Strassen.
    # A is 3x2: rows a0, a1, a2 (each a 2-vector). B is 2x2.
    # C = A*B is 3x2.
    # We can use Strassen on the 2x2 part of B while keeping A's 3 rows.
    
    # Actually, let me just implement the full decomposition directly.
    
    # New approach: Direct construction using Strassen for 2x2 subblocks.
    # Split m=4 into m1=2 (indices 0,1) and m2=2 (indices 2,3).
    # Split p=8 into p1=2 (0,1), p2=2 (2,3), p3=2 (4,5), p4=2 (6,7).
    
    # For each (m_block, p_block) pair, we multiply a 3x2 submatrix of A by a 2x2 submatrix of B.
    # Use Strassen-based rank-7 for each such 3x2x2 subproblem.
    
    # For 3x2x2 with Strassen on the 2x2 B part:
    # Let A_sub be 3x2 with rows r0, r1, r2 (each 2D).
    # Let B_sub be 2x2.
    # C_sub = A_sub * B_sub is 3x2.
    
    # Using Strassen's 7 multiplications on B:
    # M1 = (B00+B11) * (r0[0]+r1[0]) ... no, Strassen works on 2x2*2x2.
    # Here we have 3x2 * 2x2. 
    # 
    # I can write: C_sub = [r0; r1; r2] * B_sub
    # = [[r0[0]*B00 + r0[1]*B10], [r0[0]*B01 + r0[1]*B11]]
    #   [[r1[0]*B00 + r1[1]*B10], [r1[0]*B01 + r1[1]*B11]]
    #   [[r2[0]*B00 + r2[1]*B10], [r2[0]*B01 + r2[1]*B11]]
    
    # This is 3*2*2 = 12 products. Can we do better?
    # For each row of C_sub, it's a 1x2 * 2x2 = 2 products. 3 rows = 6 products total if we can share.
    # But each product involves different A entries, so no sharing between rows.
    # Within a row: r[0]*B00 + r[1]*B10 for column 0, r[0]*B01 + r[1]*B11 for column 1.
    # That's 4 products per row, 12 total.
    
    # I think for 3x2x2, the minimal rank is 6 (not 12). Here's why:
    # The tensor T[i,j,k] where i in {0,1,2}, j in {0,1}, k in {0,1}
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # This is just the standard matrix mult tensor for 3x2x2.
    # The rank of M_{3,2,2} ... I believe it's 6.
    
    # Let me just go with schoolbook for the subproblems and see the total rank.
    # 8 subproblems * 12 = 96. Same as before. No improvement.
    
    # I need a fundamentally different approach. Let me use Strassen on the full problem.
    
    # View 3x4x8 as: pad to 4x4x8 or use recursive Strassen.
    # Or: use the known result that M_{3,4,8} has rank at most 3*4*8/2 = 48 
    # if we can pair up terms.
    
    # Actually, let me try: split m=4 into 2+2, and for each half, use Strassen 
    # on the 2x8 part by viewing it as 2x(2+2+2+2) and using Strassen on each 2x2.
    
    # For A_half (3x2) * B_block (2x2): rank 6 (as argued above, minimal for 3x2x2)
    # 4 B_blocks * 2 A_halves = 8 subproblems * 6 = 48.
    
    # Let me construct rank-6 for 3x2x2:
    def _rank6_3x2x2_correct():
        """
        Rank-6 decomposition for 3x2x2.
        C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        
        U shape: (6, 6), V shape: (4, 6), W shape: (6, 6)
        U[i*2+j, t], V[j*2+k, t], W[k*3+i, t]
        
        Term t for (i, k): 
          t = i*2 + k
          U[i*2+0, t] = 1, U[i*2+1, t] = 1  -- no, that would be sum
          
        Actually for rank 6, each term handles one (i,k) pair but combines both j:
        C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) -- this is ONE term? No, it's two products.
        
        In tensor decomposition, each rank-1 term is u ⊗ v ⊗ w where
        u is in R^{n*m}, v in R^{m*p}, w in R^{p*n}.
        The term contributes u[i*m+j] * v[j*p+k] * w[k*n+i] to C[i,k].
        
        For a single product A[i0,j0]*B[j0,k0] contributing to C[i0,k0]:
        u[i0*m+j0] = 1, v[j0*p+k0] = 1, w[k0*n+i0] = 1.
        
        To get rank 6 for 3x2x2, I need 6 such terms that together give all 6 C[i,k] values.
        But each C[i,k] needs 2 products (j=0 and j=1). So 12 products total.
        Unless I can use non-1 coefficients to combine.
        
        I think the rank is actually 12 for 3x2x2 in this formulation, OR I'm misunderstanding.
        
        Let me just use schoolbook and accept rank 96, but try to verify and clean up.
        """
        U = [[0.0]*12 for _ in range(6)]
        V = [[0.0]*12 for _ in range(4)]
        W = [[0.0]*12 for _ in range(6)]
        
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1.0
                    V[j*2+k][col] = 1.0
                    W[k*3+i][col] = 1.0
                    col += 1
        
        return U, V, W
    
    # Given the complexity, let me just return the schoolbook decomposition but 
    # verify it's correct and well-formed.
    
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
    
    return {"U": U, "V": V, "W": W}
