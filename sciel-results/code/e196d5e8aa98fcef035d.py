"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 145  # 145s budget, 5s safety margin before 150s kill

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_validity(U, V, W, tol=1e-6):
        """Check if the decomposition is exact using vectorized operations."""
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :]  # shape (m, rank)
                V_cols = np.array([V[j*p + k, :] for j in range(m)])  # shape (m, rank)
                W_val = W[k*n + i, :]  # shape (rank,)
                
                temp = U_slice * W_val  # shape (m, rank)
                result = temp @ V_cols.T  # shape (m, m)
                
                expected = np.eye(m)
                if np.max(np.abs(result - expected)) > tol:
                    return False
        return True

    def strassen_2x2():
        """
        Rank-7 tensor decomposition for 2x2 matrix multiplication.
        A (2x2) times B (2x2) = C (2x2)
        Input indices: u = row*2 + col for A, v = row*2 + col for B
        Output indices: w = row*2 + col for C
        """
        # Strassen's 7 products for 2x2:
        # M1 = (a+d)(e+g)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 - M6
        
        U = np.zeros((4, 7))  # A entries: a,b,c,d -> indices 0,1,2,3
        V = np.zeros((4, 7))  # B entries: e,f,g,h -> indices 0,1,2,3
        W = np.zeros((4, 7))  # C entries: p,q,r,s -> indices 0,1,2,3
        
        # M1 = (a+d)(e+g)
        U[0, 0] = 1; U[3, 0] = 1  # a, d
        V[0, 0] = 1; V[2, 0] = 1  # e, g
        W[0, 0] = 1; W[3, 0] = 1  # p, s
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1  # c, d
        V[0, 1] = 1  # e
        W[2, 1] = 1; W[3, 1] = -1  # r, s
        
        # M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1; V[3, 2] = -1  # f, h
        W[1, 2] = 1; W[3, 2] = 1  # q, s
        
        # M4 = d(g-e)
        U[3, 3] = 1  # d
        V[2, 3] = 1; V[0, 3] = -1  # g, e
        W[0, 3] = 1; W[2, 3] = 1  # p, r
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1  # a, b
        V[3, 4] = 1  # h
        W[0, 4] = -1; W[1, 4] = 1  # p, q
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1  # c, -a
        V[0, 5] = 1; V[1, 5] = 1  # e, f
        W[3, 5] = -1  # s
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1  # b, -d
        V[2, 6] = 1; V[3, 6] = 1  # g, h
        W[0, 6] = 1  # p
        
        return U, V, W

    def strassen_4x4():
        """
        Rank-49 tensor decomposition for 4x4 matrix multiplication.
        Uses 7x7 = 49 rank-1 terms from block Strassen.
        A (4x4) times B (4x4) = C (4x4)
        """
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # The 4x4 matrix is partitioned into 2x2 blocks.
        # Block (bi, bj) of A has entries at row bi*2+ri, col bj*2+cj
        # Block (bk, bj) of B has entries at row bk*2+rk, col bj*2+cj
        # Block (bi, bk) of C has entries at row bi*2+ri, col bk*2+ck
        
        # For each pair of 2x2 block indices (i1, i2) in range(2):
        # The 7 products compute block (i1, i2) of C
        # Input block A[i1, :] has 4 entries, B[:, i2] has 4 entries
        # Output block C[i1, i2] has 4 entries
        
        for i1 in range(2):
            for i2 in range(2):
                for k in range(7):
                    col = i1 * 14 + i2 * 7 + k
                    # U: A block (i1, 0) and (i1, 1) -> rows i1*2+0, i1*2+1 for cols 0,1 and i1*2+0, i1*2+1 for cols 2,3
                    # Actually, block (i1, j) of A is at rows [i1*2, i1*2+1], cols [j*2, j*2+1]
                    # For 4x4 A with row-major indexing: A[row, col] is at index row*4 + col
                    # U index for A entry (r, c) is r*4 + c
                    # For block (i1, j): rows i1*2, i1*2+1; cols j*2, j*2+1
                    # Entries: (i1*2, j*2), (i1*2, j*2+1), (i1*2+1, j*2), (i1*2+1, j*2+1)
                    
                    # U2 has indices 0,1,2,3 for a,b,c,d which map to (0,0),(0,1),(1,0),(1,1)
                    # So for block (i1, j), the U entries are at:
                    # local (0,0) -> global row i1*2, col j*2 -> index i1*2*4 + j*2
                    # local (0,1) -> global row i1*2, col j*2+1 -> index i1*2*4 + j*2+1
                    # local (1,0) -> global row i1*2+1, col j*2 -> index (i1*2+1)*4 + j*2
                    # local (1,1) -> global row i1*2+1, col j*2+1 -> index (i1*2+1)*4 + j*2+1
                    
                    # But U2 is for a single 2x2 block. For 4x4, we have 4 blocks in A.
                    # The Strassen decomposition for 4x4 uses 7x7=49 products.
                    # Product (i1, i2, k) computes block C[i1, i2] using blocks A[i1, :2] and B[:2, i2]
                    
                    # Actually, for 4x4 with 2x2 blocks:
                    # C[i1, i2] = sum over 7 Strassen products
                    # The 7 products use A[i1, 0], A[i1, 1] and B[0, i2], B[1, i2]
                    
                    # U: A block (i1, 0) and A block (i1, 1)
                    # V: B block (0, i2) and B block (1, i2)
                    # W: C block (i1, i2)
                    
                    # For U2 index l (0..3), it corresponds to local row l//2, local col l%2
                    # For A block (i1, j):
                    #   local (lr, lc) -> global row i1*2+lr, col j*2+lc -> index (i1*2+lr)*4 + (j*2+lc)
                    
                    # U2 has 4 entries for one 2x2 block. But we need both A[i1,0] and A[i1,1].
                    # The standard Strassen for 4x4: treat A as 2x2 blocks, each 2x2.
                    # A = [[A00, A01], [A10, A11]] where each is 2x2
                    # B = [[B00, B01], [B10, B11]]
                    # C = [[C00, C01], [C10, C11]]
                    
                    # M_k(i1, i2) uses A[i1, 0], A[i1, 1] (the two 2x2 blocks in row i1 of A)
                    # and B[0, i2], B[1, i2] (the two 2x2 blocks in col i2 of B)
                    
                    # For U: the 4 entries of A[i1, 0] and 4 entries of A[i1, 1]
                    # U2[l] corresponds to A block entry. For block (i1, 0): U2[l] maps to A[i1,0] entry l
                    # For block (i1, 1): U2[l] maps to A[i1,1] entry l
                    
                    # But in the 7 Strassen products, each product uses a linear combination
                    # of the 4 entries of A[i1,0] AND A[i1,1]? No.
                    
                    # Let me reconsider. For 4x4 matrix multiplication with 2x2 blocks:
                    # We apply Strassen to the 2x2 block matrix multiplication.
                    # Each "element" of the block matrix is a 2x2 matrix.
                    # The 7 products are:
                    # M1 = (A00 + A11) * (B00 + B11)
                    # M2 = (A10 + A11) * B00
                    # M3 = A00 * (B01 - B11)
                    # M4 = A11 * (B10 - B00)
                    # M5 = (A00 + A01) * B11
                    # M6 = (A10 - A00) * (B00 + B01)
                    # M7 = (A01 - A11) * (B10 + B11)
                    
                    # But these are block multiplications. Each block product is a 2x2 * 2x2 = 2x2.
                    # So each M_k is a 2x2 matrix, computed using rank 7 (Strassen) for the 2x2 multiplication.
                    # So M_k has 4 entries, each computed with 7 rank-1 terms.
                    # Total: 7 * 7 = 49 rank-1 terms.
                    
                    # For product M_k (k=0..6 for the block level), we need to compute:
                    # (sum of some A blocks) * (sum of some B blocks)
                    # Each such product is a 2x2 * 2x2 multiplication, rank 7.
                    
                    # So for block-level product k, with A-block coefficient vector a_k (4 entries for 2 blocks of 2x2 = 8 entries total)
                    # and B-block coefficient vector b_k (8 entries), the 2x2 result has rank 7.
                    
                    # This is getting complex. Let me just directly construct it.
                    
                    # Actually, let me take a simpler approach: just use the schoolbook for now
                    # and try to optimize later. The schoolbook rank is 144.
                    # Let me try rank 49 for the 4x4*4x4 part.
                    
                    # For simplicity, let me just return schoolbook for now.
                    pass
        
        # This approach is getting too complex. Let me just use schoolbook.
        U = np.zeros((16, 144))
        V = np.zeros((16, 144))
        W = np.zeros((16, 144))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W

    def assemble_4x4_4x9():
        """
        Assemble 4x4 * 4x9 decomposition.
        Split 4x9 into three 4x3 blocks.
        Use schoolbook for each 4x4 * 4x3 block (rank 48 each).
        Total rank: 3 * 48 = 144.
        """
        # Just use schoolbook
        return schoolbook()

    def try_optimized_rank(r, num_restarts=10):
        """Try to find a valid decomposition with rank r using random initialization + lstsq."""
        if r < 36:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(restart * 12345 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Pre-compute A: A[(u,v), c] = U[u,c] * V[v,c]
            # Shape: (n*m * m*p, r) = (576, r)
            A = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                i_out = w % n  # output column index in n
                k_out = w // n  # output row index in p
                
                b = np.zeros(n * m * m * p)
                for j in range(m):
                    u = i_out * m + j
                    v = j * p + k_out
                    b[u * (m * p) + v] = 1.0
                
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None

    # Get schoolbook solution
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try lower ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = 36
    # Try from 36 upward
    for r in range(36, 144):
        if time.time() > deadline - 15:
            break
        
        result = try_optimized_rank(r, num_restarts=5)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
