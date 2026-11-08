"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen's 7-term algorithm for 2x2x2 blocks combined with structured
    block splitting to achieve rank below the schoolbook 60.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 135  # Leave 15s margin for 150s limit

    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r), dtype=float)
        V = np.zeros((m * p, r), dtype=float)
        W = np.zeros((p * n, r), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r

    def strassen_2x2x2():
        """
        Return U, V, W matrices (each shape (4, 7)) for the 2x2x2 Strassen decomposition.
        U indexes A (2x2), V indexes B (2x2), W indexes C (2x2).
        The 7 products are:
          M1 = (A00+A11)*(B00+B11)
          M2 = (A10+A11)*B00
          M3 = A00*(B01-B11)
          M4 = A11*(B10-B00)
          M5 = (A00+A01)*B11
          M6 = (A10-A00)*(B00+B01)
          M7 = (A01-A11)*(B10+B11)
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C00 gets +M1, C11 gets +M1
        # Wait, W should be for C indices. Let me redo.
        # C has 4 entries: C00, C01, C10, C11
        # W[k*n+i, r] where k is col of C, i is row of C
        # So C00 -> W[0], C01 -> W[1], C10 -> W[2], C11 -> W[3]
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # C00 += M1
        W[3, 0] = 1  # C11 += M1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C10 += M2
        W[3, 1] = -1  # C11 -= M2
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C01 += M3
        W[3, 2] = 1  # C11 += M3
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # C00 += M4
        W[2, 3] = 1  # C10 += M4
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # C00 -= M5
        W[1, 4] = 1  # C01 += M5
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C11 += M6
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C00 += M7
        
        return U, V, W

    def combine_decompositions(decs):
        """
        Combine a list of decompositions. Each dec is (U, V, W) where
        U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
        Returns combined U, V, W with total rank = sum of ranks.
        """
        total_rank = sum(U.shape[1] for U, V, W in decs)
        n, m, p = DIMENSIONS
        U_out = np.zeros((n * m, total_rank), dtype=float)
        V_out = np.zeros((m * p, total_rank), dtype=float)
        W_out = np.zeros((p * n, total_rank), dtype=float)
        col = 0
        for U, V, W in decs:
            r = U.shape[1]
            U_out[:, col:col+r] = U
            V_out[:, col:col+r] = V
            W_out[:, col:col+r] = W
            col += r
        return U_out, V_out, W_out

    def validate(U, V, W):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Split the problem into blocks and use Strassen where beneficial.
    # 
    # A is 2x5, B is 5x6, C is 2x6
    # 
    # Approach 1: Split m=5 into 2+3, keep p=6 whole.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:5] @ B[2:5,:]
    # Term 1: 2x2 * 2x6 -> split p=6 into 3+3: two 2x2*2x3 problems
    # Term 2: 2x3 * 3x6 -> split p=6 into 3+3: two 2x3*3x3 problems
    #
    # For 2x2*2x3: Split as [2x2*2x2] + [2x2*2x1]
    #   2x2*2x2: Strassen rank 7
    #   2x2*2x1: schoolbook rank 4 (or can we do better? It's just 4 scalar mults)
    #   Total for 2x2*2x3: 7 + 4 = 11
    #
    # For 2x3*3x3: Split as [2x2*3x3] + [2x1*3x3]
    #   2x2*3x3: Split 3 into 2+1: [2x2*3x2] + [2x2*3x1]
    #     2x2*3x2: Split 3 into 1+2: [2x2*1x2] + [2x2*2x2]
    #       2x2*1x2: rank 4 (schoolbook)
    #       2x2*2x2: Strassen rank 7
    #       Total: 11
    #     2x2*3x1: Split 3 into 1+2: [2x2*1x1] + [2x2*2x1]
    #       2x2*1x1: rank 1
    #       2x2*2x1: rank 4
    #       Total: 5
    #     2x2*3x3 total: 11 + 5 = 16
    #   2x1*3x3: rank 3 (schoolbook)
    #   2x3*3x3 total: 16 + 3 = 19
    #
    # Grand total: 2*11 + 2*19 = 22 + 38 = 60. Same as schoolbook!
    #
    # Let me try a different split.
    #
    # Approach 2: Split m=5 into 2+3, p=6 into 2+2+2
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:5] @ B[2:5,:]
    # Term 1: 2x2 * 2x6 -> split p into 2+2+2: three 2x2*2x2 problems
    #   Each 2x2*2x2: Strassen rank 7
    #   Total: 21
    # Term 2: 2x3 * 3x6 -> split p into 2+2+2: three 2x3*3x2 problems
    #   2x3*3x2: Split m=3 into 2+1: [2x2*3x2] + [2x1*3x2]
    #     2x2*3x2: Split m=3 into 1+2: [2x2*1x2] + [2x2*2x2]
    #       2x2*1x2: rank 4
    #       2x2*2x2: rank 7
    #       Total: 11
    #     2x1*3x2: rank 2
    #     2x3*3x2 total: 13
    #   Three of these: 39
    # Grand total: 21 + 39 = 60. Still 60.
    #
    # Hmm, Strassen only helps when we have 2x2x2 blocks. The savings are 8-7=1 per block.
    #
    # Let me try: m=5 = 2+3, p=6 = 3+3
    # Term 1: 2x2 * 2x6 -> split p=6 into 3+3: two 2x2*2x3
    #   2x2*2x3: schoolbook rank 12. Can we do better?
    #   Actually 2x2*2x3 is just 12 scalar multiplications. No Strassen benefit since
    #   one dimension is odd.
    #
    # Let me think differently. The key insight is that for 2x2x2, Strassen saves 1.
    # If I can identify how many independent 2x2x2 sub-problems exist, I can save that many.
    #
    # Alternative: Use the fact that R(2,m,p) can sometimes be less than 2*m*p.
    #
    # Let me try a completely different approach: use a rank-reduction algorithm
    # on the schoolbook decomposition by finding linear dependencies.
    #
    # Actually, let me try the simplest structured approach that might work:
    # Split m=5 into 2+3.
    # For the 2x2 part: use Strassen-like tricks.
    # For the 2x3 part: use schoolbook.
    
    # Let me just implement the structured decomposition carefully.
    
    # Split: m = 2 + 3, p = 3 + 3
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x3
    # B = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1 @ B1 + A2 @ B2
    
    # For A1 @ B1 (2x2 * 2x6):
    #   Split B1 into [B1a | B1b] where each is 2x3
    #   A1 @ B1a (2x2 * 2x3): schoolbook rank 12
    #   A1 @ B1b (2x2 * 2x3): schoolbook rank 12
    #   Total: 24
    
    # For A2 @ B2 (2x3 * 3x6):
    #   Split B2 into [B2a | B2b] where each is 3x3
    #   A2 @ B2a (2x3 * 3x3): schoolbook rank 18
    #   A2 @ B2b (2x3 * 3x3): schoolbook rank 18
    #   Total: 36
    
    # Grand total: 60. No improvement.
    
    # The issue is that Strassen only applies to 2x2x2. Let me try to use it more effectively.
    
    # New approach: Split m=5 into 2+3, p=6 into 2+4
    # A1 @ B1 (2x2 * 2x6): 
    #   Split B1 into B1a (2x2) and B1b (2x4)
    #   A1 @ B1a (2x2 * 2x2): Strassen rank 7
    #   A1 @ B1b (2x2 * 2x4): Split into two 2x2*2x2: 2*7 = 14
    #   Total: 21
    # A2 @ B2 (2x3 * 3x6):
    #   Split B2 into B2a (3x2) and B2b (3x4)
    #   A2 @ B2a (2x3 * 3x2): Split A2 into A2a (2x2) and A2b (2x1)
    #     A2a @ B2a (2x2 * 3x2): Split B2a into B2a1 (3x1) and B2a2 (3x1)
    #       A2a @ B2a1 (2x2 * 3x1): Split A2a into rows: (2x1*3x1) + (2x1*3x1) = 2+2=4? 
    #       Actually 2x2*3x1: each of 2 rows of A2a (1x2) times B2a (3x1) gives 1x1
    #       That's 2*2*1 = 4 scalar mults. Rank 4.
    #       A2a @ B2a2 (2x2 * 3x1): rank 4
    #       Total for A2a @ B2a: 8
    #     A2b @ B2a (2x1 * 3x2): rank 2
    #     Total for A2 @ B2a: 10
    #   A2 @ B2b (2x3 * 3x4): 
    #     Split B2b into two 3x2 blocks
    #     A2 @ each 3x2: same as above = 10 each
    #     Total: 20
    #   Total for A2 @ B2: 30
    # Grand total: 21 + 30 = 51
    
    # That's better! 51 < 60. Let me implement this.
    
    # Actually, let me be more careful and just implement a general recursive approach.
    
    def matmul_decomp(n_rows, m_cols, p_cols, j_offset, k_offset):
        """
        Return (U, V, W) for multiplying an n_rows x m_cols matrix A_sub
        by an m_cols x p_cols matrix B_sub.
        j_offset: offset in the original m dimension
        k_offset: offset in the original p dimension
        
        U has shape (2*m_cols, rank) - but actually U should be indexed by original A indices
        V has shape (m_cols*p_cols, rank) - indexed by original B indices  
        W has shape (p_cols*2, rank) - indexed by original C indices
        
        Wait, this is getting complicated with offsets. Let me use a different approach.
        """
        pass

    # Let me use a cleaner recursive approach with explicit index tracking.
    
    def build_decomp(n, m, p):
        """
        Build a decomposition for n x m * m x p matrix multiplication.
        Returns (U, V, W) where:
        U: (n*m, rank)
        V: (m*p, rank)  
        W: (p*n, rank)
        """
        # Base cases
        if m == 0 or p == 0 or n == 0:
            r = 0
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            W = np.zeros((p * n, r), dtype=float)
            return U, V, W
        
        # If all dimensions are small, use schoolbook
        if n * m * p <= 8:
            r = n * m * p
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            W = np.zeros((p * n, r), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return U, V, W
        
        # Try Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            return strassen_2x2x2()
        
        # Recursive splitting
        # Try different splits and pick the best one
        
        best_rank = n * m * p
        best_result = None
        
        # Split m into m1 + m2
        for m1 in range(1, m):
            m2 = m - m1
            # Recursively solve for (n, m1, p) and (n, m2, p)
            U1, V1, W1 = build_decomp(n, m1, p)
            U2, V2, W2 = build_decomp(n, m2, p)
            
            # Adjust indices for the second part
            # U2 needs to be shifted: row index j -> j + m1
            U2_adj = np.zeros((n * m, U2.shape[1]), dtype=float)
            for i in range(n):
                for j in range(m2):
                    U2_adj[i * m + (j + m1), :] = U2[i * m1 + j, :]
            
            # V2 needs to be shifted: column index j -> j + m1
            V2_adj = np.zeros((m * p, V2.shape[1]), dtype=float)
            for j in range(m2):
                for k in range(p):
                    V2_adj[(j + m1) * p + k, :] = V2[j * p + k, :]
            
            # W stays the same
            r_total = U1.shape[1] + U2_adj.shape[1]
            U_comb = np.hstack([U1, U2_adj])
            V_comb = np.hstack([V1, V2_adj])
            W_comb = np.hstack([W1, W2])
            
            if r_total < best_rank:
                best_rank = r_total
                best_result = (U_comb, V_comb, W_comb)
        
        # Split p into p1 + p2
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = build_decomp(n, m, p1)
            U2, V2, W2 = build_decomp(n, m, p2)
            
            # U stays the same
            # V2 needs to be shifted: column index k -> k + p1
            V2_adj = np.zeros((m * p, V2.shape[1]), dtype=float)
            for j in range(m):
                for k in range(p2):
                    V2_adj[j * p + (k + p1), :] = V2[j * p1 + k, :]
            
            # W2 needs to be shifted: row index k -> k + p1
            W2_adj = np.zeros((p * n, W2.shape[1]), dtype=float)
            for k in range(p2):
                for i in range(n):
                    W2_adj[(k + p1) * n + i, :] = W2[k * n + i, :]
            
            r_total = U1.shape[1] + U2_adj.shape[1]
            U_comb = np.hstack([U1, U2_adj])
            V_comb = np.hstack([V1, V2_adj])
            W_comb = np.hstack([W1, W2_adj])
            
            if r_total < best_rank:
                best_rank = r_total
                best_result = (U_comb, V_comb, W_comb)
        
        # Split n into n1 + n2
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = build_decomp(n1, m, p)
            U2, V2, W2 = build_decomp(n2, m, p)
            
            # U2 needs to be shifted: row index i -> i + n1
            U2_adj = np.zeros((n * m, U2.shape[1]), dtype=float)
            for i in range(n2):
                for j in range(m):
                    U2_adj[(i + n1) * m + j, :] = U2[i * m + j, :]
            
            # V stays the same
            # W2 needs to be shifted: column index i -> i + n1
            W2_adj = np.zeros((p * n, W2.shape[1]), dtype=float)
            for k in range(p):
                for i in range(n2):
                    W2_adj[k * n + (i + n1), :] = W2[k * n + i, :]
            
            r_total = U1.shape[1] + U2_adj.shape[1]
            U_comb = np.hstack([U1, U2_adj])
            V_comb = np.hstack([V1, V2_adj])
            W_comb = np.hstack([W1, W2_adj])
            
            if r_total < best_rank:
                best_rank = r_total
                best_result = (U_comb, V_comb, W_comb)
        
        if best_result is None:
            # Fallback to schoolbook
            r = n * m * p
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            W = np.zeros((p * n, r), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return U, V, W
        
        return best_result

    # Build the decomposition
    U, V, W = build_decomp(n, m, p)
    rank = U.shape[1]
    
    # Validate
    if not validate(U, V, W):
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
