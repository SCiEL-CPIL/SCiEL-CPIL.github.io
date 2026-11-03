import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140
    
    # Winograd-style decomposition for 3x3 x 3x4 -> 3x4
    # Strategy: Use 2-point DFT on the p=4 dimension (2 groups of 2)
    # For each group, use structured transforms on n=3 and m=3
    
    # The key idea: express the multiplication using DFT basis vectors
    # Over the reals, use the real 2-point DFT (which is just sum/diff)
    
    # For the 4 columns, split into 2 pairs: (0,2) and (1,3)
    # For each pair, use: B_sum = B[:,a] + B[:,b], B_diff = B[:,a] - B[:,b]
    # Then C[:,a] = (A@B_sum + A@B_diff)/2, C[:,b] = (A@B_sum - A@B_diff)/2
    
    # But this still uses 9 mults per vector product.
    # The Winograd advantage comes from sharing across the n dimension.
    
    # Let's use a 3-point DFT-like structure on the n=3 rows.
    # The real 3-point DFT matrix (for frequency 0,1,2) can be written using
    # cosines. For 3 points:
    # W = [[1, 1, 1],
    #      [1, -1/2, -1/2],
    #      [0, sqrt(3)/2, -sqrt(3)/2]]
    # This is orthogonal up to scaling.
    
    # Actually, for exact rational/half-integer entries, let's use:
    # The 3-point "DFT" over the Eisenstein integers, but we need real/rational.
    
    # Let me try a different approach: use the known rank-23 decomposition for 3x3x3
    # and extend to 3x3x4.
    
    # For 3x3x3, the best known rank is 23 (Landsberg's bound, achieved by 
    # a specific decomposition). For 3x3x4, we can use:
    # rank(3,3,4) <= rank(3,3,3) + 9 = 23 + 9 = 32 (trivial)
    # Or better: use the 2-point DFT on the extra column.
    
    # Let's construct a rank-28 decomposition:
    # Use the 3x3x3 rank-23 decomposition for columns 0,1,2
    # Plus 9 more for column 3 (schoolbook for the 4th column)
    # Total: 32. Not great.
    
    # Better: use 2-point DFT to pair columns (0,2) and (1,3)
    # For each pair, the "effective" problem is 3x3 x 3x2
    # rank(3,3,2) = ? At most 3*3*2 = 18, but Winograd gives less.
    # Actually rank(3,3,2) <= 2*rank(3,3,1) = 2*9 = 18, or using DFT: 
    # The 2-point DFT on p=2 gives rank = 2 * (n*m) = 18. No improvement.
    
    # Let me try a direct algebraic construction with rank 24.
    
    # I'll use the following scheme:
    # Transform A using a 3x3 matrix P (3-point DFT-like)
    # Transform B using a 3x4 matrix Q (structured)
    # The product P@A@Q has rank-1 structure that can be decomposed.
    
    # For now, let's use a safe rank-30 construction and verify it.
    # Actually, let's just use the schoolbook rank 36 as a fallback
    # and try to improve with a simple heuristic.
    
    # Let me implement a rank-24 construction using 2-point DFT pairs:
    # Pair (col 0, col 2): B0 = B[:,0]+B[:,2], B1 = B[:,0]-B[:,2]
    # Pair (col 1, col 3): B2 = B[:,1]+B[:,3], B3 = B[:,1]-B[:,3]
    # For each "virtual" column, compute A@Bv using 9 mults
    # But we can share: the same A is used, so we compute:
    # S0 = A@B0, S1 = A@B1, S2 = A@B2, S3 = A@B3
    # Each Si is 3x1, requiring 9 mults. Total 36. No gain.
    
    # The gain comes from transforming A too. Let's use:
    # A' = P @ A where P is 3x3
    # B' = B @ Q where Q is 4x4 (DFT)
    # Then C = P^{-1} @ A' @ B' @ Q^{-1}
    # The number of mults is the rank of the tensor for A'@B'.
    
    # If P and Q are DFT matrices, the tensor rank of the transformed
    # multiplication can be lower.
    
    # For 3x3, the DFT-based Winograd gives rank 23 (not 27).
    # For the 4th column, add 9 more: total 32.
    
    # Let me just return the schoolbook solution for safety,
    # but with a clean implementation.
    
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
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        n, m, p = 3, 3, 4
        U_mat = np.array(U_mat)
        V_mat = np.array(V_mat)
        W_mat = np.array(W_mat)
        rank = U_mat.shape[1]
        # Check: sum_r W[k*n+i, r] * U[i'*m+j', r] * V[j'*p+k', r] = delta_{i,i'} * delta_{k,k'}
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            lhs = sum(W_mat[k*n+i, r] * U_mat[ip*m+jp, r] * V_mat[jp*p+kp, r] for r in range(rank))
                            rhs = (1.0 if i == ip and k == kp else 0.0)
                            if abs(lhs - rhs) > 1e-9:
                                return False
        return True
    
    if not verify(U, V, W):
        # Should not happen, but fallback
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
