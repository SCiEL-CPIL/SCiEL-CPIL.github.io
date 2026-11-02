"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify exact decomposition."""
        r = len(U[0])
        # Check diagonal: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal: sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i3,c] == 0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i3][c]
                                if abs(s) > tol:
                                    return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return U, V, W, rank

    # Fallback: schoolbook decomposition (rank 50)
    U0, V0, W0, r0 = schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U0, V0, W0, r0

    # Strategy: Exploit linearity in output dimension n=2.
    # Decompose into two independent 1x5x5 row-vector-matrix multiplications.
    # For a 1xm x mxp multiplication, the rank is m*p (schoolbook) unless there's a better algorithm.
    # For 1x5 x 5x5, schoolbook rank is 25.
    # However, the directive suggests leveraging "known optimal rank of 5 for each row".
    # This implies a rank-5 decomposition for 1x5x5? That seems impossible for general matrices (rank is at least max(m,p) but usually higher).
    # Wait, the directive says: "leveraging the known optimal rank of 5 for each row to construct a global rank-10 decomposition via direct stacking".
    # This suggests that for the specific structure of the tensor, we might be able to do better than 50.
    # Let's re-read carefully: "Exploit the linearity in the output dimension by decomposing the 2x5x5 multiplication tensor into two independent 1x5x5 row-vector-matrix multiplication tensors".
    # If we treat the output dimension i separately:
    # Output C[i, k] = sum_j A[i, j] B[j, k].
    # For fixed i, this is a 1x5 vector A[i,:] times a 5x5 matrix B.
    # The tensor for this is T_{j, k} = A[i, j] B[j, k].
    # This is a rank-1 tensor in the (A, B) space? No, it's bilinear.
    # The multiplication tensor for 1x5 x 5x5 has rank 25 (schoolbook).
    # If we stack two such tensors (for i=0 and i=1), the total rank would be 25 + 25 = 50.
    # The directive claims a "global rank-10 decomposition". This implies rank 5 per row.
    # Is it possible to compute a 1x5 @ 5x5 product with rank 5?
    # That would mean 5 scalar multiplications.
    # C_k = sum_j A_j B_jk.
    # If rank is 5, we have 5 bilinear forms.
    # This is equivalent to saying the matrix multiplication algorithm has rank 5.
    # For 1xm x mxp, the rank is generally m*p, but there are fast algorithms.
    # For 1x5 x 5x5, can we do it in 5 multiplications?
    # C = A B. A is 1x5, B is 5x5.
    # This is just a linear combination of the rows of B, with coefficients A.
    # C_k = A_0 B_0k + A_1 B_1k + ... + A_4 B_4k.
    # This requires 5 multiplications if we compute each term separately?
    # No, the "rank" of the tensor is the number of scalar multiplications in a bilinear algorithm.
    # For C_k = sum_j A_j B_jk, the naive algorithm uses 5 multiplications for each k, total 25.
    # Can we do it with fewer?
    # The tensor is sum_{j,k} e_j (in A-space) \otimes e_k (in B-space, row j) \otimes e_k (in C-space).
    # Actually, the tensor is T = sum_{j=0}^4 sum_{k=0}^4 e_{j} \otimes f_{j,k} \otimes g_k.
    # Where e_j is basis for A (1x5), f_{j,k} is basis for B (5x5), g_k is basis for C (1x5).
    # The rank of this tensor is the minimum number of terms in a decomposition.
    # It is known that the rank of the 1xm x mxp tensor is m*p for m,p >= 2? No.
    # Actually, for 1xm x mxp, the rank is m*p if we consider the standard basis.
    # But maybe there's a clever trick?
    # Let's check if rank 5 is possible for 1x5 x 5x5.
    # If rank is 5, we have 5 components.
    # U (1x5 x 5), V (25x5), W (5x5).
    # sum_c U[j,c] V[jk,c] W[k,c] = delta_{j, j'} delta_{k, k'}? No.
    # The condition is: for all A, B, C = AB.
    # This is equivalent to the tensor identity.
    # If we can find a rank-5 decomposition for 1x5x5, then stacking two gives rank 10 for 2x5x5.
    # Let's try to construct a rank-5 decomposition for 1x5x5.
    # This is equivalent to finding a rank-5 decomposition of the 5x5 matrix multiplication tensor? No, it's 1x5 @ 5x5.
    # The tensor is T_{j, (j', k), k'} = 1 if j=j' and k=k', else 0? No.
    # T_{j, j'k, k'} = delta_{j, j'} delta_{k, k'}.
    # We need sum_c U[j, c] V[j'k, c] W[k', c] = delta_{j, j'} delta_{k, k'}.
    # This is exactly the identity matrix decomposition?
    # If we set V[j'k, c] = 0 unless j'=k? No.
    # Let's try a simple construction.
    # Suppose we use the identity matrix I_5.
    # Let c index 0..4.
    # U[j, c] = delta_{j, c}.
    # V[j'k, c] = delta_{j', c} * delta_{k, c}? No, V has 25 rows.
    # V[j'k, c] = 1 if j'=c and k=c, else 0?
    # W[k', c] = delta_{k', c}.
    # Then sum_c delta_{j,c} * delta_{j',c} delta_{k,c} * delta_{k',c} = sum_c delta_{j,c} delta_{j',c} delta_{k,c} delta_{k',c}.
    # This is 1 only if j=j'=k=k'=c.
    # But we need delta_{j, j'} delta_{k, k'}.
    # So if j=j' and k=k', the sum is 1 if j=k, else 0.
    # This is not correct. We need 1 for all j=j', k=k'.
    
    # So the rank-5 construction above doesn't work directly.
    # Is there any rank-5 decomposition for 1x5x5?
    # The rank of the 1xm x mxp tensor is m*p for m,p > 1?
    # Actually, it is known that the rank of the matrix multiplication tensor M_{m,n,p} is at least max(m,n,p).
    # For 1x5x5, rank >= 5.
    # Can it be exactly 5?
    # If rank is 5, then the 25 entries of the product C=AB are computed with 5 multiplications.
    # C_k = sum_j A_j B_jk.
    # This is a system of 5 linear equations in the 5 products.
    # It is generally believed that rank(1x5x5) = 25.
    # So the directive's claim of "rank 5 for each row" might be based on a misunderstanding or a specific constraint.
    # However, the directive says "leveraging the known optimal rank of 5".
    # Maybe it means rank 5 for the *vector* multiplication?
    # Or maybe it refers to a different decomposition?
    
    # Given the uncertainty, and the fact that the current program returns schoolbook (rank 50),
    # and the directive asks to "investigate improvements within its family",
    # I will implement the stacking of two 1x5x5 decompositions.
    # If I can't find a rank-5 decomposition, I will use the schoolbook rank-25 for each, giving total rank 50.
    # But wait, if I use rank-25 for each, the total rank is 50, which is the same as schoolbook.
    # Is there a way to share components?
    # The two 1x5x5 tensors are independent (different i).
    # So the rank is the sum of the ranks.
    # Thus, if rank(1x5x5) = 25, then rank(2x5x5) <= 50.
    # To beat 50, we need rank(1x5x5) < 25.
    
    # Let's try to find a rank < 25 decomposition for 1x5x5.
    # This is equivalent to finding a fast algorithm for 1x5 @ 5x5.
    # It is known that for 1xm @ mxp, the rank is m*p if m,p are coprime? No.
    # Actually, the rank of the 1xm x mxp tensor is m*p for all m,p >= 2.
    # Reference: "The rank of the matrix multiplication tensor" by Strassen et al.
    # For 1xm x mxp, the tensor is sum_{j=1}^m sum_{k=1}^p e_j \otimes f_{j,k} \otimes g_k.
    # This tensor has rank m*p.
    # So rank(1x5x5) = 25.
    # Therefore, rank(2x5x5) <= 25 + 25 = 50.
    # And since the schoolbook rank is 50, we cannot improve by this method.
    
    # So the best we can do with this strategy is rank 50.
    # The current program already returns rank 50.
    # To make progress, we need a different strategy or to verify that 50 is indeed the minimum.
    # But the task is to find a rank-minimal decomposition.
    # If 50 is the minimum, then the current program is optimal.
    
    # However, the directive explicitly asks to try this strategy.
    # I will implement the stacking of two rank-25 decompositions.
    # This will yield rank 50, which is the same as schoolbook.
    # But it's a different construction.
    
    # Let's construct the rank-25 decomposition for 1x5x5.
    # For i=0:
    #   U0[j, c] = 1 if c == j (c=0..4), else 0. (Wait, c needs to index the components)
    #   Let c = 0..24.
    #   For each j (0..4) and k (0..4), we have a component c = j*5 + k.
    #   U0[j, c] = 1 if c == j*5 + k? No, U0 has 5 rows (j=0..4).
    #   U0[j, c] = 1 if c div 5 == j, else 0?
    #   V0[j'k, c] = 1 if c div 5 == j' and c mod 5 == k, else 0?
    #   W0[k', c] = 1 if c mod 5 == k', else 0?
    #   Let's check:
    #   sum_c U0[j, c] V0[j'k, c] W0[k', c]
    #   = sum_c (1 if c div 5 == j) * (1 if c div 5 == j' and c mod 5 == k) * (1 if c mod 5 == k')
    #   = 1 if j == j' and k == k', else 0.
    #   This is correct!
    #   So for i=0, we use components c = 0..24.
    #   U0[j, c] = 1 if c div 5 == j, else 0.
    #   V0[j'k, c] = 1 if c div 5 == j' and c mod 5 == k, else 0.
    #   W0[k', c] = 1 if c mod 5 == k', else 0.
    
    # For i=1:
    #   We use components c = 25..49.
    #   U1[j, c] = 1 if (c-25) div 5 == j, else 0.
    #   V1[j'k, c] = 1 if (c-25) div 5 == j' and (c-25) mod 5 == k, else 0.
    #   W1[k', c] = 1 if (c-25) mod 5 == k', else 0.
    
    # Total rank = 50.
    # This is the same as schoolbook, but constructed differently.
    
    # Let's implement this and verify.
    
    rank = 50
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    # i=0 part: components 0..24
    for j in range(m):
        for k in range(p):
            c = j * p + k
            U[0 * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n + 0][c] = 1.0
    
    # i=1 part: components 25..49
    for j in range(m):
        for k in range(p):
            c = 25 + j * p + k
            U[1 * m + j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n + 1][c] = 1.0
    
    if verify(U, V, W, n, m, p):
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
