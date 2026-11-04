import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    def trivial_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > deadline:
        return trivial_decomp()

    # Build the tensor as a 3D array T[a, b, c]
    # a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[a, b, c] = 1 if exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Use unfolding to find lower bound on rank
    # Unfold along mode 2 (b-axis): matrix of shape (n*m, m*p * p*n)
    # Actually, let's use the slice approach:
    # For each fixed b, T[:, b, :] is a (n*m) x (p*n) matrix
    # The rank of T is at least the max over b of rank(T[:, b, :])
    
    # Let's compute the rank using SVD on unfoldings
    # Unfolding 1: reshape T to (n*m, m*p * p*n)
    U1 = T.reshape(n * m, m * p * p * n)
    r1 = np.linalg.matrix_rank(U1)
    
    # Unfolding 2: reshape T to (m*p, n*m * p*n)
    U2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    r2 = np.linalg.matrix_rank(U2)
    
    # Unfolding 3: reshape T to (p*n, n*m * m*p)
    U3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    r3 = np.linalg.matrix_rank(U3)
    
    lower_bound = max(r1, r2, r3)
    
    if time.time() > deadline:
        return trivial_decomp()
    
    # Now try to find an exact decomposition using rational Gaussian elimination
    # Strategy: express T as sum of rank-1 tensors
    # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    
    # Use the fact that we can fix W and solve for U, V
    # Or use iterative approaches
    
    # Let's try a different approach: use the structure of the problem
    # The tensor has a specific pattern based on (i,j,k) triples
    
    # For 2x3x9, let's try to find a decomposition of rank 18
    # by exploiting the block structure
    
    # Actually, let me try a systematic approach using linear algebra
    # We want to find the minimum r such that T can be written as sum of r rank-1 terms
    
    # Use the "slice rank" approach and then try to construct explicitly
    
    # Let's try to use the known structure:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor entries are T[i*m+j, j*p+k, k*n+i] = 1
    
    # For rank-1 term s: U_s[i*m+j] * V_s[j*p+k] * W_s[k*n+i]
    # We need sum_s U_s[i*m+j] * V_s[j*p+k] * W_s[k*n+i] = 1 for all (i,j,k)
    
    # This is a system of 54 equations in r * (6 + 27 + 18) = r * 51 unknowns
    # (plus the constraint that each term is rank-1)
    
    # Let me try a constructive approach for rank 18:
    # Use the fact that for each j, we have n*p = 18 terms involving that j
    # But we can share structure across j values
    
    # Alternative: try rank = lower_bound and see if we can construct it
    
    # Let's try to use numpy to find a low-rank approximation first, then round
    # Start with the SVD of the unfolding
    
    # Actually, let me try a direct construction for small ranks
    # and verify correctness
    
    def verify(U_mat, V_mat, W_mat):
        """Verify that the decomposition is exact"""
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for s in range(r):
                        val += U_mat[a][s] * V_mat[b][s] * W_mat[c][s]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    if time.time() > deadline:
        return trivial_decomp()
    
    # Try to construct a rank-18 decomposition
    # Key insight: group the k-index into 3 groups of 3
    # For each (i, j, group g, h) where k = 3*g + h:
    # We need U[i*3+j, s] * V[j*9+3*g+h, s] * W[(3*g+h)*2+i, s] to sum to 1
    
    # Let's try: for each j and g, use 3 terms (one for each h)
    # That gives 3 * 3 * 3 = 27 terms... not enough
    
    # Alternative: for each i and g, use 3 terms
    # That gives 2 * 3 * 3 = 18 terms
    
    # For fixed (i, g), the terms involve:
    # (i, j, 3*g+h) for j in {0,1,2}, h in {0,1,2}
    # That's 9 (i,j,k) triples per (i,g), and 3 (i,g) pairs, total 27... 
    # Wait, 2*3 = 6 (i,g) pairs, each with 9 triples = 54 total. Correct.
    
    # For rank 18, I need to share terms across different (i,g) pairs
    
    # Let me try: assign one term to each (j, g, h) combination
    # That's 3 * 3 * 3 = 27 terms. Still too many.
    
    # For rank 18: assign one term to each (j, h) and use structure in i and g
    # 3 * 3 = 9... not enough
    
    # Let me just try to use numerical optimization to find a low-rank decomposition
    # and then round to exact values
    
    # Actually, let me try a smarter constructive approach
    
    # The multiplication tensor for (2,3,9) can be seen as 3 copies of (2,3,3)
    # If (2,3,3) has rank 6, then total rank is 3*6 = 18
    
    # For (2,3,3): C[i,h] = sum_j A[i,j]*B[j,h]
    # Tensor: T[i*3+j, j*3+h, h*2+i] = 1 for i in {0,1}, j in {0,1,2}, h in {0,1,2}
    # That's 18 ones. Can we do rank 6?
    
    # Let's check the rank of the (2,3,3) tensor
    T233 = np.zeros((6, 9, 6), dtype=np.float64)
    for i in range(2):
        for j in range(3):
            for h in range(3):
                a = i * 3 + j
                b = j * 3 + h
                c = h * 2 + i
                T233[a, b, c] = 1.0
    
    # Check unfoldings
    r233_1 = np.linalg.matrix_rank(T233.reshape(6, 54))
    r233_2 = np.linalg.matrix_rank(T233.transpose(1,0,2).reshape(9, 36))
    r233_3 = np.linalg.matrix_rank(T233.transpose(2,0,1).reshape(6, 54))
    
    if time.time() > deadline:
        return trivial_decomp()
    
    # For (2,3,3), the lower bound is max(r233_1, r233_2, r233_3)
    # If this is 6, then rank 18 for (2,3,9) is achievable
    
    # Let me try to construct the (2,3,3) rank-6 decomposition explicitly
    # and then replicate it 3 times
    
    # For (2,3,3), try rank 6:
    # We need 6 rank-1 terms that sum to the tensor
    
    # Let's use a parametric approach
    # Term s: U_s[a] * V_s[b] * W_s[c]
    
    # Try: for each (j, h) pair, define a term
    # U_s[i*3+j] = alpha_{i,j}, V_s[j*3+h] = beta_{j,h}, W_s[h*2+i] = gamma_{h,i}
    
    # We need: sum_{(j',h')} alpha_{i,j'} * beta_{j',h'} * gamma_{h',i} = 1 for all (i,j,h)
    # Wait, that's not quite right. The term (j,h) contributes to all (i,j',h') where...
    
    # Let me think differently. For the (2,3,3) tensor:
    # T[a,b,c] = 1 if a=i*3+j, b=j*3+h, c=h*2+i for some (i,j,h)
    
    # A rank-1 term is specified by vectors u (length 6), v (length 9), w (length 6)
    # Its contribution to T[a,b,c] is u[a]*v[b]*w[c]
    
    # We need sum_s u_s[a]*v_s[b]*w_s[c] = T[a,b,c]
    
    # Let me try to use the structure: group by j
    # For fixed j, the entries are at (i*3+j, j*3+h, h*2+i) for i in {0,1}, h in {0,1,2}
    # That's 6 entries per j, 18 total
    
    # If I use 2 terms per j (one for each i), that's 6 terms total
    # Term for (j, i): u[a] nonzero only at a=i*3+j, v[b] nonzero only at b=j*3+h for h in {0,1,2}, w[c] nonzero only at c=h*2+i
    
    # But v and w need to be constant over their support for this to work as rank-1
    
    # Let me try: for each (j, i), define:
    # u_{j,i}[i*3+j] = 1, u_{j,i}[other] = 0
    # v_{j,i}[j*3+h] = 1 for h in {0,1,2}, v_{j,i}[other] = 0
    # w_{j,i}[h*2+i] = 1 for h in {0,1,2}, w_{j,i}[other] = 0
    
    # Then the contribution of term (j,i) to T[a,b,c] is:
    # 1 if a=i*3+j AND b=j*3+h for some h AND c=h*2+i for some h
    # But we need b=j*3+h and c=h*2+i to use the SAME h
    
    # This doesn't work directly because the h is coupled
    
    # Let me try a different decomposition for (2,3,3)
    # Use 3 terms, one for each h:
    # Term h: u_h[a], v_h[b], w_h[c]
    
    # For term h, we want it to contribute to entries where the h-index is h
    # Entries with h-index h: (i*3+j, j*3+h, h*2+i) for i in {0,1}, j in {0,1,2}
    
    # u_h[i*3+j] should be nonzero for all i,j
    # v_h[j*3+h] should be nonzero for all j
    # w_h[h*2+i] should be nonzero for all i
    
    # Let u_h[i*3+j] = A_{i,j}, v_h[j*3+h] = B_{j,h}, w_h[h*2+i] = C_{h,i}
    # We need sum_h A_{i,j} * B_{j,h} * C_{h,i} = 1 for all (i,j)
    # And the term should NOT contribute to other (i',j') pairs
    
    # This is getting complicated. Let me just use numerical methods to find a decomposition
    
    if time.time() > deadline:
        return trivial_decomp()
    
    # Use a randomized approach to find a low-rank decomposition
    # Start with the SVD and try to round
    
    # Get the best rank-r approximation via truncated SVD
    # Then try to find exact rational values
    
    # For the full (2,3,9) tensor, let's try rank 18
    # Use the unfolding to initialize
    
    # Actually, let me just return the trivial decomposition for now
    # and mark this as needing more work
    
    return trivial_decomp()
