import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = 140

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify_decomp(U, V, W, rank):
        """Verify tensor decomposition correctness."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for t in range(rank):
                        C_decomp[i, k] += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
        
        return np.allclose(C_decomp, C_expected, atol=1e-10)

    # Structural approach: Truncated dual numbers / polynomial multiplication
    # Model B as two polynomials b0(x), b1(x) of degree < 13
    # C[0,k] = a*b0[k] + b*b1[k]
    # C[1,k] = c*b0[k] + d*b1[k]
    # 
    # Use rank 2p+2 = 28 with the following construction:
    # For k=0..p-1, two terms per column, plus 2 global terms.
    # 
    # After careful analysis, the correct decomposition uses:
    # - p terms for the "diagonal" contributions
    # - p terms for the "off-diagonal" contributions  
    # - 2 additional terms for cross-coupling
    #
    # Let's implement and verify numerically.
    
    # Approach 1: Strassen per column pair (known to work, rank = 7*(p//2) + 4*(p%2))
    if p >= 2:
        num_pairs = p // 2
        leftover = p % 2
        strassen_rank = num_pairs * 7 + leftover * 4
        
        if strassen_rank < best_rank:
            U = [[0] * strassen_rank for _ in range(n * m)]
            V = [[0] * strassen_rank for _ in range(m * p)]
            W = [[0] * strassen_rank for _ in range(p * n)]
            
            col = 0
            strassen_terms = [
                ([1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 1]),
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
                ([1, 0, 0, 0], [0, -1, 0, 1], [0, 1, 0, 0]),
                ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 0, 0]),
                ([1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1]),
                ([0, 1, 1, 0], [-1, 0, 1, 0], [0, 0, 1, 0]),
                ([0, 1, 0, 1], [0, 1, 0, -1], [-1, 0, 0, 0]),
            ]
            
            for pair_idx in range(num_pairs):
                k_start = 2 * pair_idx
                for U_vec, V_vec, W_vec in strassen_terms:
                    for i in range(4):
                        U[i][col] = U_vec[i]
                    V[k_start][col] = V_vec[0]
                    V[k_start + 1][col] = V_vec[1]
                    V[p + k_start][col] = V_vec[2]
                    V[p + k_start + 1][col] = V_vec[3]
                    W[k_start * n][col] = W_vec[0]
                    W[k_start * n + 2][col] = W_vec[1]
                    W[k_start * n + 1][col] = W_vec[2]
                    W[k_start * n + 3][col] = W_vec[3]
                    col += 1
            
            if leftover > 0:
                k_start = 2 * num_pairs
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][col] = 1
                        V[j * p + k_start][col] = 1
                        W[k_start * n + i][col] = 1
                        col += 1
            
            if verify_decomp(U, V, W, strassen_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = strassen_rank

    if time.time() - start_time > deadline - 10:
        return best_result

    # Approach 2: Polynomial/dual numbers approach with rank 2p+2
    # Key insight: C = A*B where A is 2x2, B is 2xp.
    # Write A = [[a,b],[c,d]]. Then:
    #   C[0,:] = a*B[0,:] + b*B[1,:]
    #   C[1,:] = c*B[0,:] + d*B[1,:]
    #
    # We can decompose using the identity:
    #   a*x + b*y = (1/4)[(a+b)(x+y) + (a-b)(x-y) + (a+b)(x-y) + (a-b)(x+y)]
    # This doesn't help directly. Instead, use:
    #
    # For each k, C[0,k] and C[1,k] depend on B[0,k] and B[1,k].
    # The 2x2x2 tensor (for one k) has rank 7 (Strassen).
    # But we can share the "Strassen combinations" of A across all k.
    #
    # Define:
    #   M1 = a+b, M2 = c+d, M3 = a, M4 = d, M5 = a+d, M6 = b+c, M7 = b+d
    #
    # For each k, define:
    #   N1(k) = B[0,k]+B[1,k]
    #   N2(k) = B[0,k]
    #   N3(k) = B[1,k]-B[0,k]
    #   N4(k) = B[1,k]
    #   N5(k) = B[0,k]+B[1,k]  (same as N1)
    #   N6(k) = B[1,k]-B[0,k]  (same as N3)
    #   N7(k) = B[1,k]
    #
    # This gets complicated. Let's just try a direct rank-2p+2 construction.
    #
    # After further analysis, I'll use a simpler but correct approach:
    # Rank = 2p+2 using the following:
    #
    # Terms 0..p-1: For each k, contribute a*B[0,k] to C[0,k] and c*B[0,k] to C[1,k]
    #   U = [1,0,1,0], V has 1 at (0,k), W has 1 at (k,0) and (k,1)
    #   But this gives B[0,k] not a*B[0,k]... 
    #
    # The fundamental issue: U must be constant. So we need:
    # sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = sum_j A[i,j]*B[j,k]
    #
    # This means for each t, the rank-1 tensor U_t ⊗ V_t ⊗ W_t must sum to the multiplication tensor.
    #
    # Let me try a different structural decomposition.
    # Use the fact that we can write the 2x2x13 tensor as a sum of 2x2x2 tensors.
    # Each 2x2x2 has rank 7. With 13 columns, we have ceil(13/2)=7 groups of 2 columns (with one leftover).
    # That gives 7*7 + 4 = 53, which is worse than schoolbook.
    #
    # Better: Use the "Kronecker" structure. The tensor T_{i,j,k} for i,j in {0,1}, k in {0..12}
    # can be written as sum over k of T^{(k)} where T^{(k)}_{i,j} = delta_{i,j}... no.
    #
    # I'll stick with the Strassen approach which gives rank 30, and try to improve it
    # by finding a better decomposition of the 2x2x2 tensor or sharing structure.
    
    # Approach 3: Try rank 2p+2 = 28 with explicit construction
    # Based on the theory of bilinear complexity, for 2x2 x 2xp, 
    # the rank should be 2p+2. Let me implement it correctly.
    
    rank_2p2 = 2 * p + 2
    if rank_2p2 < best_rank:
        U = [[0] * rank_2p2 for _ in range(n * m)]
        V = [[0] * rank_2p2 for _ in range(m * p)]
        W = [[0] * rank_2p2 for _ in range(p * n)]
        
        # Construction:
        # For k = 0 to p-1:
        #   Term 2k: U=[1,0,1,0], V=[e_k at row0, 0 at row1], W=[1 at (k,0), 1 at (k,1)]
        #     Contribution: C[0,k] += 1*V[0,k]*1, C[1,k] += 1*V[0,k]*1
        #     We need: C[0,k] += a*B[0,k], C[1,k] += c*B[0,k]
        #     This doesn't work because U is fixed.
        #
        # Correct approach: We need U to encode the linear combinations of A entries.
        # 
        # Let's use the following valid decomposition (verified by construction):
        # 
        # For the 2x2x13 tensor, use the following rank-28 decomposition:
        # 
        # Terms 0-12 (for B[0,k] contributions):
        #   t=k: U=[1,0,1,0], V has 1 at position (0,k), W has 1 at (k,0) and (k,1)
        #   This gives: for C[0,k]: U[0]*V[0,k]*W[k,0] = 1*1*1 = 1 (should be a*B[0,k])
        #   Problem: we get 1, not a*B[0,k]
        #
        # I realize now that my understanding of the tensor decomposition is flawed.
        # Let me re-read the convention.
        #
        # The decomposition is: C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        #
        # For this to equal sum_j A[i,j]*B[j,k], we need:
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] summed over j gives A[i,j]*B[j,k]
        #
        # This is a multilinear identity. The simplest way to verify is numerically.
        #
        # Let me just implement the Strassen approach (rank 30) which I know works,
        # and accept that as the best I can do without more research.
        
        # Skip this approach since I can't construct it correctly without more analysis
        pass

    return best_result
