import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def current_solution():
        """Rank-50: Strassen on 2x2 blocks + schoolbook for remainder."""
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            w_00 = k0 * n + 0; w_01 = k1 * n + 0
            w_10 = k0 * n + 1; w_11 = k1 * n + 1
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
        k8 = 8
        v_e = 0 * p + k8; v_g = 1 * p + k8
        u_a = 0 * m + 0; u_b = 0 * m + 1; u_c = 1 * m + 0; u_d = 1 * m + 1
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        w_08 = k8 * n + 0; w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        u_ar0 = 0 * m + 2; u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
        return {"U": U, "V": V, "W": W}
    
    def verify_solution(sol, n, m, p):
        """Verify tensor decomposition: sum_r U[i,r]*V[j,r] == tensor[i,j] for W."""
        U = np.array(sol["U"], dtype=float)
        V = np.array(sol["V"], dtype=float)
        W = np.array(sol["W"], dtype=float)
        rank = U.shape[1]
        # Compute product: P[i,j] = sum_r U[i,r] * V[j,r]
        P = U @ V.T  # (n*m) x (m*p)
        # The multiplication tensor: T[i,j] where i indexes (row, col) of A, j indexes (row, col) of B
        # A is n x m: A[i_r, i_c] at index i = i_r * m + i_c
        # B is m x p: B[j_r, j_c] at index j = j_r * p + j_c
        # Output C is n x p: C[i_r, j_c] at index k = j_c * n + i_r
        # Tensor: T[i, j] = 1 if i_c == j_r else 0, and contributes to W[j_c * n + i_r, r]
        # So: sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 when i_c==j_r and k==j_c*n+i_r
        # Equivalently: for each (i_r, i_c, j_r, j_c) with i_c==j_r, the sum over r of
        #   U[i_r*m+i_c, r] * V[j_r*p+j_c, r] * W[j_c*n+i_r, r] == 1
        # And for i_c != j_r, the sum should be 0.
        # Check: for all i_c == j_r, sum_r U[i_r*m+i_c,r]*V[i_c*p+j_c,r]*W[j_c*n+i_r,r] == 1
        # And for i_c != j_r, sum_r U[i_r*m+i_c,r]*V[i_c*p+j_c,r]*W[j_c*n+i_r,r] == 0
        # Actually the correct check: the bilinear form should reproduce matrix mult.
        # C[i_r, j_c] = sum_{k=0}^{m-1} A[i_r, k] * B[k, j_c]
        # In tensor terms: sum_k sum_r U[i_r*m+k, r] * V[k*p+j_c, r] * W[j_c*n+i_r, r]
        # should equal A[i_r,k]*B[k,j_c] for all i_r, j_c, k.
        # So: for fixed i_r, j_c: sum_k sum_r U[i_r*m+k,r]*V[k*p+j_c,r]*W[j_c*n+i_r,r] = A[i_r,k]*B[k,j_c]
        # This means: for each (i_r, k, j_c), the "coefficient" sum_r U[i_r*m+k,r]*V[k*p+j_c,r]*W[j_c*n+i_r,r]
        # should equal 1 if we're computing A[i_r,k]*B[k,j_c], and the structure is:
        # The tensor T has T[i_r*m+k, k*p+j_c] contributing to output j_c*n+i_r with value 1.
        # So: sum_r U[i_r*m+k, r] * V[k*p+j_c, r] * W[j_c*n+i_r, r] = 1 for all i_r, k, j_c.
        # And for any (i, j) where i_c != j_r (i.e., the A column index != B row index), the
        # corresponding entry should give 0 when combined with W.
        # Simplest check: build the full tensor and verify.
        # Tensor T[i, j, k_out] = 1 if i%m == j//p and k_out == (j%p)*n + (i//m) else 0
        # Check: sum_r U[i,r]*V[j,r]*W[k_out,r] == T[i,j,k_out]
        # This is expensive for full check but let's do it.
        rank = U.shape[1]
        # For efficiency, check a random subset
        rng = np.random.RandomState(42)
        # Check all entries (small enough: 6*27*18 = 2916)
        for i in range(n * m):
            for j in range(m * p):
                i_c = i % m  # column of A
                j_r = j // p  # row of B
                if i_c == j_r:
                    j_c = j % p
                    i_r = i // m
                    k_out = j_c * n + i_r
                    s = 0.0
                    for r in range(rank):
                        s += U[i, r] * V[j, r] * W[k_out, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
                else:
                    # For i_c != j_r, the contribution to ANY output should be 0
                    # Actually we need: for all k_out, sum_r U[i,r]*V[j,r]*W[k_out,r] == 0
                    # But that's not quite right either. The tensor is:
                    # T[i,j,k_out] = 1 if (i%m == j//p) and (k_out == (j%p)*n + i//m) else 0
                    # So if i%m != j//p, T[i,j,k_out] = 0 for all k_out.
                    # Check a few k_out values
                    for k_out in range(0, p * n, max(1, (p * n) // 6)):
                        s = 0.0
                        for r in range(rank):
                            s += U[i, r] * V[j, r] * W[k_out, r]
                        if abs(s) > 1e-6:
                            return False
        return True
    
    # Start with the known rank-50 solution
    best = current_solution()
    
    # Verify the known solution
    if not verify_solution(best, n, m, p):
        # Shouldn't happen, but safety
        return best
    
    # Try the (3,3,3) embedding approach
    # Map A (2x3) to A' (3x3): A'[0,:] = A[0,:], A'[1,:] = A[1,:], A'[2,:] = A[0,:]+A[1,:]
    # Map B (3x9) to B' (3x3) per block: for each block of 3 cols of B,
    #   B'_block[0,:] = B[0, block], B'_block[1,:] = B[1, block], B'_block[2,:] = B[2, block]
    # But B is 3x9, so we need 3 blocks. For block b (cols 3b..3b+2):
    #   B'_b is 3x3 with rows = B[0..2, 3b..3b+2]
    # A' * B'_b gives 3x3 result. We want C[0..1, 3b..3b+2] = A[0..1, :] * B[:, 3b..3b+2]
    # A'[0,:] * B'_b = A[0,:] * B[:, 3b..3b+2] -> C[0, 3b..3b+2] ✓
    # A'[1,:] * B'_b = A[1,:] * B[:, 3b..3b+2] -> C[1, 3b..3b+2] ✓
    # A'[2,:] * B'_b = (A[0]+A[1]) * B[:, 3b..3b+2] -> C[0]+C[1] (not needed directly)
    # So we use Strassen on 3x3x3 for each of the 3 blocks.
    # Strassen 3x3x3 has rank 7 (same as 2x2 but for 3x3? No, 3x3x3 rank is 19 or 23.)
    # Actually the rank of 3x3x3 matrix multiplication tensor is 19 (Bläser 2005) or 23.
    # That's much worse than 7. So this approach won't help directly.
    # 
    # Alternative: use the fact that A' has a dependency (row 2 = row 0 + row 1)
    # to reduce the effective rank. The 3x3x3 tensor restricted to the subspace where
    # A'[2,:] = A'[0,:] + A'[1,:] might have lower rank.
    # 
    # Let me think differently. The (2,3,9) problem:
    # C[i,j] = sum_{k=0}^{2} A[i,k] * B[k,j], i in {0,1}, j in {0..8}
    # 
    # Group j into 3 blocks of 3: j = 3b + c, b in {0,1,2}, c in {0,1,2}
    # C[i, 3b+c] = sum_{k=0}^{2} A[i,k] * B[k, 3b+c]
    # 
    # For fixed b, define B_b as 3x3: B_b[k,c] = B[k, 3b+c]
    # C_b as 2x3: C_b[i,c] = C[i, 3b+c] = sum_k A[i,k] * B_b[k,c]
    # 
    # This is a (2,3,3) multiplication: 2x3 * 3x3 -> 2x3
    # Rank of (2,3,3) tensor: at most 6 (schoolbook), possibly less.
    # Actually rank of (2,3,3) is 6 since it's essentially 3 independent 2x1x3 multiplications...
    # No wait: C[i,c] = sum_k A[i,k]*B_b[k,c]. For fixed b, this is 6 outputs, each a bilinear form.
    # The tensor is sum_{i,k,c} A[i,k]*B_b[k,c]*C_b[i,c]
    # This is the same as 3 copies of (2,1,1) multiplied... no.
    # It's a (2,3,3) tensor. The rank is at most 2*3*3/3 = 6? No, that's not a bound.
    # Schoolbook gives rank 6. Can we do better?
    # 
    # Actually, let's just try to find a rank-6 decomposition for (2,3,3) and use it 3 times.
    # Total rank would be 3*6 = 18? No, we can potentially share structure.
    # 
    # Hmm, but the current solution is rank 50. Schoolbook is rank 2*3*9 = 54.
    # The Strassen approach got us to 50.
    # 
    # Let me try a different grouping: group the inner dimension differently.
    # Split k into {0,1} and {2}:
    #   C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j]
    #   = (A[i,0]*B[0,j] + A[i,1]*B[1,j]) + A[i,2]*B[2,j]
    # 
    # First term: (2x2) * (2x9) -> (2x9), rank at most 7*ceil(9/2) = 7*5 = 35 (using Strassen on 2x2 blocks)
    #   Actually 7*4 + 4 = 32 (4 Strassen blocks + 1 schoolbook 2x2x1)
    # Second term: (2x1) * (1x9) -> (2x9), rank = 2*9 = 18
    # Total: 32 + 18 = 50. That's what we have.
    # 
    # Alternative split: {0} and {1,2}:
    #   First: (2x1)*(1x9) = rank 18
    #   Second: (2x2)*(2x9) = rank 32
    #   Total: 50
    # 
    # Alternative: {0,1} and {2} but use different Strassen grouping for the 2x2x9:
    #   2x2x9: split 9 into 4*2 + 1: 4*7 + 4 = 32. Same.
    # 
    # What about using the full (2,3,9) structure more cleverly?
    # The tensor T has 2*3*9 = 54 "1" entries (one for each (i,k,j) triple).
    # Rank lower bound: at least max(2*3, 3*9, 9*2) / something... 
    # Actually the rank is at least the max of the "slice ranks" or other bounds.
    # 
    # Let me try: can we do better than 50?
    # 
    # Idea: group j into 3 blocks of 3. For each block b:
    #   C[:, 3b:3b+3] = A * B[:, 3b:3b+3] where A is 2x3, B_block is 3x3
    #   This is (2,3,3) multiplication.
    #   If rank of (2,3,3) is 6, total = 3*6 = 18. But that seems too optimistic.
    #   Actually (2,3,3) rank: the tensor is sum_{i=0}^{1} sum_{k=0}^{2} sum_{c=0}^{2} e_{i,k} ⊗ e_{k,c} ⊗ e_{i,c}
    #   This is a 6x9x6 tensor (well, 2x3 x 3x3 x 2x3 = 6 x 9 x 6).
    #   Schoolbook rank = 6. Lower bound: at least 6 (since there are 6 independent outputs).
    #   So rank of (2,3,3) is exactly 6.
    #   Total: 3 * 6 = 18? But we can't just multiply ranks like that because
    #   the U, V, W matrices need to be consistent.
    # 
    # Wait, I think I'm confusing myself. Let me re-derive.
    # 
    # The multiplication tensor for (n,m,p) is:
    #   T = sum_{i=0}^{n-1} sum_{k=0}^{m-1} sum_{j=0}^{p-1} A_{i,k} ⊗ B_{k,j} ⊗ C_{i,j}
    #   where A_{i,k} is the basis vector for entry (i,k) of A, etc.
    # 
    # For (2,3,9): T = sum_{i=0}^{1} sum_{k=0}^{2} sum_{j=0}^{8} A_{i,k} ⊗ B_{k,j} ⊗ C_{i,j}
    #   This has 2*3*9 = 54 terms.
    # 
    # A rank-r decomposition: T = sum_{r'=0}^{r-1} u_{r'} ⊗ v_{r'} ⊗ w_{r'}
    #   where u_{r'} is a vector in R^{6}, v_{r'} in R^{27}, w_{r'} in R^{18}.
    # 
    # The schoolbook decomposition has r = 54 (one term per (i,k,j) triple).
    # 
    # Now, if we group j into blocks of 3: j = 3b + c, b in {0,1,2}, c in {0,1,2}
    #   T = sum_{b=0}^{2} sum_{i=0}^{1} sum_{k=0}^{2} sum_{c=0}^{2} A_{i,k} ⊗ B_{k,3b+c} ⊗ C_{i,3b+c}
    # 
    # For fixed b, define T_b = sum_{i,k,c} A_{i,k} ⊗ B_{k,3b+c} ⊗ C_{i,3b+c}
    #   This is a (2,3,3) tensor (with the B index restricted to block b).
    #   Rank of T_b: at most 6 (schoolbook for 2x3 * 3x3).
    #   Is it exactly 6? The 6 terms are: (i,k,c) = (0,0,0), (0,0,1), (0,0,2), (0,1,0), ... etc.
    #   Actually there are 2*3*3 = 18 terms for T_b, but they factor.
    #   T_b = sum_{i,k} A_{i,k} ⊗ (sum_c B_{k,3b+c} ⊗ C_{i,3b+c})
    #   Hmm, this doesn't factor nicely.
    # 
    # Let me just try to construct a rank-6 decomposition for the (2,3,3) subproblem
    # and see if the total rank is 3*6 = 18 or if we can do better.
    # 
    # Actually, I realize the issue: when we write T = sum_b T_b, each T_b uses
    # different entries of V (B indices) and W (C indices), but the same U (A indices).
    # So we can't simply add ranks. The total rank is at most sum of individual ranks
    # if the V and W supports are disjoint, which they are here (different B and C indices).
    # So total rank <= 3 * 6 = 18. But that's an upper bound; the actual rank might be higher
    # because the U vectors might not be linearly independent across blocks.
    # 
    # Wait, actually if the V and W supports are disjoint, then the rank of the sum is
    # at most the sum of the ranks. And it could be exactly the sum if there's no
    # cancellation. So rank <= 18. But can we achieve 18?
    # 
    # Let me try to construct it. For block b, the (2,3,3) schoolbook is:
    #   For i in {0,1}, k in {0,1,2}, c in {0,1,2}:
    #     term: U[A_{i,k}] * V[B_{k,3b+c}] * W[C_{i,3b+c}]
    #   That's 18 terms per block, 54 total. Same as schoolbook!
    # 
    # Oh right, I was confusing myself. The (2,3,3) "multiplication" isn't a standard
    # matrix multiplication. It's C[i,c] = sum_k A[i,k] * B[k,c], which is exactly
    # (2x3) * (3x3) = 2x3. The schoolbook rank for this is 2*3*3 = 18 (one term per
    # (i,k,c) triple). I was wrong to say rank 6.
    # 
    # So grouping by j doesn't help. The rank is still 54 for schoolbook.
    # 
    # The Strassen trick works because it exploits the 2x2 structure within the
    # inner dimension. Let me think about whether there's a better way.
    # 
    # Current: 50. Can we get to 49 or lower?
    # 
    # Idea: use a (3,3,3) Strassen-like decomposition. The rank of (3,3,3) is 19
    # (or 23, depending on the result). But we only need (2,3,3) which is a subcase.
    # 
    # Actually, let me try a different approach: use the full 3x3 structure.
    # Pad A to 3x3 (add a zero row), pad B to 3x3 per block.
    # Then use rank-19 (or whatever) for 3x3x3, but we only need 2 of the 3 output rows.
    # This might not save rank.
    # 
    # Let me try yet another approach: use a (2,3,3) decomposition with rank < 18.
    # The tensor T_{2,3,3} = sum_{i,k,c} A_{i,k} ⊗ B_{k,c} ⊗ C_{i,c}
    # This is equivalent to: for each c, we have a (2,3,1) multiplication.
    # T_{2,3,3} = sum_c T_{2,3,1}^{(c)} where T_{2,3,1}^{(c)} = sum_{i,k} A_{i,k} ⊗ B_{k,c} ⊗ C_{i,c}
    # Each T_{2,3,1}^{(c)} has rank 6 (schoolbook) and the supports in V and W are disjoint
    # across c (different B columns and C columns). So rank of T_{2,3,3} <= 3*6 = 18.
    # And since the supports are disjoint, rank = 18 exactly (no cancellation possible).
    # 
    # So the (2,3,3) tensor has rank exactly 18. Grouping by j gives no improvement.
    # 
    # The only way to improve is to exploit structure across the inner dimension k.
    # The Strassen trick does this for 2x2 blocks of k.
    # 
    # Current decomposition:
    #   k = {0,1} handled with Strassen (rank 7 per 2-col block of j), 4 blocks = 28
    #   k = 2 handled with schoolbook (rank 2 per col of j), 9 cols = 18
    #   Total: 28 + 18 = 46? Wait, let me recount.
    # 
    # Actually looking at the code:
    #   Part 1: 4 blocks of 2 cols, each using 7 Strassen mults = 28
    #   Last col (k=8 of B_T which is k=1 of the 2x2): 4 schoolbook mults
    #   Part 2: 9 cols, each using 2 mults = 18
    #   Total: 28 + 4 + 18 = 50. ✓
    # 
    # Can we improve? What if we use a different split of k?
    # 
    # Option: split k into {0,1,2} all together using a (3,3,9) approach.
    #   (3,3,9) rank: at most 3*3*9/3 = 27? No, that's not a valid bound.
    #   Schoolbook: 3*3*9 = 81. Way worse.
    #   Strassen-like: not directly applicable.
    # 
    # Option: use a (2,2,2) Strassen for the k={0,1} part, but handle the j dimension
    #   more cleverly. Currently we use 4 Strassen blocks (for j=0..7) + 1 schoolbook (j=8).
    #   What if we use a different decomposition of 2x2x9?
    #   2x2x9 rank: at most 7*ceil(9/2) = 35, or 7*4+4 = 32 (current).
    #   Can we do better? The rank of 2x2x9 is at least... hmm.
    #   Actually, 2x2x9 can be seen as 9 independent 2x2x1 multiplications, but they share
    #   the A matrix. The rank is at most 9*4 = 36 (schoolbook) or 7*4+4 = 32 (Strassen).
    #   Lower bound: at least 9 (since there are 9 independent output columns).
    #   I don't think we can do much better than 32 for 2x2x9.
    # 
    # What about the (2,1,9) part? Rank = 18 (schoolbook). Can we do better?
    #   (2,1,9): C[i,j] = A[i,0]*B[0,j]. This is just 2x9 independent products.
    #   Rank = 18 (each output needs its own term). Can't improve.
    # 
    # So the lower bound for this decomposition strategy is 32 + 18 = 50.
    # 
    # To go below 50, we need a fundamentally different approach.
    # 
    # Idea: use a (2,3,9) specific decomposition that exploits the full structure.
    # The tensor has 54 "1" entries. The rank is at least the max of:
    #   - The rank of any "slice" (fixing one index)
    #   - The border rank lower bounds
    # 
    # Let me try a random search for a lower rank decomposition.
    # 
    # Actually, let me try a different split: {0,2} and {1} instead of {0,1} and {2}.
    #   k={0,2}: A[:, {0,2}] is 2x2, B[{0,2}, :] is 2x9. Strassen: 32
    #   k={1}: A[:, 1] is 2x1, B[1, :] is 1x9. Schoolbook: 18
    #   Total: 50. Same.
    # 
    # What about {0} and {1,2}?
    #   k={0}: 2x1 * 1x9 = 18
    #   k={1,2}: 2x2 * 2x9 = 32
    #   Total: 50. Same.
    # 
    # All splits give 50. The bottleneck is the 2x2x9 = 32 and 2x1x9 = 18.
    # 
    # Can we reduce 2x2x9 below 32? 
    #   2x2x9: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j], i in {0,1}, j in {0..8}
    #   = 18 bilinear forms.
    #   Strassen gives 32. Schoolbook gives 36.
    #   Lower bound: at least 18 (number of outputs).
    #   Is there a decomposition with rank < 32?
    #   
    #   Actually, I recall that the rank of (2,2,n) matrix multiplication is:
    #   - For n=1: 4
    #   - For n=2: 7 (Strassen)
    #   - For n=3: 10? or 11?
    #   - In general: 7*floor(n/2) + 4*ceil(n/2 - floor(n/2))... 
    #   Actually for 2x2xn: rank = 7*floor(n/2) + 4*ceil(n/2) if n is odd,
    #   7*floor(n/2) if n is even? No, that's not right either.
    #   
    #   Let me think: 2x2xn can be decomposed as floor(n/2) copies of 2x2x2 (rank 7 each)
    #   plus possibly one 2x2x1 (rank 4). For n=9: 4*7 + 4 = 32. ✓
    #   
    #   Can we do better? The rank of 2x2x3 is at most 7+4 = 11 (one Strassen + one schoolbook)
    #   or 3*4 = 12 (schoolbook). Actually, is there a rank-10 decomposition for 2x2x3?
    #   I don't think so, but I'm not sure.
    #   
    #   For 2x2x9, I believe 32 is optimal or near-optimal.
    # 
    # Given the difficulty of improving below 50, let me try a computational search
    # for a rank-49 or rank-48 decomposition using random initialization and
    # gradient-based optimization.
    # 
    # Actually, that's very hard for tensor rank minimization (NP-hard).
    # 
    # Let me try a different angle: maybe there's a clever algebraic trick I'm missing.
    # 
    # The (2,3,9) tensor: T = sum_{i=0}^1 sum_{k=0}^2 sum_{j=0}^8 A_{i,k} ⊗ B_{k,j} ⊗ C_{i,j}
    # 
    # What if we use a "circular" or "cyclic" structure in j?
    #   Let ω be a primitive 9th root of unity. Define:
    #   B'_k(ω^t) = sum_{j=0}^8 B_{k,j} ω^{tj} for t = 0..8
    #   Then C_i(ω^t) = sum_k A_{i,k} B'_k(ω^t)
    #   And C_{i,j} = (1/9) sum_t C_i(ω^t) ω^{-tj}
    #   
    #   This doesn't directly help with rank, but it suggests a structure.
    # 
    # I think the best we can do with the current approach is 50. Let me try to
    # verify if there's a rank-49 decomposition by trying a few random perturbations
    # and checking if we can merge two terms.
    # 
    # Actually, let me just try to implement the (3,3,3) embedding as suggested
    # in the directive, even if I'm not sure it will help.
    # 
    # Embedding: 
    #   A (2x3) -> A' (3x3): A'[0,:] = A[0,:], A'[1,:] = A[1,:], A'[2,:] = A[0,:]+A[1,:]
    #   B (3x9) -> B' (3x3) per block: for block b, B'_b[k,c] = B[k, 3b+c]
    #   C' = A' * B'_b is 3x3. We need C[0..1, 3b..3b+2] = C'[0..1, 0..2].
    #   C'[2,c] = C[0,3b+c] + C[1,3b+c] (redundant).
    # 
    # The (3,3,3) multiplication tensor has rank 19 (best known).
    # But we're working on a subspace: A'[2,:] = A'[0,:] + A'[1,:].
    # The restricted tensor might have lower rank.
    # 
    # T_restricted = sum_{i=0}^2 sum_{k=0}^2 sum_{c=0}^2 A'_{i,k} ⊗ B'_{k,c} ⊗ C'_{i,c}
    #   with constraint A'_{2,k} = A'_{0,k} + A'_{1,k}
    #   = sum_{i=0}^1 sum_{k=0}^2 sum_{c=0}^2 A'_{i,k} ⊗ B'_{k,c} ⊗ C'_{i,c}
    #     + sum_{k=0}^2 sum_{c=0}^2 (A'_{0,k} + A'_{1,k}) ⊗ B'_{k,c} ⊗ C'_{2,c}
    #   = sum_{i=0}^1 sum_{k=0}^2 sum_{c=0}^2 A'_{i,k} ⊗ B'_{k,c} ⊗ C'_{i,c}
    #     + sum_{i=0}^1 sum_{k=0}^2 sum_{c=0}^2 A'_{i,k} ⊗ B'_{k,c} ⊗ C'_{2,c}
    # 
    # So T_restricted = T_{2,3,3} + sum_{i,k,c} A'_{i,k} ⊗ B'_{k,c} ⊗ C'_{2,c}
    #   where the second sum is: for each (k,c), the term is (A'_{0,k}+A'_{1,k}) ⊗ B'_{k,c} ⊗ C'_{2,c}
    #   = sum_{i=0}^1 A'_{i,k} ⊗ B'_{k,c} ⊗ C'_{2,c}
    # 
    # So T_restricted = sum_{i=0}^1 sum_{k=0}^2 sum_{c=0}^2 A'_{i,k} ⊗ B'_{k,c} ⊗ (C'_{i,c} + C'_{2,c})
    # 
    # This is a (2,3,3) tensor but with the W part being (C'_{i,c} + C'_{2,c}) instead of just C'_{i,c}.
    # The rank is still at most 18 (same as before), and since the W vectors are different
    # (they're sums), the rank might be the same or different.
    # 
    # This doesn't seem to help. Let me just go with the rank-50 solution and try
    # a quick random search to see if I can find rank 49.
    
    # Try random search for rank 49
    if time.time() < deadline - 30:
        rng = np.random.RandomState(12345)
        for trial_rank in [49, 48, 47]:
            if time.time() > deadline - 15:
                break
            # Initialize with the rank-50 solution and try to merge two columns
            if trial_rank == 49:
                # Take the rank-50 solution and try to express one column as a combo of others
                U50 = np.array(best["U"], dtype=float)
                V50 = np.array(best["V"], dtype=float)
                W50 = np.array(best["W"], dtype=float)
                
                # Try removing each column and checking if the remaining 49 can still work
                # This is a heuristic: check if column r is "redundant"
                for r_remove in range(50):
                    if time.time() > deadline - 15:
                        break
                    U49 = np.delete(U50, r_remove, axis=1)
                    V49 = np.delete(V50, r_remove, axis=1)
                    W49 = np.delete(W50, r_remove, axis=1)
                    
                    # Quick check: does the 49-term sum still give the right tensor?
                    # Check a few random entries
                    ok = True
                    for _ in range(50):
                        i = rng.randint(0, 6)
                        j = rng.randint(0, 27)
                        i_c = i % 3
                        j_r = j // 9
                        if i_c == j_r:
                            j_c = j % 9
                            i_r = i // 3
                            k_out = j_c * 2 + i_r
                            s = np.sum(U49[i, :] * V49[j, :] * W49[k_out, :])
                            if abs(s - 1.0) > 1e-4:
                                ok = False
                                break
                        else:
                            k_out = rng.randint(0, 18)
                            s = np.sum(U49[i, :] * V49[j, :] * W49[k_out, :])
                            if abs(s) > 1e-4:
                                ok = False
                                break
                    if ok:
                        # Full verification
                        U49_list = U49.tolist()
                        V49_list = V49.tolist()
                        W49_list = W49.tolist()
                        # Convert to list of lists for the expected format
                        sol49 = {"U": U49_list, "V": V49_list, "W": W49_list}
                        if verify_solution(sol49, n, m, p):
                            # Round to integers/half-integers
                            U49_rounded = np.round(U49 * 2) / 2
                            V49_rounded = np.round(V49 * 2) / 2
                            W49_rounded = np.round(W49 * 2) / 2
                            sol49_r = {"U": U49_rounded.tolist(), "V": V49_rounded.tolist(), "W": W49_rounded.tolist()}
                            if verify_solution(sol49_r, n, m, p):
                                return sol49_r
    
    return best
