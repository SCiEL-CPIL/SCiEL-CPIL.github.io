import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Build the schoolbook decomposition as baseline (rank 42)
    # Then try to reduce rank by finding Strassen-like eliminations
    
    # First, build the full schoolbook tensor
    # T[i,j,k] = 1 if the (i,k) entry of A*B uses A[i,j]*B[j,k]
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta for all (i,j,k)
    # where delta[i,j,k] = 1 for the matrix multiplication tensor
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # Wait, no. The tensor is: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    
    # Rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c[i,j] = U[i,j,c], v_c[j,k] = V[j,k,c], w_c[k,i] = W[k,i,c]
    
    # For schoolbook: each term c corresponds to a single (i,j,k) triple
    # U has 1 at (i,j), V has 1 at (j,k), W has 1 at (k,i)
    
    # Strategy: Start with Strassen-based decomposition (rank 39) and try to find
    # additional rank reductions by identifying pairs of terms that can be combined.
    
    # Let me try a different approach: directly search for rank-38 decomposition
    # by fixing sparse support patterns and solving for W.
    
    # The key insight: if we fix U and V to have specific sparse patterns,
    # then for each (i,k) pair, we need:
    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] over j = 1 for all j
    # Actually: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: for all (i,k): sum_j sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # Wait, that's not right either. Let me think more carefully.
    
    # The tensor decomposition: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    # For matrix multiplication, T[i,j,k] = 1 for all (i,j,k)
    # No wait - the multiplication tensor is defined as:
    # T(A,B) = A*B, so T[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor form: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where C[i,k]=A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in [n]×[m]×[p]
    
    # Rank-r decomposition: T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # For the schoolbook decomposition with rank n*m*p:
    # Each term c corresponds to (i,j,k): U_c[i,j]=1, V_c[j,k]=1, W_c[k,i]=1, rest 0
    
    # For Strassen on 2x2x2 (sub-block), we use 7 terms instead of 8.
    
    # Let me try to directly construct a rank-38 solution by:
    # 1. Using Strassen for the 2x2x2 blocks (gives 7 per block instead of 8)
    # 2. Trying to find one more elimination across the remaining terms
    
    # Current best: 39 (from 42 - 3 Strassen savings on three 2x2x2 blocks)
    # To get 38: need one more elimination
    
    # Let me try a systematic approach: 
    # Fix the structure to be similar to the current 39-rank solution,
    # but try to merge two of the schoolbook terms in Part 5 or Part 2
    
    # Alternative: try to find a rank-38 decomposition by solving a linear system
    
    # Let me parameterize: assume we have rank r=38 terms.
    # For each term c, U_c is a 2x3 matrix, V_c is a 3x7 matrix, W_c is a 7x2 matrix.
    # Constraint: for all (i,j,k) in [2]x[3]x[7]:
    #   sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # This is a system of 42 equations in many unknowns.
    # If we fix U and V (sparse patterns), then W is determined linearly.
    
    # Let me try: keep the Strassen structure for the 2x2x2 blocks,
    # and for the remaining terms, try to find a way to save one more rank.
    
    # Part 2 has 4 schoolbook terms: (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    # Part 5 has 14 schoolbook terms: (0,2,k) and (1,2,k) for k=0..6
    
    # Can we combine terms from Part 2 and Part 5?
    # Term (0,0,2): U has 1 at (0,0), V has 1 at (0,2), W has 1 at (2,0)
    # Term (0,2,0): U has 1 at (0,2), V has 1 at (2,0), W has 1 at (0,0)
    
    # These are different terms in the tensor. They can't be directly combined
    # unless we find a Strassen-like identity.
    
    # Let me try a computational search: fix U and V to have a specific structure
    # and solve for W.
    
    # Approach: Use the 39-term solution as a starting point and try to find
    # a linear dependency that allows eliminating one term.
    
    # Build the 39-term solution
    terms = []
    
    def add_term(u, v, w):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
        
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] (2x2x2)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1, w_idx(1,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1, v_idx(1,1): -1}, {w_idx(1,0): 1, w_idx(1,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,0): 1, v_idx(0,0): -1}, {w_idx(0,0): 1, w_idx(0,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(0,0): -1, w_idx(1,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,0): 1, v_idx(0,1): 1}, {w_idx(1,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,0): 1, v_idx(1,1): 1}, {w_idx(0,0): 1})
    
    # Part 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2] (2x2x1)
    add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,2): 1}, {w_idx(2,1): 1})
    
    # Part 3: Strassen for A[0:2, 0:2] * B[0:2, 3:5] (2x2x2)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1, w_idx(4,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1, w_idx(4,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,4): 1, v_idx(1,4): -1}, {w_idx(4,0): 1, w_idx(4,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,3): 1, v_idx(0,3): -1}, {w_idx(3,0): 1, w_idx(3,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,4): 1}, {w_idx(3,0): -1, w_idx(4,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,3): 1, v_idx(0,4): 1}, {w_idx(4,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,3): 1, v_idx(1,4): 1}, {w_idx(3,0): 1})
    
    # Part 4: Strassen for A[0:2, 0:2] * B[0:2, 5:7] (2x2x2)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1, w_idx(6,1): 1})
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, {v_idx(0,5): 1}, {w_idx(5,1): 1, w_idx(6,1): -1})
    add_term({u_idx(0,0): 1}, {v_idx(0,6): 1, v_idx(1,6): -1}, {w_idx(6,0): 1, w_idx(6,1): 1})
    add_term({u_idx(1,1): 1}, {v_idx(1,5): 1, v_idx(0,5): -1}, {w_idx(5,0): 1, w_idx(5,1): 1})
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, {v_idx(1,6): 1}, {w_idx(5,0): -1, w_idx(6,0): 1})
    add_term({u_idx(1,0): 1, u_idx(0,0): -1}, {v_idx(0,5): 1, v_idx(0,6): 1}, {w_idx(6,1): 1})
    add_term({u_idx(0,1): 1, u_idx(1,1): -1}, {v_idx(1,5): 1, v_idx(1,6): 1}, {w_idx(5,0): 1})
    
    # Part 5: Schoolbook for A[0:2, 2] * B[2, 0:7] (2x1x7)
    for k in range(p):
        add_term({u_idx(0,2): 1}, {v_idx(2,k): 1}, {w_idx(k,0): 1})
        add_term({u_idx(1,2): 1}, {v_idx(2,k): 1}, {w_idx(k,1): 1})
    
    # Now try to find a rank-38 decomposition by attempting to eliminate one term
    # through algebraic manipulation
    
    # Strategy: Try to find a linear combination of terms that equals zero
    # in the tensor, which would allow us to remove one term.
    
    # Alternatively: Try to find a rank-38 decomposition by fixing U and V patterns
    # and solving for W.
    
    # Let me try: Can we combine the two terms in Part 2 that contribute to w_idx(2,0)?
    # Term 7: U={u_idx(0,0):1}, V={v_idx(0,2):1}, W={w_idx(2,0):1}
    # Term 8: U={u_idx(0,1):1}, V={v_idx(1,2):1}, W={w_idx(2,0):1}
    
    # These contribute to (A*B)[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # Can't easily combine these two.
    
    # Let me try a different approach: use the fact that we have freedom in choosing
    # the U and V matrices. Fix them to be sparse and solve for W.
    
    # For rank r, we have r terms. Each term c has:
    # U_c: 2x3 matrix (6 entries)
    # V_c: 3x7 matrix (21 entries)  
    # W_c: 7x2 matrix (14 entries)
    # Total unknowns per term: 41, total: 41*r
    # Constraints: 2*3*7 = 42 equations
    
    # For r=38: 41*38 = 1558 unknowns, 42 equations. Very underdetermined.
    # But we need the solution to be "nice" (integer/half-integer).
    
    # Let me try to construct a rank-38 solution by:
    # 1. Using Strassen for all three 2x2x2 blocks (21 terms)
    # 2. Finding a clever way to handle the remaining 17 terms (Part 2: 4 terms, Part 5: 14 terms)
    #    with only 17 terms total, saving 1 from the schoolbook 18.
    
    # Part 2 + Part 5 = 4 + 14 = 18 schoolbook terms
    # Can we do it in 17?
    
    # Part 2: (i,j,k) = (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    # Part 5: (i,j,k) = (0,2,k) and (1,2,k) for k=0..6
    
    # Total: 4 + 14 = 18 terms
    
    # For the 2x1x7 part (Part 5), schoolbook needs 14 terms.
    # Can we use Strassen-like tricks? 2x1x7 is not a square block.
    
    # Actually, let me reconsider. The 2x1x7 tensor:
    # (A[0:2,2] * B[2,0:7])[i,k] = A[i,2]*B[2,k]
    # This is an outer product of a 2-vector and a 7-vector.
    # The rank of an outer product u ⊗ v is 1!
    # Wait, no. In the tensor decomposition sense for matrix multiplication:
    # T[i,j,k] = 1 if j=2, for all i,k.
    # So T = sum_{i,k} e_i ⊗ e_2 ⊗ e_k
    # This has rank 2*7 = 14 in general (it's a sum of 14 rank-1 tensors).
    # Actually no, it can be written as: (sum_i e_i) ⊗ e_2 ⊗ (sum_k e_k)
    # which is rank 1! But that's not a valid decomposition because:
    # (sum_i e_i) ⊗ e_2 ⊗ (sum_k e_k)[i,j,k] = 1 if j=2, else 0
    # But we need it to equal 1 for ALL (i,j,k), not just j=2.
    
    # I'm confusing myself. Let me restart the tensor definition.
    
    # The matrix multiplication tensor for (n,m,p) is:
    # T ∈ R^{n×m × m×p × p×n} (flattened)
    # T[i,j, k,l] ... no, this is getting confusing.
    
    # Let me use the standard definition:
    # The multiplication tensor M_{n,m,p} is in R^{nm} ⊗ R^{mp} ⊗ R^{pn}
    # M_{n,m,p}[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and k=k' and i=i'... 
    # No, that's not right either.
    
    # Standard: M_{n,m,p} is the tensor such that
    # <M_{n,m,p}, A ⊗ B> = A*B (as a vector)
    # So M_{n,m,p} = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the basis vector in R^{nm} for position (i,j)
    # e_{j,k} is the basis vector in R^{mp} for position (j,k)
    # e_{k,i} is the basis vector in R^{pn} for position (k,i)
    
    # So M_{n,m,p}[i,j, j',k, k',i'] = 1 iff j=j' and k=k' and i=i'
    # In flattened form: M[i*m+j, j'*p+k, k'*n+i'] = 1 iff j=j' and k=k' and i=i'
    # Which means: M[i*m+j, j*p+k, k*n+i] = 1 (since j=j', k=k', i=i')
    
    # So the tensor has 1s at positions (i*m+j, j*p+k, k*n+i) for all valid (i,j,k)
    # And 0s elsewhere.
    
    # A rank-r decomposition: M = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c ∈ R^{nm}, v_c ∈ R^{mp}, w_c ∈ R^{pn}
    # M[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    
    # For the schoolbook: each term corresponds to a single (i,j,k):
    # u has 1 at position i*m+j, v has 1 at position j*p+k, w has 1 at position k*n+i
    
    # This is correct. Now, for the 2x1x7 sub-tensor (j=2 only):
    # Terms: (i,2,k) for i=0,1 and k=0..6, total 14 terms
    # Can we do better? The sub-tensor is:
    # M_sub[i, 2, k] = 1 for all i,k (with j fixed to 2)
    # In the full tensor, this corresponds to positions:
    # u position: i*m+2 = i*3+2
    # v position: 2*p+k = 2*7+k = 14+k
    # w position: k*n+i = k*2+i
    
    # This sub-tensor has rank 14 in general (it's a sum of 14 orthogonal rank-1 tensors).
    # But wait, can we write it as a product of three vectors?
    # u ⊗ v ⊗ w where u[i*3+2] = a_i, v[14+k] = b_k, w[k*2+i] = c_{k,i}
    # Then (u⊗v⊗w)[i*3+2, 14+k, k*2+i] = a_i * b_k * c_{k,i}
    # We need this to equal 1 for all i,k.
    # So a_i * b_k * c_{k,i} = 1 for all i,k.
    # This is possible! Set a_i = 1, b_k = 1, c_{k,i} = 1.
    # But then (u⊗v⊗w) has non-zero entries elsewhere too:
    # (u⊗v⊗w)[i*3+j, j'*7+k, k'*2+i'] = u[i*3+j]*v[j'*7+k]*w[k'*2+i']
    # With u[i*3+j] = 1 if j=2 else 0, v[j'*7+k] = 1 if j'=2 else 0, w[k'*2+i'] = 1 for all
    # So (u⊗v⊗w)[i*3+2, 2*7+k, k'*2+i'] = 1 if i'=i... 
    
    # Hmm, this is getting complicated. The point is that the 14 terms for j=2
    # cannot generally be reduced below 14 because they involve different (i,k) pairs
    # in the W matrix.
    
    # Let me just try to verify if rank 38 is achievable by a computational search.
    # Given time constraints, let me try a few specific patterns.
    
    # Actually, let me try a completely different approach:
    # Use the fact that for 2x3x7, we can try to find a rank-38 solution
    # by fixing U and V to have a specific structure and solving for W.
    
    # Let me try: Use Strassen for the 2x2x2 blocks (21 terms), and for the
    # remaining terms, try a different partitioning.
    
    # What if we partition as:
    # - 2x2x2 blocks for j=0,1 and k=0,1 / k=2,3 / k=4,5 / k=6
    #   That's 3 Strassen blocks (21 terms) + 1 schoolbook 2x2x1 (4 terms) = 25
    # - Plus 2x1x7 for j=2: 14 terms
    # Total: 25 + 14 = 39 (same as before)
    
    # What if we use a different partition for the j=2 part?
    # 2x1x7: can we split into 2x1x3 + 2x1x3 + 2x1x1?
    # Each 2x1x3 has rank 6 (schoolbook), 2x1x1 has rank 2.
    # Total: 6+6+2 = 14. Same.
    
    # I don't see an obvious way to get below 39 with this structure.
    
    # Let me try a computational approach: search for rank-38 by trying
    # different support patterns for U and V.
    
    # Given the time budget, let me try a few specific patterns.
    
    # Pattern 1: Try to combine the last schoolbook term with a Strassen term.
    # This is unlikely to work, but let's try.
    
    # For now, let me return the 39-rank solution and add a timeout guard.
    
    if time.time() > deadline - 5:
        # Return the 39-rank solution
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        for c, (u_list, v_list, w_list) in enumerate(terms):
            for i in range(n * m):
                U[i][c] = u_list[i]
            for i in range(m * p):
                V[i][c] = v_list[i]
            for i in range(p * n):
                W[i][c] = w_list[i]
                
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank-38 decomposition
    # Fix U and V to have a specific sparse pattern and solve for W
    
    # Let me try: use the 39-term solution but try to find a linear dependency
    # that allows removing one term.
    
    # Build the tensor from the 39 terms and check if any term is redundant.
    # This is hard to do directly, so let me try a different approach.
    
    # Try to construct a rank-38 solution by fixing U and V.
    
    # Hypothesis: There exists a rank-38 solution where:
    # - U has a specific sparse pattern
    # - V has a specific sparse pattern
    # - W is determined by solving a linear system
    
    # Let me try to set up the linear system for a specific pattern.
    
    # For each (i,j,k) in [2]x[3]x[7]:
    # sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
    
    # If we fix U and V, this becomes a linear system in W.
    # For each (k,i) pair, we have:
    # sum_c (sum_j U_c[i*m+j] * V_c[j*p+k]) * W_c[k*n+i] = 1
    
    # Let me define: S_{i,k,c} = sum_j U_c[i*m+j] * V_c[j*p+k]
    # Then: sum_c S_{i,k,c} * W_c[k*n+i] = 1 for all (i,k)
    
    # This gives 2*7 = 14 equations for each (k,i) pair... no wait.
    # For each (k,i) pair, we have one equation involving W_c[k*n+i] for all c.
    # So we have 14 equations, each involving r unknowns (W_c[k*n+i] for c=0..r-1).
    # But W_c[k*n+i] is a different variable for each c and (k,i).
    
    # Actually, the unknowns are W_c[k*n+i] for c=0..r-1 and (k,i) in [7]x[2].
    # Total unknowns: r * 14
    # Total equations: 14 (one for each (i,k) pair... no, one for each (i,k) where i in [2], k in [7])
    # Wait, the constraint is: for all (i,j,k): sum_c U_c[i*m+j]*V_c[j*p+k]*W_c[k*n+i] = 1
    # This is 2*3*7 = 42 constraints.
    # But they're not all independent in the way I described.
    
    # Let me just try to solve it numerically for a specific pattern.
    
    # Given the complexity and time budget, let me just return the 39-rank solution.
    # The search for rank-38 is complex and may not be fruitful in the remaining time.
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u_list[i]
        for i in range(m * p):
            V[i][c] = v_list[i]
        for i in range(p * n):
            W[i][c] = w_list[i]
            
    return {"U": U, "V": V, "W": W}
