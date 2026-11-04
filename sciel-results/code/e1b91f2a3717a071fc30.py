"""Matrix multiplication 2x3 by 3x6: rank-minimal exact tensor decomposition."""
import time
import itertools

DIMENSIONS = [2, 3, 6]
DEADLINE_SECONDS = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    Returns: {'U': (6, rank), 'V': (18, rank), 'W': (12, rank)}
    where U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] sums to the tensor.
    
    The schoolbook rank is n*m*p = 36.
    We attempt to find a lower-rank decomposition using integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    
    # Tensor T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # In the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid (i,j,k)
    
    # The tensor has n*m*p = 36 non-zero entries (all 1s)
    
    # Strategy: Try to find a decomposition of rank < 36 using structured patterns.
    # For small dimensions, we can try:
    # 1. Known lower bounds and constructions
    # 2. Greedy/ILP approach with small rank candidates
    # 3. Exploit structure: since all tensor entries are 1, look for factorizations
    
    # Lower bound: rank >= max(n*m, m*p, p*n) is not quite right.
    # Actually rank >= max(n, m, p) * min(n, m, p) ... no.
    # The rank of the matrix multiplication tensor is at least max(n*m, m*p, p*n) / something.
    # For 2x3 * 3x6: lower bound is at least max(6, 18, 12) = 18? No, that's not a valid lower bound.
    # 
    # Known results: The rank of matrix multiplication tensor M_{n,m,p} is at least
    # n*m*p / max(n,m,p) ... no.
    # 
    # Actually, a simple lower bound: consider the "slice" where we fix one index.
    # If we fix k, we get a tensor of size n*m with entries that are linear in the rank-1 terms.
    # This gives rank >= n*m for each fixed... no, that's not right either.
    #
    # Let's use the fact that for the standard matrix multiplication, the rank is at least
    # the maximum of the three matrix dimensions times the minimum... 
    # 
    # For 2x3 * 3x6, the schoolbook is 36. Known best for similar:
    # 2x2*2x2: rank 7 (Strassen) vs schoolbook 8
    # 2x3*3x2: rank ? schoolbook 12
    # 2x3*3x6: schoolbook 36
    #
    # Let's try a structured approach: use the fact that all entries are 1.
    # We can try to group terms.
    
    # Approach: Try ranks from a lower bound up to 36.
    # Lower bound: The rank must be at least max(n*m, m*p, p*n) / max(n, m, p)
    # Actually, a better lower bound: rank >= n*m*p / max(n,m,p) is not standard.
    # 
    # Simple lower bound: For any decomposition of rank r,
    # the "fiber" in the (i,j) direction has dimension at most r, but we need n*m = 6 independent fibers...
    # Actually, the rank is at least max(n, m, p) * min(n, m, p) ... 
    #
    # Let's just try from rank 18 (a loose lower bound) up to 36.
    # But searching for arbitrary rank-r decompositions is NP-hard in general.
    #
    # Alternative: Use a known construction or pattern.
    #
    # For 2x3 * 3x6, note that 6 = 2*3, so we can think of the 3x6 matrix as two 3x3 blocks.
    # A(2x3) * B(3x6) = A * [B1 B2] where B1, B2 are 3x3.
    # So the result is [A*B1 A*B2], each 2x3.
    #
    # The tensor for 2x3 * 3x3 has rank at most 18 (schoolbook) or possibly less.
    # If we can find rank r for 2x3*3x3, then 2x3*3x6 might have rank 2r or less.
    #
    # Known: 2x3*3x2 has rank 12 (schoolbook). Is there a lower rank?
    # 
    # Let's try a direct approach: construct a decomposition using a pattern.
    #
    # Idea: Use the fact that we can combine terms.
    # For example, if we have terms (i,j,k) and (i',j',k'), we might be able to
    # represent multiple tensor entries with one rank-1 term if the pattern matches.
    #
    # A rank-1 term contributes U_val * V_val * W_val to T[i,j,k] where
    # i is from U's row index (i*m+j), j is from V's row index (j*p+k), k is from W's row index (k*n+i).
    # 
    # For a rank-1 term to contribute to T[i,j,k] = 1, we need:
    # U[i*m+j] = a, V[j*p+k] = b, W[k*n+i] = c, with a*b*c = 1.
    # But it also contributes to other (i',j',k') where U[i'*m+j'] = a, V[j'*p+k'] = b, W[k'*n+i'] = c.
    # So the contribution to T[i',j',k'] is a*b*c = 1 as well, which is fine (we want 1 there too).
    # But if a*b*c != 1, we have a problem.
    #
    # So a rank-1 term with values (a, b, c) where a*b*c = 1 contributes 1 to ALL (i',j',k')
    # such that U[i'*m+j'] = a, V[j'*p+k'] = b, W[k'*n+i'] = c.
    #
    # This means we need to partition the 36 (i,j,k) triples into groups, where each group
    # corresponds to a rank-1 term, and within each group, the U, V, W values are constant
    # (up to scaling that multiplies to 1).
    #
    # More precisely: A rank-1 term defines a "pattern" - a set of (i,j,k) it covers.
    # The U matrix has a value at row i*m+j for each (i,j) in the pattern.
    # Similarly for V and W.
    #
    # For the term to be valid, for each (i,j,k) in the pattern:
    #   U[i*m+j] * V[j*p+k] * W[k*n+i] = 1
    #
    # And for (i,j,k) NOT in the pattern, the product should be 0 (or the sum over all terms should still be 1).
    #
    # This is getting complex. Let's try a simpler approach:
    # 
    # Use a known result or try small ranks with a heuristic.
    #
    # For now, let's return the schoolbook decomposition (rank 36) as a fallback,
    # but first try to see if we can do better with a structured pattern.
    
    # Try a pattern-based approach:
    # Group the k-index (0..5) into pairs: (0,1), (2,3), (4,5)
    # For each pair (2k, 2k+1), treat them together.
    
    # Actually, let's try a different approach: use the fact that we can have
    # half-integer values to create cancellations.
    
    # Let's try to find a rank-30 decomposition or similar by solving a small system.
    # 
    # Given the complexity, let's use a time-bounded search for a structured pattern.
    
    start_time = time.time()
    
    # First, let's try the schoolbook as default
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using a structured pattern.
    # 
    # Idea: Use the 2x3 * 3x6 structure.
    # Let's try to use a "block" decomposition where we treat the 6 columns as 2 blocks of 3.
    # 
    # For each block b (0 or 1), we compute A * B_b where B_b is 3x3.
    # The tensor for 2x3 * 3x3 has 18 entries.
    # 
    # If we can find a rank-r decomposition for 2x3*3x3, we might combine them.
    #
    # Known: For 2x3*3x2, the rank is 12 (I believe no lower rank is known for this specific case).
    # For 2x3*3x3, schoolbook is 18.
    #
    # Let's try a pattern for 2x3*3x3 first, then extend to 2x3*3x6.
    
    # Actually, let's try a direct approach with a small number of terms.
    # 
    # Pattern idea: Use terms that cover multiple (i,j,k) at once.
    # 
    # For a term to cover (i,j,k) and (i,j,k+1) (adjacent k values), we need:
    # U[i*m+j] = a, V[j*p+k] = b1, V[j*p+k+1] = b2, W[k*n+i] = c1, W[(k+1)*n+i] = c2
    # with a*b1*c1 = 1 and a*b2*c2 = 1.
    #
    # This is possible if we set b1 = b2 and c1 = c2, or other combinations.
    #
    # Let's try a specific pattern:
    # 
    # For each (i, j), we have p=6 k-values.
    # If we use 3 terms per (i,j) pair (covering 2 k-values each), we get 6*3 = 18 terms.
    # But we need to check if the W values are consistent.
    #
    # Term for (i, j, k) and (i, j, k+1):
    #   U[i*m+j] = a
    #   V[j*p+k] = b, V[j*p+k+1] = b
    #   W[k*n+i] = c, W[(k+1)*n+i] = c
    #   with a*b*c = 1
    #
    # But W[k*n+i] is shared across different j values!
    # W[k*n+i] depends only on k and i, not on j.
    # So if two different j values use the same W[k*n+i], they must have consistent values.
    #
    # This is the key constraint. Let's formalize:
    # 
    # For a rank-1 term, it assigns:
    #   U values: for each (i,j) in its support, U[i*m+j] = some value
    #   V values: for each (j,k) in its support, V[j*p+k] = some value  
    #   W values: for each (k,i) in its support, W[k*n+i] = some value
    #
    # The term is valid if for every (i,j,k) in its support:
    #   U[i*m+j] * V[j*p+k] * W[k*n+i] = 1
    #
    # And the union of all terms' supports must cover all 36 (i,j,k) triples exactly once
    # (or with appropriate weights summing to 1).
    #
    # Let's try a partition of the 36 triples into groups where each group is "consistent"
    # (there exist U, V, W values that make the product 1 for all triples in the group).
    #
    # A group is consistent if:
    #   For all (i,j,k) in the group: U_ij * V_jk * W_ki = 1
    #   This means U_ij = 1/(V_jk * W_ki)
    #   So U_ij must be the same for all (i,j,k) in the group with the same (i,j),
    #   which requires V_jk * W_ki to be constant over k for fixed (i,j) in the group.
    #   Similarly, V_jk must be consistent over i for fixed (j,k),
    #   and W_ki must be consistent over j for fixed (k,i).
    #
    # This is a strong constraint. Let's try to find such a partition.
    
    # Try: group by (i, j) and split k into pairs.
    # For each (i,j), group k into {(0,1), (2,3), (4,5)}.
    # For group (i, j, {k, k+1}):
    #   U[i*m+j] = a
    #   V[j*p+k] = b, V[j*p+k+1] = b
    #   W[k*n+i] = c, W[(k+1)*n+i] = c
    #   with a*b*c = 1
    #
    # Constraint: W[k*n+i] must be the same for all j that use it.
    # W[k*n+i] is used by groups (i, j, {k, k+1}) for all j.
    # So for fixed (k, i), all groups (i, j, {k, k+1}) for j=0,1,2 must have the same W value.
    #
    # Similarly, V[j*p+k] is used by groups (i, j, {k, k+1}) for all i.
    # For fixed (j, k), all groups (i, j, {k, k+1}) for i=0,1 must have the same V value.
    #
    # Let's set up the system:
    # For each (i, j, k_pair) where k_pair in {(0,1), (2,3), (4,5)}:
    #   Let a_{i,j,kp} = U[i*m+j] (but U[i*m+j] is shared across k_pairs for the same (i,j)!)
    #
    # Wait, U[i*m+j] is a single value, not per k_pair.
    # So for fixed (i,j), all three k_pairs share the same U value.
    #
    # Let's denote:
    #   u_{i,j} = U[i*m+j]
    #   v_{j,k} = V[j*p+k]
    #   w_{k,i} = W[k*n+i]
    #
    # For the term covering (i, j, k) and (i, j, k+1):
    #   u_{i,j} * v_{j,k} * w_{k,i} = 1
    #   u_{i,j} * v_{j,k+1} * w_{k+1,i} = 1
    #
    # From these: v_{j,k} * w_{k,i} = v_{j,k+1} * w_{k+1,i} = 1/u_{i,j}
    #
    # So for fixed (i,j), the product v_{j,k} * w_{k,i} is the same for k and k+1.
    #
    # This gives us: v_{j,k} * w_{k,i} = v_{j,k+1} * w_{k+1,i}
    # => v_{j,k} / v_{j,k+1} = w_{k+1,i} / w_{k,i}
    #
    # Let's try to assign values. Set w_{k,i} = 1 for all k,i.
    # Then v_{j,k} * 1 = v_{j,k+1} * 1 => v_{j,k} = v_{j,k+1}
    # So v_{j,k} = v_{j,k+1} for all k. Thus v_{j,k} = v_j (independent of k).
    # Then u_{i,j} * v_j * 1 = 1 => u_{i,j} = 1/v_j.
    #
    # So with w_{k,i} = 1, v_{j,k} = v_j, u_{i,j} = 1/v_j:
    #   The term covers (i, j, k) and (i, j, k+1) with value 1.
    #
    # But wait, this means the term covers ALL k values for a given (i,j), not just a pair.
    # Because v_{j,k} = v_j for all k, and w_{k,i} = 1 for all k.
    # So u_{i,j} * v_j * 1 = 1 for all k.
    #
    # This means one term per (i,j) covers all 6 k-values!
    # That gives rank = n*m = 6.
    #
    # Let's verify:
    #   U[i*m+j] = 1/v_j
    #   V[j*p+k] = v_j for all k
    #   W[k*n+i] = 1 for all k,i
    #
    # For (i,j,k): U[i*m+j] * V[j*p+k] * W[k*n+i] = (1/v_j) * v_j * 1 = 1. ✓
    #
    # But does this work? The issue is that V[j*p+k] = v_j for all k means
    # V[j*p+k] depends only on j, not on k.
    # And W[k*n+i] = 1 for all k,i.
    #
    # Let's check if this is a valid decomposition:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # With rank 6, one term per (i,j):
    #   Term (i,j): U[i*m+j] = 1/v_j, V[j*p+k] = v_j, W[k*n+i] = 1
    #   Contribution to (i,j,k): (1/v_j) * v_j * 1 = 1 ✓
    #
    # But wait, V[j*p+k] = v_j for ALL k. So for a fixed j, V[j*p+0] = V[j*p+1] = ... = V[j*p+5] = v_j.
    # And W[k*n+i] = 1 for all k,i.
    #
    # This seems to work! Rank 6!
    #
    # But is this correct? Let me double-check the tensor definition.
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,2)x[0,3)x[0,6).
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    #
    # With our construction:
    #   For term c = (i', j'):
    #     U[i*m+j, c] = 1/v_{j'} if (i,j) = (i',j'), else 0
    #     V[j*p+k, c] = v_{j'} for all k
    #     W[k*n+i, c] = 1 for all k,i
    #
    # So the sum over c:
    #   sum_{c=(i',j')} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #   = sum_{c=(i',j')} [1/v_{j'} if (i,j)=(i',j') else 0] * v_{j'} * 1
    #   = [1/v_j * v_j * 1] if (i,j) is in the sum, else 0
    #   = 1
    #
    # Yes! This works for all (i,j,k). Rank = 6.
    #
    # But wait, this seems too good. Let me check if there's a flaw.
    #
    # The issue: V[j*p+k, c] = v_{j'} for all k. But V has rows indexed by (j,k) pairs.
    # For term c = (i', j'), V[j*p+k, c] = v_{j'} for ALL (j,k), not just j=j'.
    #
    # So V[j*p+k, c] is non-zero for all j, not just j=j'.
    #
    # Let me recompute:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #   = sum_{(i',j')} U[i*m+j, (i',j')] * V[j*p+k, (i',j')] * W[k*n+i, (i',j')]
    #
    # U[i*m+j, (i',j')] = 1/v_{j'} if (i,j) = (i',j'), else 0
    # V[j*p+k, (i',j')] = v_{j'} for all (j,k)
    # W[k*n+i, (i',j')] = 1 for all (k,i)
    #
    # So the sum is:
    #   sum_{(i',j')} [1/v_{j'} * delta_{(i,j),(i',j')}] * v_{j'} * 1
    #   = 1/v_j * v_j * 1 = 1
    #
    # The delta ensures only the term with (i',j') = (i,j) contributes.
    # So yes, it works!
    #
    # But this means V[j*p+k, c] is non-zero for all j, which is fine.
    # The key insight is that U has the "selective" pattern, while V and W are "broad".
    #
    # Let's construct this:
    
    def try_rank_6():
        rank = n * m  # 6
        # Map (i,j) to column index
        col_map = {}
        col = 0
        for i in range(n):
            for j in range(m):
                col_map[(i, j)] = col
                col += 1
        
        # Choose v_j values. Let's use v_j = 1 for all j for simplicity.
        v_vals = [1] * m  # v_0, v_1, v_2 = 1, 1, 1
        
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for i in range(n):
            for j in range(m):
                c = col_map[(i, j)]
                U[i * m + j][c] = 1.0 / v_vals[j]  # = 1
                for k in range(p):
                    V[j * p + k][c] = v_vals[j]  # = 1
                for k in range(p):
                    W[k * n + i][c] = 1
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return None
        return {"U": U, "V": V, "W": W}
    
    result = try_rank_6()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
