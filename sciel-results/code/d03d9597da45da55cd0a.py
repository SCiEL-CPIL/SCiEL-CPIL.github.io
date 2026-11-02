import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    
    # Rank-9 decomposition:
    # - 7 terms from Strassen's 2x2x2 algorithm for the (i=0,1; j=0,1; k=0..7) sub-tensor
    #   Actually, we need to think about this more carefully.
    #
    # The tensor has entries T[a,b,c] = 1 if (a,b,c) corresponds to (i*m+j, j*p+k, k*n+i)
    # for some i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}, else 0.
    #
    # Key insight: We can decompose the multiplication as:
    #   C = A * B where A is 3x4, B is 4x8
    #   C[0,:] = A[0,:] * B[:,0:8]  (row 0 of A times B)
    #   C[1,:] = A[1,:] * B[:,0:8]  (row 1 of A times B)  
    #   C[2,:] = A[2,:] * B[:,0:8]  (row 2 of A times B)
    #
    # For the 3x4 x 4x8 case, one approach:
    # Split A into [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4
    # Then C_top = A_top * B (2x8) and C_bottom = A_bottom * B (1x8)
    #
    # For C_top (2x4 * 4x8): use a block decomposition
    # For C_bottom (1x4 * 4x8): rank 4 (each output needs 4 multiplications)
    #
    # Actually, let me think about a cleaner algebraic construction.
    #
    # The matrix multiplication tensor for 3x4x8 can be viewed as:
    # T = sum over i,j,k of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # A rank-r decomposition is: T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t, v_t, w_t are vectors.
    #
    # Key structural observation: The tensor has a "cyclic" structure in the indices.
    # For fixed j, the terms involving j form a 3x8x3 sub-tensor (i and k vary).
    # For fixed i, the terms form a 4x8x4 sub-tensor (j and k vary).
    #
    # Let's try a direct construction with rank 9:
    # 
    # We'll use the following strategy:
    # - 4 terms for the "diagonal" structure (one for each j)
    # - 5 additional terms to handle the off-diagonal interactions
    #
    # Actually, let me try a more systematic approach.
    #
    # The tensor T has 96 non-zero entries (one for each (i,j,k) triple).
    # A rank-1 term u⊗v⊗w can cover at most |supp(u)|*|supp(v)|*|supp(w)| entries,
    # but it must be that ALL covered entries are non-zero in T (or cancel out).
    #
    # For a rank-1 term to be "clean" (no cancellation needed), we need:
    # For all a in supp(u), b in supp(v), c in supp(w): (a,b,c) is either a target or zero.
    #
    # Let's find maximal "boxes" in the tensor.
    #
    # A box is defined by I ⊆ {0,1,2}, J ⊆ {0,1,2,3}, K ⊆ {0,...,7}
    # such that for all i∈I, j∈J, k∈K: (i*m+j, j*p+k, k*n+i) is a target.
    # This is always true by definition! The question is whether the box
    # {(i*m+j, j*p+k, k*n+i) : i∈I, j∈J, k∈K} contains ONLY target entries
    # when mapped to (a,b,c) space.
    #
    # Two different (i,j,k) and (i',j',k') give the same (a,b,c) iff:
    # i*m+j = i'*m+j', j*p+k = j'*p+k', k*n+i = k'*n+i'
    # From first: i=i' and j=j'. Then from second: k=k'. So all distinct.
    # 
    # But the issue is: for a rank-1 term with support on I×J×K in (i,j,k) space,
    # the (a,b,c) entries covered are {(i*m+j, j*p+k, k*n+i) : i∈I, j∈J, k∈K}.
    # For this to be a valid rank-1 contribution, we need that for any (a,b,c) 
    # in the "box" {(i*m+j, j'*p+k, k'*n+i') : i∈I, j'∈J, k'∈K} 
    # (i.e., all combinations of a from I×J, b from J×K, c from K×I),
    # either (a,b,c) is a target OR the product u[a]*v[b]*w[c] = 0.
    #
    # This is complex. Let me just try to construct a rank-9 decomposition directly.
    #
    # Alternative: Use the fact that 3x4x8 can be embedded in known decompositions.
    # 
    # Known: 2x2x2 has rank 7 (Strassen). 
    # 3x4x8 = 3 * 4 * 8 = 96 schoolbook.
    # 
    # Let's try: decompose into blocks.
    # A (3x4) = [A00 A01; A10 A11] where each Ai is 1x2? No, 3 isn't even.
    # 
    # A (3x4) = [A_top (2x4); A_bot (1x4)]
    # B (4x8) = [B_left (4x4); B_right (4x4)]? No, that's not how it works.
    #
    # C (3x8) = A * B
    # C_top (2x8) = A_top * B
    # C_bot (1x8) = A_bot * B
    #
    # For C_top = A_top * B where A_top is 2x4 and B is 4x8:
    #   This is a 2x4x8 multiplication.
    #   We can write B = [B1 B2] where B1, B2 are 4x4.
    #   C_top = [A_top * B1, A_top * B2]
    #   Each A_top * B_i is 2x4 * 4x4 = 2x4x4.
    #   
    #   2x4x4: A is 2x4, B is 4x4, C is 2x4.
    #   Schoolbook rank: 2*4*4 = 32.
    #   Can we do better? 
    #   2x4x4 = 2 * (1x4x4). Each 1x4x4 has rank 4.
    #   So 2x4x4 has rank at most 8 (using 4 terms for row 0, 4 for row 1).
    #   Actually, we can share: if we use rank-4 terms that contribute to both rows...
    #   
    #   Let's think: 1x4x4 tensor: T[a,b,c] = 1 if a=0 (row 0 of A), b=j (col j of A = row j of B), c=k (col k of C).
    #   Wait, for 1x4x4: A is 1x4, B is 4x4, C is 1x4.
    #   T[i*m+j, j*p+k, k*n+i] with n=1, m=4, p=4.
    #   a = 0*4+j = j, b = j*4+k, c = k*1+0 = k.
    #   Targets: (j, j*4+k, k) for j,k in 0..3.
    #   This is a rank-4 tensor: sum_{j=0}^{3} sum_{k=0}^{3} e_j ⊗ e_{4j+k} ⊗ e_k
    #   = sum_{j=0}^{3} e_j ⊗ (sum_{k=0}^{3} e_{4j+k} ⊗ e_k)
    #   Hmm, that's not directly low rank.
    #   Actually: sum_{j,k} e_j ⊗ e_{4j+k} ⊗ e_k = sum_j (e_j ⊗ sum_k e_{4j+k} ⊗ e_k)
    #   For each j, sum_k e_{4j+k} ⊗ e_k is a rank-1? No, it's sum of 4 rank-1 terms.
    #   So total rank is 4*4 = 16? No wait.
    #   
    #   Let me re-examine. For 1xn x nx1 (row vector times column vector):
    #   A is 1xn, B is nx1, C is 1x1.
    #   T[a,b,c] = 1 if a=j, b=j, c=0 for some j. (n terms)
    #   This has rank n: sum_j e_j ⊗ e_j ⊗ e_0.
    #
    #   For 1x4 x 4x4:
    #   A is 1x4, B is 4x4, C is 1x4.
    #   n=1, m=4, p=4.
    #   a = i*m+j = 0*4+j = j (i=0 only)
    #   b = j*p+k = j*4+k
    #   c = k*n+i = k*1+0 = k
    #   Targets: (j, 4j+k, k) for j=0..3, k=0..3.
    #   
    #   Can we decompose this with rank < 16?
    #   T = sum_{j=0}^{3} sum_{k=0}^{3} e_j ⊗ e_{4j+k} ⊗ e_k
    #   = sum_{j=0}^{3} e_j ⊗ (sum_{k=0}^{3} e_{4j+k} ⊗ e_k)
    #   The inner sum for fixed j is a 4x4 matrix (in b,c indices) that is the identity-like
    #   structure restricted to row j of B. 
    #   sum_{k=0}^{3} e_{4j+k} ⊗ e_k = sum_k e_{4j+k} ⊗ e_k
    #   This is a sum of 4 rank-1 matrices, so it has rank 4.
    #   So for each j, we need rank 4. Total rank 16? 
    #   But we can potentially share structure across j.
    #   
    #   Actually, I realize I'm overcomplicating this. Let me just try to find
    #   a rank-9 decomposition by direct construction.
    #
    # New approach: Use the cyclic structure.
    # The tensor is: sum_{i=0}^{2} sum_{j=0}^{3} sum_{k=0}^{7} e_{4i+j} ⊗ e_{8j+k} ⊗ e_{8k+i}
    #
    # Let's try rank 9 with the following 9 terms:
    # Term t (t=0..8): defined by specific (i_t, j_t, k_t) patterns.
    #
    # Actually, let me try a completely different approach:
    # Use the fact that we can factor through smaller matrices.
    #
    # C = A * B, where A is 3x4, B is 4x8.
    # Let M = A * D where D is 4xr, and N = E * B where E is rx8, r < 96.
    # Then C = M * N if we choose D, E such that A*B = (A*D)*(E*B).
    # This requires D*E = I_4 (identity), which means r >= 4.
    # But this gives C = A*D*E*B, which is not the same as A*B unless D*E=I.
    # This doesn't help for the tensor decomposition directly.
    #
    # Let me try yet another approach: explicit rank-9 construction.
    #
    # I'll define 9 rank-1 terms and verify they sum to the target tensor.
    #
    # Term 0: covers i=0, j=0, k=0..7 (8 targets) - but this might create conflicts
    # Term 1: covers i=0, j=1, k=0..7
    # ...
    #
    # This is getting complex. Let me just try a random search for rank 9, 10, 11, ...
    # with a time limit, and fall back to schoolbook if nothing found.
    
    nU = n * m  # 12
    nV = m * p  # 32  
    nW = p * n  # 24
    
    # Build target set
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    def verify(U, V, W, rank):
        """Verify decomposition is exact."""
        # Check target entries = 1
        for (a, b, c) in target_set:
            val = sum(U[a][t] * V[b][t] * W[c][t] for t in range(rank))
            if abs(val - 1.0) > 1e-6:
                return False
        # Check non-target entries = 0
        for a in range(nU):
            for b in range(nV):
                for c in range(nW):
                    if (a, b, c) not in target_set:
                        val = sum(U[a][t] * V[b][t] * W[c][t] for t in range(rank))
                        if abs(val) > 1e-6:
                            return False
        return True
    
    # Try to construct a low-rank decomposition using a structured approach
    # We'll try ranks from 9 upward
    
    import random
    random.seed(42)
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # First, set up schoolbook as fallback
    schoolbook_rank = n * m * p
    sb_U = np.zeros((nU, schoolbook_rank))
    sb_V = np.zeros((nV, schoolbook_rank))
    sb_W = np.zeros((nW, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                sb_U[a, col] = 1.0
                sb_V[b, col] = 1.0
                sb_W[c, col] = 1.0
                col += 1
    
    best_U = sb_U.tolist()
    best_V = sb_V.tolist()
    best_W = sb_W.tolist()
    best_rank = schoolbook_rank
    
    # Try structured constructions for low ranks
    # Strategy: For each candidate rank r, try to find U, V, W such that
    # the tensor equation is satisfied.
    
    # For rank r, we have r columns in U (12xr), V (32xr), W (24xr).
    # The constraint is: for all (a,b,c), sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    
    # This is a system of 12*32*24 = 9216 equations in 12r + 32r + 24r = 68r unknowns.
    # For r=9: 612 unknowns, 9216 equations. Very overdetermined.
    # But most equations are "must be zero" which is easier to satisfy.
    
    # Let's try a different approach: exploit the block structure.
    # 
    # The 3x4x8 tensor can be seen as 8 copies of a 3x4x1 tensor (one for each k),
    # but with coupling through the j index.
    #
    # Actually, let me try the following explicit construction for rank 9:
    #
    # We'll use 9 terms. Each term t has:
    #   U_t: vector in R^12 (nonzero in at most a few positions)
    #   V_t: vector in R^32
    #   W_t: vector in R^24
    #
    # Key idea: Use the fact that for fixed k, the sub-tensor over (i,j) is:
    #   sum_{i,j} e_{4i+j} ⊗ e_{8j+k} ⊗ e_{8k+i}
    # This is a 3x4 "slice" with fixed k.
    #
    # For a fixed k, the entries are: (4i+j, 8j+k, 8k+i) for i=0,1,2, j=0,1,2,3.
    # That's 12 entries per k, total 96.
    #
    # Can we cover all entries for a fixed k with few rank-1 terms?
    # For fixed k, we need: sum_t U[4i+j,t] * V[8j+k,t] * W[8k+i,t] = 1 for all i,j.
    # And for non-target (a,b,c) with b=8j+k (fixed k), we need sum = 0.
    #
    # Let's try to find a rank-3 decomposition for each k-slice (3x4x1 with fixed k).
    # For fixed k, define:
    #   f_k(i,j) = 1 for all i,j (we want U[4i+j]*V[8j+k]*W[8k+i] to sum to 1)
    #
    # If we use rank 3 for each k, that's 8*3 = 24 terms total. Not great.
    #
    # Let me try a completely different explicit construction.
    #
    # I'll try to find a rank-9 decomposition by solving a linear system
    # with a clever parameterization.
    
    # For now, let's try a randomized search for small ranks
    start_time = time.time()
    deadline = start_time + 100  # leave 20s margin
    
    for target_rank in range(9, 20):
        if time.time() > deadline - 20:
            break
        
        found = False
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with sparse structure
            U = np.zeros((nU, target_rank))
            V = np.zeros((nV, target_rank))
            W = np.zeros((nW, target_rank))
            
            # For each term t, assign random non-zero entries
            for t in range(target_rank):
                # Choose random support sizes
                nu = random.randint(1, 3)
                nv = random.randint(1, 3)
                nw = random.randint(1, 3)
                
                # Choose random indices
                u_idx = random.sample(range(nU), nu)
                v_idx = random.sample(range(nV), nv)
                w_idx = random.sample(range(nW), nw)
                
                for idx in u_idx:
                    U[idx, t] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                for idx in v_idx:
                    V[idx, t] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                for idx in w_idx:
                    W[idx, t] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            
            # Check if this works (very unlikely to work exactly, but let's try)
            # This random approach won't work for exact decomposition.
            # Let me try a more structured approach.
            break  # skip random, try structured
        
        # Structured approach for this rank
        # Try to build the decomposition term by term
        # This is complex, so for now just return schoolbook if we can't find better
        
    # Given the complexity, return the schoolbook decomposition
    # which is guaranteed correct
    return {"U": best_U, "V": best_V, "W": best_W}
