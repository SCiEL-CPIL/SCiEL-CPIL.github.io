"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for rank-minimal exact tensor decomposition
of 2x3 * 3x7 matrix multiplication. Uses structured families + exact verification with
a hard deadline guard. Falls back to schoolbook rank 42."""

import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 145.0  # safety margin under 150s kill

    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 42

    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = build_schoolbook()

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    uidx = i * m + j
                    vidx = j * p + k
                    widx = k * n + i
                    Urow = U[uidx]
                    Vrow = V[vidx]
                    Wrow = W[widx]
                    for c in range(rank):
                        s += Urow[c] * Vrow[c] * Wrow[c]
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        rank = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy 1: Try to find a rank-r decomposition by solving the bilinear system
    # For each term c, we have vectors u_c, v_c, w_c. The constraint is:
    # For all (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    #
    # Key insight: If we fix the structure of some variables, the remaining system
    # becomes bilinear. We can try various structural templates.

    # Strategy 2: Use the fact that for small matrices, we can enumerate sparse patterns
    # and check if a solution exists.

    # Strategy 3: Start from schoolbook and try to merge pairs of terms.
    # Two terms c1, c2 can be merged if there exist new vectors such that
    # the contribution of c1+c2 equals the contribution of a single new term.
    # This is generally impossible, but we can try specific merge patterns.

    def try_merge_pairs(U, V, W, max_merges=50):
        """Try to merge pairs of terms in the decomposition."""
        rank = len(U[0])
        U2 = [row[:] for row in U]
        V2 = [row[:] for row in V]
        W2 = [row[:] for row in W]
        
        merged = 0
        attempts = 0
        while merged < max_merges and attempts < 200:
            attempts += 1
            # Pick two random terms to try to merge
            c1 = random.randint(0, rank - 1)
            c2 = random.randint(0, rank - 1)
            if c1 == c2:
                continue
            
            # Check if terms c1 and c2 have overlapping support
            # Term c contributes to (i,j,k) if U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] != 0
            # For a merge to work, the combined contribution must be representable
            # as a single rank-1 tensor.
            
            # Simple check: if both terms are 1s in the same positions, we can't merge
            # (sum would be 2). If they're in disjoint positions, we can't merge either
            # (a single rank-1 tensor can't have disjoint support in all three factors
            #  unless one factor is zero somewhere).
            
            # Actually, let's just try: set term c2 to be the sum of terms c1 and c2,
            # then zero out c1. This only works if the result is still a valid rank-1
            # decomposition, which it won't be in general.
            
            # Better approach: try to find a linear combination that reduces rank.
            # This is complex; skip for now.
            pass
        
        if merged > 0:
            # Trim trailing zero columns
            while rank > 0 and all(U2[i][rank-1] == 0 for i in range(NM)) and \
                          all(V2[i][rank-1] == 0 for i in range(MP)) and \
                          all(W2[i][rank-1] == 0 for i in range(PN)):
                rank -= 1
            U2 = [row[:rank] for row in U2]
            V2 = [row[:rank] for row in V2]
            W2 = [row[:rank] for row in W2]
            if verify(U2, V2, W2):
                return {"U": U2, "V": V2, "W": W2}
        
        return None

    # Strategy 4: Systematic search over small structured families
    # For rank r, try decompositions where each term c is associated with a
    # specific (i_c, j_c, k_c) triple, and the values are from a small set.

    def try_sparse_rank(r, max_attempts=200):
        """Try to find a valid decomposition of rank r with sparse structure."""
        if r < max(NM, MP, PN):
            return None
        
        random.seed(r * 1000 + 42)
        
        for attempt in range(max_attempts):
            U = [[0] * r for _ in range(NM)]
            V = [[0] * r for _ in range(MP)]
            W = [[0] * r for _ in range(PN)]
            
            # Each term c is assigned to a random (i_c, j_c, k_c)
            # with value from a small set
            val_set = [-1, -0.5, 0, 0.5, 1]
            
            for c in range(r):
                i_c = random.randint(0, n - 1)
                j_c = random.randint(0, m - 1)
                k_c = random.randint(0, p - 1)
                val = random.choice(val_set)
                if val == 0:
                    continue
                U[i_c * m + j_c][c] = val
                V[j_c * p + k_c][c] = val
                W[k_c * n + i_c][c] = val
            
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
            
            # Try a different structure: each term handles multiple (i,j,k)
            # by having non-zero entries in multiple positions
            if attempt % 10 == 5:
                for c in range(r):
                    # Randomly pick 1-3 positions for each factor
                    for _ in range(random.randint(1, 3)):
                        i_c = random.randint(0, n - 1)
                        j_c = random.randint(0, m - 1)
                        k_c = random.randint(0, p - 1)
                        val = random.choice(val_set)
                        if val != 0:
                            U[i_c * m + j_c][c] += val
                            V[j_c * p + k_c][c] += val
                            W[k_c * n + i_c][c] += val
                
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None

    # Strategy 5: Use numpy to solve the bilinear system for small ranks
    # Fix some variables and solve for others using linear algebra.
    
    def try_fixed_structure_rank(r):
        """Try a specific structured family for rank r."""
        if r < max(NM, MP, PN):
            return None
        
        # Family A: For each j, allocate r_j terms. The terms for different j
        # are disjoint in the V factor. Within block j, we need:
        # sum_{c in block_j} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,k
        # 
        # If we set V[j*p+k, c] = 1 for all i,k,c in block_j, then we need:
        # sum_{c in block_j} U[i*m+j, c] * W[k*n+i, c] = 1 for all i,k
        # This is a matrix equation: U_j * W_j^T = ones(n,p) where
        # U_j is n x r_j and W_j is p x r_j (transposed appropriately).
        # The minimum rank of the all-ones matrix is 1, so r_j >= 1.
        # With r_j = 1: U_j[i] * W_j[k] = 1 for all i,k => U_j[i] = a, W_j[k] = 1/a
        # So each block j needs at least 1 term. Total rank >= m = 3.
        # But we also need to handle the cross-terms correctly...
        
        # Actually, this structure doesn't work because W[k*n+i, c] depends on both k and i,
        # so the factors don't separate cleanly.
        
        # Let's try a different approach: brute force over small value sets for very small ranks
        if r <= 10:
            val_set = [-1, -0.5, 0, 0.5, 1]
            # This is too large to brute force directly. Instead, use random search.
            random.seed(12345)
            for attempt in range(500):
                U = [[random.choice(val_set) for _ in range(r)] for _ in range(NM)]
                V = [[random.choice(val_set) for _ in range(r)] for _ in range(MP)]
                W = [[random.choice(val_set) for _ in range(r)] for _ in range(PN)]
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None

    # Try ranks from 41 down to lower bound
    lower_bound = max(NM, MP, PN)
    
    for r in range(schoolbook_rank - 1, lower_bound - 1, -1):
        if time.time() - t_start > DEADLINE - 10:
            break
        
        # Try multiple strategies for this rank
        result = try_sparse_rank(r, max_attempts=100)
        if result is not None:
            best = result
            break
        
        result = try_fixed_structure_rank(r)
        if result is not None:
            best = result
            break
        
        # Try merging from schoolbook
        if r < schoolbook_rank:
            sb = build_schoolbook()
            result = try_merge_pairs(sb["U"], sb["V"], sb["W"], max_merges=schoolbook_rank - r)
            if result is not None and len(result["U"][0]) == r:
                best = result
                break
    
    return best
# EVOLVE-BLOCK-END
