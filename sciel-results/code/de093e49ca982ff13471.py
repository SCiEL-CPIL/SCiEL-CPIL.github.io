"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    
    Uses algebraic methods (Groebner-basis-inspired feasibility checks) to search
    for lower-rank decompositions, with schoolbook as fallback.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n  # 12, 20, 15
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((N, M, P), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    def make_decomposition(U, V, W):
        """Create a decomposition from U, V, W matrices."""
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        T_rec = np.zeros((N, M, P), dtype=np.int64)
        for c in range(rank):
            T_rec += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        return np.array_equal(T_rec, T)
    
    def schoolbook_decomposition():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((N, rank), dtype=np.int64)
        V = np.zeros((M, rank), dtype=np.int64)
        W = np.zeros((P, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Start with schoolbook
    best_U, best_V, best_W = schoolbook_decomposition()
    best_rank = n * m * p
    
    def check_time():
        return time.time() > deadline - 5
    
    def try_rank(r):
        """
        Try to find a rank-r decomposition using structured search.
        
        For the matrix multiplication tensor, we can exploit the structure:
        T[i*m+j, j*p+k, k*n+i] = 1
        
        We seek U (N×r), V (M×r), W (P×r) such that:
        sum_c U[a,c]*V[b,c]*W[d,c] = T[a,b,d]
        
        Strategy: Fix some entries to be 0 or 1 (half-integers), then solve
        the remaining system. Use a greedy assignment with backtracking.
        
        For efficiency, we'll try to find sparse solutions where each term
        has limited support.
        """
        if check_time():
            return None
        
        # For small rank reductions from schoolbook, try to merge terms
        # where the index patterns allow cancellation.
        
        # Key insight: In the schoolbook decomposition, each term (i,j,k) has
        # U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
        
        # To reduce rank, we need to find linear dependencies or use
        # Strassen-like combinations.
        
        # Approach: For a given rank r < 60, we can try to assign each of the
        # 60 tensor entries to one of r "slots" (terms), where each slot
        # covers multiple entries via a rank-1 tensor.
        
        # A rank-1 tensor u⊗v⊗w has entries u[a]*v[b]*w[d].
        # For it to contribute to T[a,b,d] = 1 at specific positions, we need
        # u[a]*v[b]*w[d] to match the target.
        
        # Let's try a different approach: use the fact that for the MM tensor,
        # we can write it as a sum of terms based on the "middle" index j.
        
        # For fixed j, the sub-tensor over i,k has rank min(n,p) in general,
        # but for the specific MM structure, each j-slice is rank-1 (outer product
        # of e_i and e_k patterns).
        
        # Actually, let's try a more direct algebraic approach:
        # Fix V to have a specific structure, then solve for U and W.
        
        # If we fix V to be the identity-like structure, we can reduce the problem.
        
        # Let's try: set V to have 1s at specific positions and solve for U, W.
        
        # For rank r, we have r terms. Let's try to assign which tensor entries
        # each term covers.
        
        # Simpler heuristic: try to find if rank r is feasible by checking
        # if we can partition the 60 entries into r groups where each group
        # is "rank-1 compatible."
        
        # Two entries (i1,j1,k1) and (i2,j2,k2) can be in the same rank-1 term
        # if there exist u, v, w such that:
        # u[i1*m+j1]*v[j1*p+k1]*w[k1*n+i1] = 1
        # u[i2*m+j2]*v[j2*p+k2]*w[k2*n+i2] = 1
        # And for all other (a,b,d), u[a]*v[b]*w[d] = 0 or matches T.
        
        # This is complex. Let's try a simpler greedy approach.
        
        # Greedy: Start with schoolbook (60 terms). Try to merge pairs of terms
        # if possible.
        
        if r >= best_rank:
            return None
        
        # Try to find a decomposition with rank r using a structured pattern.
        # For the MM tensor, one known approach is to use the "border rank" 
        # properties, but for exact rank we need explicit decomposition.
        
        # Let's try a randomized search with structured initialization.
        
        # Initialize U, V, W with sparse patterns
        np.random.seed(42)
        
        for attempt in range(10):
            if check_time():
                return None
            
            # Create sparse U, V, W
            U = np.zeros((N, r), dtype=np.float64)
            V = np.zeros((M, r), dtype=np.float64)
            W = np.zeros((P, r), dtype=np.float64)
            
            # Assign each of the 60 entries to one of r terms
            entries = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        entries.append((i, j, k))
            
            # Randomly assign entries to terms
            assignment = np.random.randint(0, r, len(entries))
            
            for idx, (i, j, k) in enumerate(entries):
                c = assignment[idx]
                a = i * m + j
                b = j * p + k
                d = k * n + i
                U[a, c] = 1
                V[b, c] = 1
                W[d, c] = 1
            
            # Now check if this assignment is valid (no conflicts)
            # A conflict occurs if for some term c, there are two entries
            # (a1,b1,d1) and (a2,b2,d2) assigned to c such that the rank-1
            # tensor u_c⊗v_c⊗w_c doesn't match T at all positions.
            
            valid = True
            for c in range(r):
                # Get all entries assigned to term c
                term_entries = [(entries[idx][0], entries[idx][1], entries[idx][2]) 
                               for idx in range(len(entries)) if assignment[idx] == c]
                
                if not term_entries:
                    continue
                
                # Check if a single rank-1 tensor can cover all these entries
                # u_c⊗v_c⊗w_c must have 1s exactly at the positions of term_entries
                # and 0s elsewhere (or at least not create spurious non-zeros)
                
                # For this to work, all entries in term_entries must be "compatible"
                # i.e., they must lie on a "combinatorial line" in the tensor.
                
                # Two entries (i1,j1,k1) and (i2,j2,k2) are compatible if:
                # - They share the same i, same j, or same k (at least one coordinate same)
                # - More precisely, the set of (a,b,d) positions must form a rank-1 pattern
                
                # For the MM tensor, entries (i,j,k) map to positions:
                # a = i*m+j, b = j*p+k, d = k*n+i
                # A rank-1 term u⊗v⊗w has non-zeros at (a,b,d) where u[a]≠0, v[b]≠0, w[d]≠0
                # So the support is a "box" in the (a,b,d) space.
                
                # For the MM tensor, the 60 entries are all distinct positions.
                # A rank-1 term can cover multiple entries only if they form a
                # "sub-box" pattern.
                
                # Check compatibility: all entries must have the same set of
                # "free" coordinates.
                
                i_vals = set(e[0] for e in term_entries)
                j_vals = set(e[1] for e in term_entries)
                k_vals = set(e[2] for e in term_entries)
                
                # For a rank-1 tensor to cover exactly these entries (and no others
                # from the 60), we need the product of the sizes to equal the count,
                # AND the entries must form a complete grid.
                
                expected_count = len(i_vals) * len(j_vals) * len(k_vals)
                actual_count = len(term_entries)
                
                if expected_count != actual_count:
                    valid = False
                    break
                
                # Also check that no "missing" entries in the grid are in the 60
                grid_entries = set()
                for i in i_vals:
                    for j in j_vals:
                        for k in k_vals:
                            grid_entries.add((i, j, k))
                
                if grid_entries != set(term_entries):
                    valid = False
                    break
            
            if valid:
                # Verify the full decomposition
                U_int = U.astype(np.int64)
                V_int = V.astype(np.int64)
                W_int = W.astype(np.int64)
                if verify_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
        
        return None
    
    # Try ranks from best_rank-1 down to a lower bound
    # Lower bound: max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # Realistic lower bound is higher, but let's try a few
    
    lower_bound = max(n * p, m * p, n * m)
    
    for r in range(best_rank - 1, lower_bound - 1, -1):
        if check_time():
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            # Found a valid decomposition at rank r; stop searching lower
            break
    
    return make_decomposition(best_U, best_V, best_W)

# EVOLVE-BLOCK-END
