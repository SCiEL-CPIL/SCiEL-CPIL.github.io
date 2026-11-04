"""Matrix multiplication 4x5x6 rank minimization."""
import time
import itertools
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Tensor entries: (i, j, k) for i in range(n), j in range(m), k in range(p)
    # Total entries: n*m*p = 120
    all_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # For each entry (i,j,k), the set of other entries that share:
    # - same i: entries (i, j', k') for all j', k'
    # - same j: entries (i', j, k') for all i', k'  
    # - same k: entries (i', j', k) for all i', j'
    # A rank term covering (i,j,k) with U[i,j]=1, V[j,k]=1, W[i,k]=1
    # can also cover any (i',j',k') where U[i',j'] != 0, V[j',k'] != 0, W[i',k'] != 0
    
    # Greedy approach: 
    # A "full" rank term for entry (i,j,k) covers all entries sharing i OR j OR k
    # Let's compute the coverage of a "star" centered at (i,j,k)
    
    def star_coverage(i, j, k):
        """All entries that share i with (i,j,k), OR share j, OR share k."""
        covered = set()
        for j2 in range(m):
            for k2 in range(p):
                covered.add((i, j2, k2))
        for i2 in range(n):
            for k2 in range(p):
                covered.add((i2, j, k2))
        for i2 in range(n):
            for j2 in range(m):
                covered.add((i2, j2, k))
        return covered
    
    # Try greedy: repeatedly pick the star that covers most uncovered entries
    # Then try to improve with local search / backtracking
    
    def greedy_solve(entries_set):
        """Greedy set cover using stars."""
        uncovered = set(entries_set)
        terms = []
        while uncovered:
            # Find star that covers most uncovered
            best_star = None
            best_count = -1
            for (i, j, k) in list(uncovered):
                cov = star_coverage(i, j, k) & uncovered
                if len(cov) > best_count:
                    best_count = len(cov)
                    best_star = (i, j, k)
            if best_star is None:
                break
            i, j, k = best_star
            terms.append((i, j, k))
            cov = star_coverage(i, j, k) & uncovered
            uncovered -= cov
        return terms
    
    # Initial greedy solution
    initial_terms = greedy_solve(set(all_entries))
    best_rank = len(initial_terms)
    best_terms = list(initial_terms)
    
    # Now try to improve: use a more sophisticated approach
    # Key insight: we can use "partial" stars or combine constraints
    # Actually, the star coverage is the maximum a single rank term can cover
    # given the tensor structure. Each rank term MUST be a star (or subset).
    
    # Let's try: for each entry, the rank term covering it must have
    # U[i,j]=1, V[j,k]=1, W[i,k]=1 (non-zero). The term can also have
    # other non-zero entries in U, V, W, which would cover additional tensor entries.
    # But it CANNOT cover an entry (i',j',k') unless U[i',j'] * V[j',k'] * W[i',k'] != 0
    # AND that entry exists in the tensor (which all do for 4x5x6).
    
    # So the maximum coverage of a rank term centered to cover (i,j,k) is indeed the star.
    # The question is: can we do better than greedy?
    
    # Let's try randomized greedy with different orderings
    import random
    rng = random.Random(42)
    
    best_rank = len(initial_terms)
    best_terms = list(initial_terms)
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        shuffled = list(all_entries)
        rng.shuffle(shuffled)
        uncovered = set(all_entries)
        terms = []
        for entry in shuffled:
            if entry not in uncovered:
                continue
            i, j, k = entry
            cov = star_coverage(i, j, k) & uncovered
            if cov:
                terms.append((i, j, k))
                uncovered -= cov
        if len(terms) < best_rank:
            best_rank = len(terms)
            best_terms = list(terms)
    
    # Try a more careful greedy: at each step, consider ALL entries (not just uncovered)
    # and pick the one whose star covers most uncovered
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        uncovered = set(all_entries)
        terms = []
        while uncovered:
            best_entry = None
            best_count = -1
            # Sample from uncovered for speed
            candidates = list(uncovered)
            if len(candidates) > 30:
                candidates = rng.sample(candidates, 30)
            for entry in candidates:
                i, j, k = entry
                cov = star_coverage(i, j, k) & uncovered
                c = len(cov)
                if c > best_count:
                    best_count = c
                    best_entry = entry
            if best_entry is None:
                break
            i, j, k = best_entry
            terms.append((i, j, k))
            cov = star_coverage(i, j, k) & uncovered
            uncovered -= cov
        if len(terms) < best_rank:
            best_rank = len(terms)
            best_terms = list(terms)
    
    # Now try local improvement: swap terms
    # If we have terms list, try removing one and re-solving the rest
    if time.time() < deadline - 10:
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            # Try removing a random term and re-greedy the rest
            terms_copy = list(best_terms)
            if not terms_copy:
                break
            idx = rng.randint(0, len(terms_copy) - 1)
            removed = terms_copy.pop(idx)
            # What's uncovered now?
            covered_by_others = set()
            for (i, j, k) in terms_copy:
                covered_by_others |= star_coverage(i, j, k)
            uncovered = set(all_entries) - covered_by_others
            if not uncovered:
                # Removed term wasn't needed!
                best_rank = len(terms_copy)
                best_terms = terms_copy
                continue
            # Re-solve uncovered greedily
            new_terms = list(terms_copy)
            while uncovered:
                best_entry = None
                best_count = -1
                candidates = list(uncovered)
                if len(candidates) > 20:
                    candidates = rng.sample(candidates, 20)
                for entry in candidates:
                    i, j, k = entry
                    cov = star_coverage(i, j, k) & uncovered
                    c = len(cov)
                    if c > best_count:
                        best_count = c
                        best_entry = entry
                if best_entry is None:
                    break
                i, j, k = best_entry
                new_terms.append((i, j, k))
                cov = star_coverage(i, j, k) & uncovered
                uncovered -= cov
            if len(new_terms) < best_rank:
                best_rank = len(new_terms)
                best_terms = new_terms
    
    # Build the decomposition from best_terms
    # Each term (i,j,k) means: U has 1 at (i,j), V has 1 at (j,k), W has 1 at (i,k)
    # But multiple terms may share the same (i,j) in U, etc.
    # We need to assign coefficients so that for each tensor entry (i,j,k):
    # sum over terms t of U_t[i,j] * V_t[j,k] * W_t[i,k] = 1
    # And for entries NOT in the tensor (but all entries ARE in the tensor for full matmul):
    # Actually for 4x5x6, ALL (i,j,k) with i<4,j<5,k<6 are tensor entries.
    
    # Wait - the tensor for matrix multiplication (4x5)*(5x6) has entries at ALL
    # (i,j,k) positions. So we need the decomposition to give 1 at every position.
    
    # With the star approach: term (i0,j0,k0) gives U[i0,j0]=1, V[j0,k0]=1, W[i0,k0]=1
    # This term contributes 1 to entry (i,j,k) iff i==i0 AND j==j0 AND k==k0... NO!
    # It contributes U[i,j]*V[j,k]*W[i,k]. If U only has 1 at (i0,j0), then U[i,j]=1 only if (i,j)==(i0,j0).
    # So a "pure" star term only covers ONE entry!
    
    # I was confused. Let me reconsider.
    # A rank term is defined by vectors u (length n*m), v (length m*p), w (length p*n).
    # The contribution to entry (i,j,k) is u[i*m+j] * v[j*p+k] * w[k*n+i].
    # 
    # For the term to cover entry (i0,j0,k0), we need u[i0*m+j0]*v[j0*p+k0]*w[k0*n+i0] != 0.
    # The term ALSO covers entry (i,j,k) if u[i*m+j]*v[j*p+k]*w[k*n+i] != 0.
    #
    # To maximize coverage, we want u to have non-zeros at many (i,j) positions,
    # v at many (j,k), w at many (k,i).
    #
    # The "star" centered at (i0,j0,k0) would have:
    # u[i*m+j] != 0 for all j (fixed i=i0)  -> covers all (i0, j, *)
    # v[j*p+k] != 0 for all k (fixed j=j0)  -> covers all (*, j0, *)
    # w[k*n+i] != 0 for all i (fixed k=k0)  -> covers all (*, *, k0)
    #
    # So the star covers entries where i=i0 OR j=j0 OR k=k0.
    # That's what I had before! The star DOES cover multiple entries.
    #
    # BUT: the contribution to each covered entry is 1 (if we set all to 1).
    # For entries covered by MULTIPLE terms, the contributions ADD.
    # We need the sum to equal 1 for each entry.
    
    # So the problem is: assign coefficients to terms so that for each (i,j,k),
    # sum_{t: (i,j,k) in star(t)} coeff_t = 1
    
    # This is a system of linear equations! Let me solve it.
    
    # For the current best_terms, set up the system:
    # For each entry e=(i,j,k): sum of coeff_t for t in terms where e in star(t) = 1
    
    # Let's build this system and solve it
    def solve_coefficients(terms, all_entries):
        """Solve for coefficients such that each entry gets sum=1."""
        num_terms = len(terms)
        num_entries = len(all_entries)
        
        # Build incidence matrix: A[entry_idx, term_idx] = 1 if entry in star(term)
        # A is num_entries x num_terms
        # We want A^T * coeff = 1 (vector of ones)
        # i.e., for each entry: sum of coeff_t over terms covering it = 1
        
        # Use numpy for the linear solve
        import numpy as np
        
        A = np.zeros((num_entries, num_terms), dtype=float)
        for t_idx, (i, j, k) in enumerate(terms):
            cov = star_coverage(i, j, k)
            for e_idx, (ei, ej, ek) in enumerate(all_entries):
                if (ei, ej, ek) in cov:
                    A[e_idx, t_idx] = 1.0
        
        b = np.ones(num_entries, dtype=float)
        
        # Solve A^T * x = b in least squares sense, but we want exact
        # Actually we want: for each row e of A, sum(A[e,:]*x) = 1
        # This is A @ x = b
        
        try:
            # Use least squares
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if solution is exact
            residual = np.max(np.abs(A @ x - b))
            if residual < 1e-10:
                return x, True
            else:
                # Try to find a sparse/exact solution
                return x, False
        except:
            return None, False
    
    # Let's verify our best solution
    if best_terms:
        coeffs, is_exact = solve_coefficients(best_terms, all_entries)
        if coeffs is not None and is_exact:
            # Great! Build the matrices
            rank = len(best_terms)
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            for t_idx, (i, j, k) in enumerate(best_terms):
                c = coeffs[t_idx]
                # Star coverage: u has 1s at (i, j') for all j'
                for j2 in range(m):
                    U[i * m + j2][t_idx] = c
                # v has 1s at (j, k') for all k'
                for k2 in range(p):
                    V[j * p + k2][t_idx] = c
                # w has 1s at (k, i') for all i'
                for i2 in range(n):
                    W[k * n + i2][t_idx] = c
            
            # Verify
            import numpy as np
            U_np = np.array(U)
            V_np = np.array(V)
            W_np = np.array(W)
            
            # Check: for each (i,j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for t in range(rank):
                            val += U_np[i*m+j, t] * V_np[j*p+k, t] * W_np[k*n+i, t]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-9:
                # Convert to fractions for exactness
                from fractions import Fraction
                U_frac = [[str(Fraction(U[i][t]).limit_denominator(1000)) for t in range(rank)] for i in range(n*m)]
                # Actually, let's just return floats that are exact
                # The coefficients from lstsq might not be clean
                # Let's try to round to simple fractions
                U_out = [[0] * rank for _ in range(n * m)]
                V_out = [[0] * rank for _ in range(m * p)]
                W_out = [[0] * rank for _ in range(p * n)]
                
                for t_idx, (i, j, k) in enumerate(best_terms):
                    c = coeffs[t_idx]
                    # Round to nearest simple fraction
                    from fractions import Fraction
                    frac = Fraction(c).limit_denominator(100)
                    cv = float(frac)
                    for j2 in range(m):
                        U_out[i * m + j2][t_idx] = cv
                    for k2 in range(p):
                        V_out[j * p + k2][t_idx] = cv
                    for i2 in range(n):
                        W_out[k * n + i2][t_idx] = cv
                
                # Final verification with rounded values
                max_err2 = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for t in range(rank):
                                val += U_out[i*m+j][t] * V_out[j*p+k][t] * W_out[k*n+i][t]
                            err = abs(val - 1.0)
                            if err > max_err2:
                                max_err2 = err
                
                if max_err2 < 1e-9:
                    return {"U": U_out, "V": V_out, "W": W_out}
        
        # If exact solution not found with current terms, fall through
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
