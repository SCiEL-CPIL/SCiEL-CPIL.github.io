import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    nU = n * m  # 12
    nV = m * p  # 32
    nW = p * n  # 24
    
    # Build target tensor indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    # All possible (a,b,c) triples
    all_triples = []
    for a in range(nU):
        for b in range(nV):
            for c in range(nW):
                all_triples.append((a, b, c))
    
    target_set = set(target_indices)
    non_target_indices = [t for t in all_triples if t not in target_set]
    
    def full_verify(U, V, W, rank):
        """Check if decomposition is exact - full check"""
        # Check target entries equal 1
        for a, b, c in target_indices:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            if abs(val - 1.0) > 1e-9:
                return False
        # Check ALL non-target entries equal 0
        for a, b, c in non_target_indices:
            val = 0.0
            for cp in range(rank):
                val += U[a][cp] * V[b][cp] * W[c][cp]
            if abs(val) > 1e-9:
                return False
        return True
    
    # Build schoolbook decomposition
    schoolbook_rank = n * m * p
    schoolbook_U = [[0.0] * schoolbook_rank for _ in range(nU)]
    schoolbook_V = [[0.0] * schoolbook_rank for _ in range(nV)]
    schoolbook_W = [[0.0] * schoolbook_rank for _ in range(nW)]
    
    # Map from (i,j,k) to column index
    term_to_col = {}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                schoolbook_U[a][col] = 1.0
                schoolbook_V[b][col] = 1.0
                schoolbook_W[c][col] = 1.0
                term_to_col[(i, j, k)] = col
                col += 1
    
    best_U = [row[:] for row in schoolbook_U]
    best_V = [row[:] for row in schoolbook_V]
    best_W = [row[:] for row in schoolbook_W]
    best_rank = schoolbook_rank
    
    # Helper: try to merge a set of terms into fewer columns
    # For a set of terms S, check if we can replace their columns with fewer columns
    # that produce the same tensor contributions
    
    def try_merge_terms(terms):
        """
        Try to merge a set of terms into a single column.
        Returns (U_vals, V_vals, W_vals) if mergeable, else None.
        U_vals: dict from a-index to value
        V_vals: dict from b-index to value  
        W_vals: dict from c-index to value
        """
        if len(terms) <= 1:
            return None
        
        # For each term (i,j,k), the required contribution is:
        # U[a] * V[b] * W[c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
        # And for any non-target (a',b',c'), the sum must be 0
        
        # For a single column merge, we need:
        # For each term t in terms: U[a_t] * V[b_t] * W[c_t] = 1
        # For each non-target (a',b',c') that overlaps with these indices:
        #   U[a'] * V[b'] * W[c'] = 0 (if only this column contributes)
        
        # Collect all unique indices
        a_indices = set()
        b_indices = set()
        c_indices = set()
        for (i, j, k) in terms:
            a_indices.add(i * m + j)
            b_indices.add(j * p + k)
            c_indices.add(k * n + i)
        
        # For a valid merge with one column, we need to assign values such that:
        # 1. For each (i,j,k) in terms: U[i*m+j] * V[j*p+k] * W[k*n+i] = 1
        # 2. For any (a',b',c') where a' in a_indices, b' in b_indices, c' in c_indices
        #    but (a',b',c') is NOT a target: U[a'] * V[b'] * W[c'] = 0
        
        # This is a system of equations. Let's set up variables.
        # Use log-space: let u_a = log(U[a]), v_b = log(V[b]), w_c = log(W[c])
        # Then u_a + v_b + w_c = 0 for targets, and u_a + v_b + w_c = -inf for non-targets
        
        # Actually, let's work directly with the values.
        # The constraint is that the product U[a]*V[b]*W[c] = 1 for targets
        # and = 0 for non-targets within the index sets.
        
        # A non-target (a',b',c') within our index sets requires at least one of
        # U[a'], V[b'], W[c'] to be 0.
        
        # Let's find all non-target triples within our index sets
        conflicting_triples = []
        for a in a_indices:
            for b in b_indices:
                for c in c_indices:
                    if (a, b, c) not in target_set:
                        conflicting_triples.append((a, b, c))
        
        # For the merge to work, for each conflicting triple (a,b,c),
        # at least one of U[a], V[b], W[c] must be 0.
        
        # Also, for each target (i,j,k) in terms, U[i*m+j]*V[j*p+k]*W[k*n+i] = 1,
        # which means none of U[i*m+j], V[j*p+k], W[k*n+i] can be 0.
        
        # So we need to assign 0/non-zero values to U, V, W entries such that:
        # - All target indices have non-zero values
        # - Every conflicting triple has at least one zero
        
        # This is a hitting set problem. Let's try to find a valid assignment.
        
        # For simplicity, try a few patterns:
        # Pattern 1: Set some U values to 0
        # Pattern 2: Set some V values to 0
        # Pattern 3: Set some W values to 0
        
        # Actually, for a single-column merge of multiple terms, it's generally
        # impossible unless the terms are very structured. Let's check if it's
        # even possible by looking at the structure.
        
        # For terms with same (i,j) but different k:
        # a is the same, b varies, c varies
        # Conflicting: (a, b_k1, c_k2) for k1 != k2
        # Need U[a]*V[b_k1]*W[c_k2] = 0
        # But U[a]*V[b_k1]*W[c_k1] = 1 and U[a]*V[b_k2]*W[c_k2] = 1
        # So U[a] != 0, V[b_k1] != 0, W[c_k1] != 0, V[b_k2] != 0, W[c_k2] != 0
        # Then U[a]*V[b_k1]*W[c_k2] != 0, contradiction!
        # So we CANNOT merge terms with same (i,j) different k into one column.
        
        # Similarly for other patterns.
        
        # What about merging terms that don't share indices?
        # If two terms (i1,j1,k1) and (i2,j2,k2) have:
        # a1 != a2, b1 != b2, c1 != c2
        # Then there are no conflicting triples within the index sets
        # (since any (a',b',c') with a' in {a1,a2}, b' in {b1,b2}, c' in {c1,c2}
        #  that's a target must be one of the two terms)
        # Wait, that's not right. (a1, b1, c2) might be a non-target.
        
        # Let me just check directly
        for (i1, j1, k1) in terms:
            a1 = i1 * m + j1
            b1 = j1 * p + k1
            c1 = k1 * n + i1
            for (i2, j2, k2) in terms:
                if (i1,j1,k1) == (i2,j2,k2):
                    continue
                a2 = i2 * m + j2
                b2 = j2 * p + k2
                c2 = k2 * n + i2
                # Check if (a1, b1, c2) is a non-target
                if (a1, b1, c2) not in target_set:
                    return None
                if (a1, b2, c1) not in target_set:
                    return None
                if (a2, b1, c2) not in target_set:
                    return None
                if (a2, b2, c1) not in target_set:
                    return None
        
        # If we get here, the merge might be possible
        # For simplicity, set all values to 1
        U_vals = {}
        V_vals = {}
        W_vals = {}
        for (i, j, k) in terms:
            U_vals[i*m+j] = 1.0
            V_vals[j*p+k] = 1.0
            W_vals[k*n+i] = 1.0
        
        return (U_vals, V_vals, W_vals)
    
    # Try to find rank reductions by merging terms
    # Start with all terms, try to merge subsets
    
    # First, check if any 2-term merge is possible
    all_terms = list(term_to_col.keys())
    
    mergeable_pairs = []
    for i1 in range(len(all_terms)):
        if time.time() > deadline - 30:
            break
        for i2 in range(i1+1, len(all_terms)):
            pair = [all_terms[i1], all_terms[i2]]
            result = try_merge_terms(pair)
            if result is not None:
                mergeable_pairs.append(pair)
    
    # Try to apply merges
    # This is a complex optimization problem. Let's try a greedy approach.
    # Build a graph where edges are mergeable pairs, find independent set of merges
    
    # Actually, let's just try to find one merge and see if it works
    # For now, return schoolbook if no simple merges found
    
    if not mergeable_pairs:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try applying merges one at a time
    # For each mergeable pair, try to construct a decomposition with one fewer rank
    
    for pair in mergeable_pairs:
        if time.time() > deadline - 20:
            break
        
        (U_vals, V_vals, W_vals) = try_merge_terms(pair)
        
        # Create a new decomposition with rank = schoolbook_rank - 1
        # Remove the two columns for the pair, add one new column
        new_rank = schoolbook_rank - 1
        new_U = [[0.0] * new_rank for _ in range(nU)]
        new_V = [[0.0] * new_rank for _ in range(nV)]
        new_W = [[0.0] * new_rank for _ in range(nW)]
        
        # Copy all columns except the two for the pair
        col1 = term_to_col[pair[0]]
        col2 = term_to_col[pair[1]]
        
        new_col = 0
        for old_col in range(schoolbook_rank):
            if old_col == col1 or old_col == col2:
                continue
            for a in range(nU):
                new_U[a][new_col] = schoolbook_U[a][old_col]
            for b in range(nV):
                new_V[b][new_col] = schoolbook_V[b][old_col]
            for c in range(nW):
                new_W[c][new_col] = schoolbook_W[c][old_col]
            new_col += 1
        
        # Add the merged column
        for a, val in U_vals.items():
            new_U[a][new_col] = val
        for b, val in V_vals.items():
            new_V[b][new_col] = val
        for c, val in W_vals.items():
            new_W[c][new_col] = val
        
        # Verify
        if full_verify(new_U, new_V, new_W, new_rank):
            best_U = new_U
            best_V = new_V
            best_W = new_W
            best_rank = new_rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
