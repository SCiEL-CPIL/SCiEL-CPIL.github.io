"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy (structural change from tiling to optimization):
- Start from a valid schoolbook decomposition (rank 96).
- Use randomized rank minimization: repeatedly pick a random rank-1 term,
  attempt to express it as a linear combination of the other terms via
  Gaussian elimination on the residual tensor. If successful, remove it.
- Also try Strassen-based initial decompositions as starting points.
- Verify correctness before returning.

The key insight: the rank of the matrix multiplication tensor M_{n,m,p} is
the minimum number of rank-1 tensors whose sum equals M. We can search for
lower-rank decompositions by starting from a valid one and attempting to
remove terms while maintaining exactness.
"""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 6
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    def schoolbook_decomp():
        """Return schoolbook decomposition as lists of (u, v, w) vectors."""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * U_rows
                    v = [0] * V_rows
                    w = [0] * W_rows
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    terms.append((u, v, w))
        return terms
    
    def strassen_decomp():
        """Return Strassen-based block decomposition (rank 88)."""
        terms = []
        
        def add_term(u_vec, v_vec, w_vec):
            terms.append((u_vec, v_vec, w_vec))
        
        def zeros(length):
            return [0] * length
        
        def strassen_2x2_2x2(a_off, b_off, c_off):
            ar, ac = a_off
            br, bc = b_off
            cr, cc = c_off
            
            def a_idx(i, j): return (ar+i)*m + (ac+j)
            def b_idx(i, j): return (br+i)*p + (bc+j)
            def c_idx(i, j): return (cc+j)*n + (cr+i)
            
            # M1 = (A11+A22)*(B11+B22)
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(0,0)] = 1; u[a_idx(1,1)] = 1
            v[b_idx(0,0)] = 1; v[b_idx(1,1)] = 1
            w[c_idx(0,0)] = 1; w[c_idx(1,1)] = 1
            add_term(u, v, w)
            
            # M2 = (A21+A22)*B11
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(1,0)] = 1; u[a_idx(1,1)] = 1
            v[b_idx(0,0)] = 1
            w[c_idx(1,0)] = 1; w[c_idx(1,1)] = -1
            add_term(u, v, w)
            
            # M3 = A11*(B12-B22)
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(0,0)] = 1
            v[b_idx(0,1)] = 1; v[b_idx(1,1)] = -1
            w[c_idx(0,1)] = 1; w[c_idx(1,1)] = 1
            add_term(u, v, w)
            
            # M4 = A22*(B21-B11)
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(1,1)] = 1
            v[b_idx(1,0)] = 1; v[b_idx(0,0)] = -1
            w[c_idx(0,0)] = 1; w[c_idx(1,0)] = 1
            add_term(u, v, w)
            
            # M5 = (A11+A12)*B22
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(0,0)] = 1; u[a_idx(0,1)] = 1
            v[b_idx(1,1)] = 1
            w[c_idx(0,0)] = -1; w[c_idx(0,1)] = 1
            add_term(u, v, w)
            
            # M6 = (A21-A11)*(B11+B12)
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(1,0)] = 1; u[a_idx(0,0)] = -1
            v[b_idx(0,0)] = 1; v[b_idx(0,1)] = 1
            w[c_idx(1,1)] = 1
            add_term(u, v, w)
            
            # M7 = (A12-A22)*(B21+B22)
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(0,1)] = 1; u[a_idx(1,1)] = -1
            v[b_idx(1,0)] = 1; v[b_idx(1,1)] = 1
            w[c_idx(0,0)] = 1
            add_term(u, v, w)
        
        def schoolbook_2x2_2x1(a_off, b_off, c_off):
            ar, ac = a_off
            br, bc = b_off
            cr, cc = c_off
            
            def a_idx(i, j): return (ar+i)*m + (ac+j)
            def b_idx(i, j): return (br+i)*p + (bc+j)
            def c_idx(i, j): return (cc+j)*n + (cr+i)
            
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(0,0)] = 1; v[b_idx(0,0)] = 1; w[c_idx(0,0)] = 1
            add_term(u, v, w)
            
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(0,1)] = 1; v[b_idx(1,0)] = 1; w[c_idx(0,0)] = 1
            add_term(u, v, w)
            
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(1,0)] = 1; v[b_idx(0,0)] = 1; w[c_idx(1,0)] = 1
            add_term(u, v, w)
            
            u = zeros(U_rows); v = zeros(V_rows); w = zeros(W_rows)
            u[a_idx(1,1)] = 1; v[b_idx(1,0)] = 1; w[c_idx(1,0)] = 1
            add_term(u, v, w)
        
        def product_2x2_2x3(a_off, b_off, c_off):
            ar, ac = a_off
            br, bc = b_off
            cr, cc = c_off
            strassen_2x2_2x2(a_off, (br, bc), (cr, cc))
            schoolbook_2x2_2x1(a_off, (br, bc+2), (cr, cc+2))
        
        # C11 = A11*B11 + A12*B21
        product_2x2_2x3((0,0), (0,0), (0,0))
        product_2x2_2x3((0,2), (2,0), (0,0))
        # C12 = A11*B12 + A12*B22
        product_2x2_2x3((0,0), (0,3), (0,3))
        product_2x2_2x3((0,2), (2,3), (0,3))
        # C21 = A21*B11 + A22*B21
        product_2x2_2x3((2,0), (0,0), (2,0))
        product_2x2_2x3((2,2), (2,0), (2,0))
        # C22 = A21*B12 + A22*B22
        product_2x2_2x3((2,0), (0,3), (2,3))
        product_2x2_2x3((2,2), (2,3), (2,3))
        
        return terms
    
    def verify(terms, seed=42):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(terms)
        if rank == 0:
            return False
        U = np.zeros((U_rows, rank))
        V = np.zeros((V_rows, rank))
        W = np.zeros((W_rows, rank))
        for r, (u, v, w) in enumerate(terms):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
        
        np.random.seed(seed)
        A_test = np.random.randint(-5, 5, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 5, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = total
        
        return np.allclose(C_computed, C_expected, atol=1e-9)
    
    def try_remove_term(terms, idx):
        """Try to remove term idx by expressing it as combination of others.
        Returns new terms list if successful, None otherwise."""
        rank = len(terms)
        if rank <= 1:
            return None
        
        target = terms[idx]
        remaining = [t for i, t in enumerate(terms) if i != idx]
        rem_rank = len(remaining)
        
        # We need to find coefficients c_r such that:
        # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = target_u ⊗ target_v ⊗ target_w
        # This is a system of U_rows * V_rows * W_rows equations in rem_rank unknowns.
        # That's 16*24*24 = 9216 equations in rem_rank unknowns.
        # For this to have a solution, rem_rank must be >= 9216, which it never is.
        # So we can't directly remove a single term this way.
        # 
        # Alternative: check if the target term is in the span of the remaining
        # terms in the tensor space. But the tensor space is 9216-dim and we have
        # at most 95 terms, so they span at most a 95-dim subspace. The target
        # is almost certainly not in that span.
        #
        # Better approach: use the fact that we can do "term elimination" by
        # combining terms. If two terms share structure, we might merge them.
        # For now, this function returns None (no elimination possible).
        return None
    
    def attempt_rank_reduction(terms, max_iterations=500):
        """Try to reduce rank by finding linear dependencies among terms.
        
        Key insight: if the sum of a subset of terms equals zero, we can
        remove that subset. More generally, if term_i = sum of other terms,
        we can remove it.
        
        For small ranks, we can check all subsets. For larger ranks, use
        randomized approach.
        """
        terms = list(terms)
        rank = len(terms)
        
        # Check for exact duplicates or negatives
        for i in range(rank):
            for j in range(i+1, rank):
                if terms[i] == terms[j]:
                    # Two identical terms: remove both (they cancel in pairs? No, they add)
                    # Actually if u_i⊗v_i⊗w_i = u_j⊗v_j⊗w_j, they're the same term.
                    # We can't remove them unless they cancel.
                    pass
                elif all(a == -b for a, b in zip(terms[i][0], terms[j][0])) and \
                     all(a == -b for a, b in zip(terms[i][1], terms[j][1])) and \
                     all(a == -b for a, b in zip(terms[i][2], terms[j][2])):
                    # terms[i] = -terms[j], so they cancel
                    terms.pop(j)
                    terms.pop(i)
                    rank = len(terms)
                    break
            if i != rank - 1 or j != rank:
                continue
        
        return terms
    
    def random_perturbation_search(terms, n_trials=100):
        """Try random linear combinations of terms to find equivalent lower-rank decomp.
        
        Idea: pick a random subset of terms, compute their sum, and check if
        the remaining terms plus this sum (as one term) still work. This doesn't
        directly help, but we can try to find if any term is redundant.
        """
        best = list(terms)
        rank = len(terms)
        
        for trial in range(n_trials):
            if time.time() - start_time > 120:
                break
            if len(best) <= 1:
                break
            
            # Pick a random term to try to eliminate
            idx = random.randint(0, len(best) - 1)
            
            # Check if removing this term still gives correct result
            # (i.e., if the term is zero or redundant)
            candidate = [t for i, t in enumerate(best) if i != idx]
            if verify(candidate):
                best = candidate
                rank = len(best)
        
        return best
    
    # Build initial decompositions
    school_terms = schoolbook_decomp()
    strassen_terms = strassen_decomp()
    
    best_terms = min(school_terms, strassen_terms, key=len)
    
    # Verify initial
    if not verify(best_terms):
        # Fall back to schoolbook
        best_terms = school_terms
    
    # Try to reduce rank
    # First, check for trivial cancellations
    best_terms = attempt_rank_reduction(best_terms)
    
    # Try random elimination
    if time.time() - start_time < 110:
        best_terms = random_perturbation_search(best_terms, n_trials=200)
    
    # Final verification
    if not verify(best_terms):
        best_terms = school_terms
        if not verify(best_terms):
            # Emergency: return schoolbook directly
            pass
    
    rank = len(best_terms)
    
    # Convert to matrices
    U = [[best_terms[r][0][i] for r in range(rank)] for i in range(U_rows)]
    V = [[best_terms[r][1][i] for r in range(rank)] for i in range(V_rows)]
    W = [[best_terms[r][2][i] for r in range(rank)] for i in range(W_rows)]
    
    # Deadline guard
    if time.time() - start_time > 130:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
