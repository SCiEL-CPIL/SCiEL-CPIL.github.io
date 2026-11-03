"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product, combinations
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    start_time = time.time()
    deadline = start_time + 135.0
    
    def validate_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid matrix multiplication tensor decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_expected[a, b, c] = 1.0
        
        T_actual = np.einsum('as,bs,cs->abc', U, V, W)
        
        return np.allclose(T_actual, T_expected, atol=1e-9)
    
    def schoolbook_decomposition():
        """Return the standard rank-60 decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        
        U_list = [[int(x) for x in row] for row in U.tolist()]
        V_list = [[int(x) for x in row] for row in V.tolist()]
        W_list = [[int(x) for x in row] for row in W.tolist()]
        
        return U_list, V_list, W_list
    
    def try_structured_sparse_search(target_rank, max_terms_per_row=3):
        """
        Try to find a rank-target_rank decomposition using structured sparse patterns.
        
        Each rank term s has:
        - A subset of U rows (at most max_terms_per_row) with values in {-1, 0, 1, 1/2}
        - A subset of V rows (at most max_terms_per_row) with values in {-1, 0, 1, 1/2}
        - A subset of W rows (at most max_terms_per_row) with values in {-1, 0, 1, 1/2}
        
        We search over structured patterns based on the natural index structure.
        """
        if time.time() > deadline:
            return None
        
        # For each rank term, we'll define it by which (i,j,k) indices it "covers"
        # A term can cover multiple (i,j,k) combinations if the pattern allows it.
        
        # Strategy: Try to merge schoolbook terms where possible.
        # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
        # if they share structure in a way that allows a single rank term to cover both.
        
        # A rank term with U supported on rows {i1*m+j1, i2*m+j2}, V on {j1*p+k1, j2*p+k2},
        # W on {k1*n+i1, k2*n+i2} can cover (i1,j1,k1) and (i2,j2,k2) if:
        # U[i1*m+j1]*V[j1*p+k1]*W[k1*n+i1] = 1
        # U[i2*m+j2]*V[j2*p+k2]*W[k2*n+i2] = 1
        # U[i1*m+j1]*V[j2*p+k2]*W[k1*n+i1] = 0 (cross terms must vanish)
        # U[i2*m+j2]*V[j1*p+k1]*W[k2*n+i2] = 0
        
        # This is possible if we can assign values such that the cross terms cancel.
        # With values in {±1, ±1/2}, this is sometimes possible.
        
        # Let's try a more systematic approach:
        # Generate all possible "patterns" for a rank term and try to cover all 60 entries.
        
        # A pattern is defined by:
        # - U_support: list of (row_idx, value) pairs
        # - V_support: list of (row_idx, value) pairs  
        # - W_support: list of (row_idx, value) pairs
        
        # The term contributes to T[a,b,c] = U[a]*V[b]*W[c] for all (a,b,c) in the supports.
        
        # We need the sum over all terms to give T[a,b,c] = 1 for valid (a,b,c) and 0 otherwise.
        
        # This is a set cover / exact cover problem. Let's use backtracking.
        
        # First, enumerate all valid (a,b,c) triples
        valid_triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_triples.append((a, b, c, i, j, k))
        
        # For a rank term to be "useful", it must contribute positively to at least one valid triple
        # and not contribute to any invalid triple (or if it does, other terms must cancel).
        
        # Let's try a simpler approach: search for decompositions where each term
        # covers exactly one or two valid triples, and cross terms are zero.
        
        # A term covers (i,j,k) if U[i*m+j], V[j*p+k], W[k*n+i] are all nonzero
        # and their product is 1.
        
        # For a term to cover (i1,j1,k1) and (i2,j2,k2) with no cross terms:
        # Need U[i1*m+j1]*V[j2*p+k2]*W[k1*n+i1] = 0
        # and U[i2*m+j2]*V[j1*p+k1]*W[k2*n+i2] = 0
        
        # This happens if, e.g., U[i1*m+j1] = 0 or V[j2*p+k2] = 0 or W[k1*n+i1] = 0
        # But we need U[i1*m+j1], V[j1*p+k1], W[k1*n+i1] all nonzero for the first triple.
        
        # So we need V[j2*p+k2] = 0 or W[k1*n+i1] = 0 or U[i1*m+j1] = 0 for the cross term.
        # But U[i1*m+j1] and W[k1*n+i1] are nonzero (needed for triple 1).
        # So we need V[j2*p+k2] = 0. But V[j2*p+k2] must be nonzero for triple 2.
        # Contradiction! Unless i1 != i2 or j1 != j2 or k1 != k2 in specific ways.
        
        # Actually, let me reconsider. The cross term for (i1, j2, k1) is:
        # U[i1*m+j1] * V[j2*p+k1] * W[k1*n+i1]
        # Note: V index is j2*p+k1, not j2*p+k2!
        
        # So the cross terms are:
        # For triple (i1,j1,k1) and (i2,j2,k2):
        # Cross1: U[i1*m+j1] * V[j2*p+k1] * W[k1*n+i1]  -> contributes to T[i1*m+j1, j2*p+k1, k1*n+i1]
        # Cross2: U[i2*m+j2] * V[j1*p+k2] * W[k2*n+i2]  -> contributes to T[i2*m+j2, j1*p+k2, k2*n+i2]
        
        # These are invalid triples (unless they happen to be valid).
        # T[i1*m+j1, j2*p+k1, k1*n+i1] is valid only if j2 == j1 (from the b index) 
        # AND k1 == k1 (trivially) AND i1 == i1 (trivially). So valid only if j1 == j2.
        # Similarly, T[i2*m+j2, j1*p+k2, k2*n+i2] is valid only if j1 == j2.
        
        # So if j1 != j2, the cross terms go to invalid positions and must be zero.
        # We can achieve this by setting V[j2*p+k1] = 0 or V[j1*p+k2] = 0,
        # but we need V[j1*p+k1] and V[j2*p+k2] nonzero.
        
        # If j1 != j2, then j2*p+k1 != j1*p+k2 (since j1 != j2), so we can independently
        # set V[j2*p+k1] = 0 and V[j1*p+k2] = 0 while keeping V[j1*p+k1] and V[j2*p+k2] nonzero.
        
        # So terms with different j values can be merged!
        
        # Let's try: group schoolbook terms by j value. For each j, we have n*p = 15 terms.
        # We can try to merge terms with the same j but different (i,k) pairs.
        
        # For terms (i1,j,k1) and (i2,j,k2) with same j:
        # Cross1: U[i1*m+j] * V[j*p+k1] * W[k1*n+i1] -> T[i1*m+j, j*p+k1, k1*n+i1] which is valid (it's triple 1 itself)
        # Wait, that's the same as the main term. Let me recompute.
        
        # Term covers (i1,j,k1) and (i2,j,k2):
        # U support: {i1*m+j, i2*m+j}
        # V support: {j*p+k1, j*p+k2}
        # W support: {k1*n+i1, k2*n+i2}
        
        # Contributions:
        # U[i1*m+j]*V[j*p+k1]*W[k1*n+i1] -> T[i1*m+j, j*p+k1, k1*n+i1] = should be 1 (triple 1)
        # U[i2*m+j]*V[j*p+k2]*W[k2*n+i2] -> T[i2*m+j, j*p+k2, k2*n+i2] = should be 1 (triple 2)
        # U[i1*m+j]*V[j*p+k2]*W[k1*n+i1] -> T[i1*m+j, j*p+k2, k1*n+i1] = should be 0 (invalid, since k2 != k1 generally)
        # U[i2*m+j]*V[j*p+k1]*W[k2*n+i2] -> T[i2*m+j, j*p+k1, k2*n+i2] = should be 0 (invalid)
        
        # The invalid positions: (i1*m+j, j*p+k2, k1*n+i1) is valid only if k2 == k1 (no) 
        # and the output index k1*n+i1 corresponds to (i1, k1) which is fine, but the V index j*p+k2
        # means the column is k2, so this is T for output (i1, k1) using input B column k2 - invalid.
        
        # To make cross terms zero: U[i1*m+j]*V[j*p+k2]*W[k1*n+i1] = 0
        # Since U[i1*m+j] and W[k1*n+i1] are nonzero (needed for triple 1), we need V[j*p+k2] = 0.
        # But V[j*p+k2] must be nonzero for triple 2. Contradiction again!
        
        # So we can't merge terms with the same j using simple ±1 values.
        
        # What if we use fractional values? Let U[i1*m+j] = a, U[i2*m+j] = b,
        # V[j*p+k1] = c, V[j*p+k2] = d, W[k1*n+i1] = e, W[k2*n+i2] = f.
        
        # Need: a*c*e = 1, b*d*f = 1, a*d*e = 0, b*c*f = 0.
        # From a*d*e = 0: since a, e nonzero, d = 0. But b*d*f = 0, not 1. Contradiction.
        
        # So we fundamentally cannot merge two terms with the same j using a single rank term.
        
        # What about merging terms with different j values?
        # Terms (i1,j1,k1) and (i2,j2,k2) with j1 != j2:
        # U support: {i1*m+j1, i2*m+j2}
        # V support: {j1*p+k1, j2*p+k2}
        # W support: {k1*n+i1, k2*n+i2}
        
        # Contributions to valid positions:
        # a*c*e -> T[i1*m+j1, j1*p+k1, k1*n+i1] = 1 (triple 1) ✓
        # b*d*f -> T[i2*m+j2, j2*p+k2, k2*n+i2] = 1 (triple 2) ✓
        
        # Cross terms:
        # a*d*e -> T[i1*m+j1, j2*p+k1, k1*n+i1]: Is this valid? 
        #   Valid if j2 == j1 (no, j1 != j2). So invalid, must be 0.
        #   Need a*d*e = 0. Since a, e nonzero, need d = 0. But b*d*f = 0, not 1. Contradiction!
        
        # Hmm, so even with different j, we can't merge.
        
        # The fundamental issue: any rank term that covers two valid triples will have
        # cross terms that land on invalid positions, and we can't zero them out without
        # also zeroing out the desired terms.
        
        # UNLESS the cross term lands on a position that is also valid and gets its
        # contribution from another term. But that complicates things.
        
        # Let me try a different approach: use the known lower bound and try to find
        # a decomposition near the lower bound using numerical optimization.
        
        # Actually, let me try a completely different strategy: use the fact that
        # we can add and subtract terms. A rank-r decomposition can have terms that
        # partially cancel.
        
        # For example, consider:
        # T = sum_s U_s ⊗ V_s ⊗ W_s
        # where some U_s, V_s, W_s have multiple nonzeros.
        
        # Let's try to find a rank-55 or so decomposition using gradient descent
        # on the tensor reconstruction error.
        
        return None
    
    def try_gradient_descent(target_rank, max_time=30.0):
        """
        Try to find a rank-target_rank decomposition using gradient descent.
        Minimize ||T_actual - T_expected||_F^2.
        """
        if time.time() > deadline:
            return None
        
        T_expected = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_expected[a, b, c] = 1.0
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        n_iters = 5000
        lr = 0.01
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7 + 42)
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            for it in range(n_iters):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                # T_actual[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
                # dT/dc = U[a,s]*V[b,s] for each s
                # dL/dU[a,s] = sum_{b,c} (T_actual[a,b,c] - T_expected[a,b,c]) * V[b,s] * W[c,s]
                
                # Efficient computation using einsum
                # T_actual = np.einsum('as,bs,cs->abc', U, V, W)
                # error = T_actual - T_expected
                # dU[a,s] = einsum('abc,bs,cs->as', error, V, W)
                # dV[b,s] = einsum('abc,as,cs->bs', error, U, W)
                # dW[c,s] = einsum('abc,as,bs->cs', error, U, V)
                
                T_actual = np.einsum('as,bs,cs->abc', U, V, W)
                error = T_actual - T_expected
                loss = np.sum(error * error)
                
                if loss < best_error:
                    best_error = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if loss < 1e-10:
                    break
                
                # Compute gradients
                dU = np.einsum('abc,bs,cs->as', error, V, W)
                dV = np.einsum('abc,as,cs->bs', error, U, W)
                dW = np.einsum('abc,as,bs->cs', error, U, V)
                
                # Gradient descent
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if it % 1000 == 999:
                    lr *= 0.5
        
        # Check if we found a valid decomposition
        if best_U is not None and best_error < 1e-8:
            # Round to nearest half-integer if close
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            # Validate rounded version
            T_actual = np.einsum('as,bs,cs->abc', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_actual, T_expected, atol=1e-9):
                U_list = [[float(x) for x in row] for row in U_rounded.tolist()]
                V_list = [[float(x) for x in row] for row in V_rounded.tolist()]
                W_list = [[float(x) for x in row] for row in W_rounded.tolist()]
                return U_list, V_list, W_list
        
        return None
    
    # Try gradient descent for various ranks
    for target_rank in range(55, 60):
        if time.time() > deadline:
            break
        
        result = try_gradient_descent(target_rank, max_time=20.0)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U_list, V_list, W_list = schoolbook_decomposition()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
