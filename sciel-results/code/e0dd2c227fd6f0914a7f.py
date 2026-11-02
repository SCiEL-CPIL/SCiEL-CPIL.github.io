import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nU = n * m  # 8
    nV = m * p  # 28
    nW = p * n  # 14
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify tensor decomposition exactly."""
        r = U_mat.shape[1]
        if r == 0:
            return n == 0 or m == 0 or p == 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U_mat[i*m + j, c] * V_mat[jp*p + k, c] * W_mat[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
    
    def tensor_residual(U_mat, V_mat, W_mat):
        """Compute the residual tensor T - sum of rank-1 terms."""
        r = U_mat.shape[1]
        residual = np.zeros((n, p, m, m), dtype=float)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U_mat[i*m + j, c] * V_mat[jp*p + k, c] * W_mat[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        residual[i, k, j, jp] = expected - s
        return residual
    
    def build_rank50():
        """Build the rank-50 Strassen+schoolbook decomposition."""
        rank = 50
        U = np.zeros((nU, rank), dtype=float)
        V = np.zeros((nV, rank), dtype=float)
        W = np.zeros((nW, rank), dtype=float)
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            w00 = k0 * n + i0
            w01 = (k0 + 1) * n + i0
            w10 = k0 * n + (i0 + 1)
            w11 = (k0 + 1) * n + (i0 + 1)
            
            U[a, col] += 1; U[d, col] += 1
            V[e, col] += 1; V[h, col] += 1
            W[w00, col] += 1; W[w11, col] += 1
            col += 1
            
            U[c, col] += 1; U[d, col] += 1
            V[e, col] += 1
            W[w10, col] += 1; W[w11, col] -= 1
            col += 1
            
            U[a, col] += 1
            V[f, col] += 1; V[h, col] -= 1
            W[w01, col] += 1; W[w11, col] += 1
            col += 1
            
            U[d, col] += 1
            V[g, col] += 1; V[e, col] -= 1
            W[w00, col] += 1; W[w10, col] += 1
            col += 1
            
            U[a, col] += 1; U[b, col] += 1
            V[h, col] += 1
            W[w00, col] -= 1; W[w01, col] += 1
            col += 1
            
            U[c, col] += 1; U[a, col] -= 1
            V[e, col] += 1; V[f, col] += 1
            W[w11, col] += 1
            col += 1
            
            U[b, col] += 1; U[d, col] -= 1
            V[g, col] += 1; V[h, col] += 1
            W[w00, col] += 1
            col += 1

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            w0 = k0 * n + i0
            w1 = k0 * n + (i0 + 1)
            
            U[a, col] += 1; V[e, col] += 1; W[w0, col] += 1
            col += 1
            U[b, col] += 1; V[g, col] += 1; W[w0, col] += 1
            col += 1
            U[c, col] += 1; V[e, col] += 1; W[w1, col] += 1
            col += 1
            U[d, col] += 1; V[g, col] += 1; W[w1, col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_222(0, j_start, k_start)
            add_schoolbook_221(0, j_start, 6)
        
        assert col == rank
        return U, V, W
    
    def try_term_elimination(U, V, W, max_rounds=200):
        """
        Structural approach: try to eliminate terms by expressing the
        contribution of one term as a combination of others.
        
        For each term t, check if its 'slice' (the rank-1 tensor it contributes)
        can be matched by adjusting other terms. We use the fact that if term t
        has a specific pattern, we can try to absorb it into nearby terms.
        
        Key insight: two terms can sometimes be merged if they share structure.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Strategy 1: Remove zero terms
            for r in range(rank - 1, -1, -1):
                if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    improved = True
                    break
            if improved:
                continue
            
            # Strategy 2: Try to remove each term and verify
            for r in range(rank - 1, -1, -1):
                if time.time() > deadline:
                    return U, V, W
                
                U_save = U[:, r].copy()
                V_save = V[:, r].copy()
                W_save = W[:, r].copy()
                
                U[:, r] = 0
                V[:, r] = 0
                W[:, r] = 0
                
                if verify(U, V, W):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    improved = True
                    break
                else:
                    U[:, r] = U_save
                    V[:, r] = V_save
                    W[:, r] = W_save
            
            if not improved:
                break
        
        return U, V, W
    
    def try_pair_merging(U, V, W, max_rounds=100):
        """
        Try to replace pairs of terms with a single term (if possible).
        This is a structural change: instead of just removing terms,
        we look for algebraic simplifications.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            merged = False
            for i in range(rank - 1):
                if time.time() > deadline:
                    return U, V, W
                for j in range(i + 1, rank):
                    # Check if terms i and j can be merged
                    # Two terms can be merged if their tensor contributions
                    # sum to a rank-1 tensor
                    # Term i contributes: U[:,i] * V[:,i] * W[:,i] (outer product structure)
                    # We need: U_i ⊗ V_i ⊗ W_i + U_j ⊗ V_j ⊗ W_j = rank-1 tensor
                    # This is a strong condition; let's check via residual
                    U_save = U[:, i].copy()
                    V_save = V[:, i].copy()
                    W_save = W[:, i].copy()
                    U_save2 = U[:, j].copy()
                    V_save2 = V[:, j].copy()
                    W_save2 = W[:, j].copy()
                    
                    # Try: set term i to zero, see if we can adjust term j
                    # to compensate. This is complex, so use a simpler heuristic:
                    # If the two terms have the same support pattern in one factor,
                    # they might be mergeable.
                    
                    # Check if U[:,i] and U[:,j] are proportional
                    ui = U[:, i]
                    uj = U[:, j]
                    vi = V[:, i]
                    vj = V[:, j]
                    wi = W[:, i]
                    wj = W[:, j]
                    
                    # Simple check: if U_i = alpha * U_j and V_i = beta * V_j
                    # then the sum is (alpha*U_j + U_j) ⊗ ... which isn't rank-1
                    # unless one is zero. So pair merging is rare.
                    # Skip for now - too expensive to check properly.
                    pass
            
            if not merged:
                break
        
        return U, V, W
    
    def greedy_rank_reduction(U, V, W):
        """
        Aggressive approach: try all single-term removals in random order,
        with multiple restarts.
        """
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
        
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
            rank = U_cur.shape[1]
            
            # Random order of terms
            order = np.random.permutation(rank)
            
            for idx in order:
                if time.time() > deadline - 5:
                    break
                r = idx
                U_save = U_cur[:, r].copy()
                V_save = V_cur[:, r].copy()
                W_save = W_cur[:, r].copy()
                
                U_cur[:, r] = 0
                V_cur[:, r] = 0
                W_cur[:, r] = 0
                
                if verify(U_cur, V_cur, W_cur):
                    U_cur = np.delete(U_cur, r, axis=1)
                    V_cur = np.delete(V_cur, r, axis=1)
                    W_cur = np.delete(W_cur, r, axis=1)
                    rank -= 1
                    order = np.random.permutation(rank)  # reshuffle
                else:
                    U_cur[:, r] = U_save
                    V_cur[:, r] = V_save
                    W_cur[:, r] = W_save
            
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U_cur, V_cur, W_cur
        
        return best_U, best_V, best_W
    
    def build_schoolbook():
        """Fallback schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((nU, rank), dtype=float)
        V = np.zeros((nV, rank), dtype=float)
        W = np.zeros((nW, rank), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        return U, V, W
    
    # Build initial rank-50 decomposition
    U, V, W = build_rank50()
    
    if not verify(U, V, W):
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Phase 1: Simple term elimination
    if time.time() < deadline - 30:
        U, V, W = try_term_elimination(U, V, W)
    
    # Phase 2: Greedy reduction with random restarts
    if time.time() < deadline - 10:
        U, V, W = greedy_rank_reduction(U, V, W)
    
    # Final verification
    if not verify(U, V, W):
        # Fallback
        U, V, W = build_schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
