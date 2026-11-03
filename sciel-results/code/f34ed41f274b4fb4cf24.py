"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 135  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook decomposition
    def schoolbook():
        rank = nm * mp // m  # = n*m*p = 105
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    # Fast verification using vectorized operations
    def verify(U, V, W):
        """Verify that sum_c U[:,c] * V[:,c] * W[:,c] matches the matmul tensor."""
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        # Vectorized: build the full tensor check
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We can compute this as: for each c, outer product contribution
        # Faster: use einsum-like approach with broadcasting
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape U to (n,m,R), V to (m,p,R), W to (p,n,R)
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # Compute via: for each pair (i,j), sum_c Ur[i,j,c]*Wr[:,i,c] gives vector over k weighted by Vr
        # Actually: T[i,j,k] = sum_c Ur[i,j,c] * (Vr[j,k,c] * Wr[k,i,c])
        # Let's do: M[j,k] for fixed i = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # This is like: for each i, compute B[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # Use: B[i,j,k] = sum_c (Ur[i,j,c] * Wr[k,i,c]) * Vr[j,k,c]
        # Define A[i,j,k] = Ur[i,j,c] * Wr[k,i,c] summed over c? No, c is the sum index.
        # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # = sum_c (Ur[i,j,c] * Wr[k,i,c]) * Vr[j,k,c]  -- but k is also an index here, not summed
        # Let me just do it with a loop over c for clarity and speed with numpy:
        # Actually the fastest is:
        # T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, 1.0, atol=1e-10)

    # Strassen block for 2x2x2
    def strassen_block_terms(i0, i1, j0, j1, k0, k1):
        """Return 7 (u, v, w) vectors for a 2x2x2 Strassen block."""
        terms = []
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i1*m+j1] = 1.0
        v[j0*p+k0] = 1.0
        w[k0*n+i1] = 1.0; w[k1*n+i1] = -1.0
        terms.append((u, v, w))
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0
        v[j0*p+k1] = 1.0; v[j1*p+k1] = -1.0
        w[k1*n+i0] = 1.0; w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j1] = 1.0
        v[j1*p+k0] = 1.0; v[j0*p+k0] = -1.0
        w[k0*n+i0] = 1.0; w[k0*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j0] = 1.0; u[i0*m+j1] = 1.0
        v[j1*p+k1] = 1.0
        w[k0*n+i0] = -1.0; w[k1*n+i0] = 1.0
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i1*m+j0] = 1.0; u[i0*m+j0] = -1.0
        v[j0*p+k0] = 1.0; v[j0*p+k1] = 1.0
        w[k1*n+i1] = 1.0
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
        u[i0*m+j1] = 1.0; u[i1*m+j1] = -1.0
        v[j1*p+k0] = 1.0; v[j1*p+k1] = 1.0
        w[k0*n+i0] = 1.0
        terms.append((u, v, w))
        
        return terms

    # Define candidate block configurations
    # Each block: (I_set, J_set, K_set) where each is a 2-element set
    # Must be disjoint in the sense that each (i,j,k) is in at most one block
    candidate_blocks = [
        # Original 6 blocks
        [({0,1},{0,1},{0,1}), ({1,2},{0,1},{2,3}), ({0,2},{0,1},{4,5}),
         ({0,1},{2,3},{0,1}), ({1,2},{2,3},{2,3}), ({0,2},{2,3},{4,5})],
        # Alternative: try different J partitions
        [({0,1},{0,1},{0,1}), ({1,2},{2,3},{0,1}), ({0,2},{3,4},{0,1}),
         ({0,1},{2,3},{2,3}), ({1,2},{3,4},{2,3}), ({0,2},{3,4},{4,5})],
        # Another alternative
        [({0,1},{0,2},{0,1}), ({1,2},{2,3},{0,1}), ({0,2},{3,4},{0,1}),
         ({0,1},{1,3},{2,3}), ({1,2},{2,4},{2,3}), ({0,2},{3,4},{4,5})],
    ]

    def is_valid_block_set(blocks):
        """Check that no (i,j,k) appears in more than one block."""
        seen = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        if (i, j, k) in seen:
                            return False
                        seen.add((i, j, k))
        return True

    def build_decomposition(blocks):
        """Build U, V, W from a set of Strassen blocks + schoolbook for the rest."""
        U_list, V_list, W_list = [], [], []
        
        for I, J, K in blocks:
            i0, i1 = sorted(I)
            j0, j1 = sorted(J)
            k0, k1 = sorted(K)
            terms = strassen_block_terms(i0, i1, j0, j1, k0, k1)
            for u, v, w in terms:
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
        
        # Add schoolbook terms for uncovered (i,j,k)
        covered = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i*m+j] = 1.0
                        v[j*p+k] = 1.0
                        w[k*n+i] = 1.0
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
        
        U = np.array(U_list).T  # (nm, rank)
        V = np.array(V_list).T  # (mp, rank)
        W = np.array(W_list).T  # (pn, rank)
        return U, V, W

    # Try each block configuration
    best_U, best_V, best_W = None, None, None
    best_rank = rank_sb
    
    for blocks in candidate_blocks:
        if not is_valid_block_set(blocks):
            continue
        U, V, W = build_decomposition(blocks)
        if verify(U, V, W):
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U, V, W
    
    # If we found a valid solution, try to remove redundant terms
    if best_U is not None:
        U, V, W = best_U, best_V, best_W
        R = U.shape[1]
        
        # Fast removal: for each column c, check if removing it still gives valid result
        # Precompute: T_full[i,j,k] = sum_c Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
        # If we remove column c, T_new[i,j,k] = T_full[i,j,k] - Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
        # We need T_new = 1 for all (i,j,k)
        # So we need: Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c] = 0 for all (i,j,k) where T_full != 1
        # But T_full should be exactly 1 if the decomposition is valid.
        # Actually, for a valid decomposition, T_full[i,j,k] = 1 for all (i,j,k).
        # Removing column c: T_new = T_full - contribution_c
        # We need T_new = 1, so contribution_c must be 0 everywhere.
        # contribution_c[i,j,k] = Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # This is zero everywhere iff for every (i,j,k), at least one of Ur[i,j,c], Vr[j,k,c], Wr[k,i,c] is 0.
        
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        
        removed_any = True
        while removed_any and time.time() < deadline - 15:
            removed_any = False
            for c in range(R):
                if time.time() >= deadline - 15:
                    break
                # Check if column c contributes nothing
                # contribution[i,j,k] = Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
                # This is zero for all (i,j,k) iff:
                # For all (i,j,k): Ur[i,j,c]==0 or Vr[j,k,c]==0 or Wr[k,i,c]==0
                # Equivalently: the set of (i,j) with Ur[i,j,c]!=0, 
                # the set of (j,k) with Vr[j,k,c]!=0,
                # the set of (k,i) with Wr[k,i,c]!=0
                # must have no common (i,j,k)
                
                # Get support sets
                ij_support = set()
                for i in range(n):
                    for j in range(m):
                        if Ur[i,j,c] != 0:
                            ij_support.add((i,j))
                
                if not ij_support:
                    # Column c of U is zero, so it contributes nothing
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
                
                jk_support = set()
                for j in range(m):
                    for k in range(p):
                        if Vr[j,k,c] != 0:
                            jk_support.add((j,k))
                
                if not jk_support:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
                
                ki_support = set()
                for k in range(p):
                    for i in range(n):
                        if Wr[k,i,c] != 0:
                            ki_support.add((k,i))
                
                if not ki_support:
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
                
                # Check if there exists (i,j,k) in all three supports
                found_common = False
                for (i,j) in ij_support:
                    for (j2,k) in jk_support:
                        if j == j2:
                            for (k2,i2) in ki_support:
                                if k == k2 and i == i2:
                                    found_common = True
                                    break
                            if found_common:
                                break
                    if found_common:
                        break
                if not found_common:
                    # No common (i,j,k), so this column contributes nothing
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed_any = True
                    break
        
        # Final verification
        if verify(U, V, W):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
