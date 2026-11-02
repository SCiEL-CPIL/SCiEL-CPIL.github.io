"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin (150s limit)
    
    n, m, p = DIMENSIONS  # 2, 2, 6
    U_dim = n * m  # 4
    V_dim = m * p  # 12
    W_dim = p * n  # 12
    
    # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def is_half_integer(arr, tol=1e-8):
        """Check if all entries are multiples of 0.5"""
        doubled = arr * 2
        return np.allclose(doubled, np.round(doubled), atol=tol)
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == target"""
        rank = U.shape[1]
        T = np.zeros((U_dim, V_dim, W_dim))
        for r in range(rank):
            T += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        return np.linalg.norm(T - target) < tol
    
    def try_snap_and_verify(U, V, W, tol=1e-6):
        """Snap to half-integers and verify"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        T = np.zeros((U_dim, V_dim, W_dim))
        for r in range(U_snap.shape[1]):
            T += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
        if np.linalg.norm(T - target) < tol:
            return U_snap, V_snap, W_snap
        return None
    
    def strassen_2x2x2():
        """
        Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        U: (4,7), V: (4,7), W: (4,7)
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        # 7 products in Strassen's algorithm
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # U entries (A matrix coefficients)
        # M1: A00+A11 -> U[0]=1, U[3]=1
        U[0, 0] = 1; U[3, 0] = 1
        # M2: A10+A11 -> U[2]=1, U[3]=1
        U[2, 1] = 1; U[3, 1] = 1
        # M3: A00 -> U[0]=1
        U[0, 2] = 1
        # M4: A11 -> U[3]=1
        U[3, 3] = 1
        # M5: A00+A01 -> U[0]=1, U[1]=1
        U[0, 4] = 1; U[1, 4] = 1
        # M6: A10-A00 -> U[2]=1, U[0]=-1
        U[2, 5] = 1; U[0, 5] = -1
        # M7: A01-A11 -> U[1]=1, U[3]=-1
        U[1, 6] = 1; U[3, 6] = -1
        
        # V entries (B matrix coefficients)
        # M1: B00+B11 -> V[0]=1, V[3]=1
        V[0, 0] = 1; V[3, 0] = 1
        # M2: B00 -> V[0]=1
        V[0, 1] = 1
        # M3: B01-B11 -> V[1]=1, V[3]=-1
        V[1, 2] = 1; V[3, 2] = -1
        # M4: B10-B00 -> V[2]=1, V[0]=-1
        V[2, 3] = 1; V[0, 3] = -1
        # M5: B11 -> V[3]=1
        V[3, 4] = 1
        # M6: B00+B01 -> V[0]=1, V[1]=1
        V[0, 5] = 1; V[1, 5] = 1
        # M7: B10+B11 -> V[2]=1, V[3]=1
        V[2, 6] = 1; V[3, 6] = 1
        
        # W entries (C matrix coefficients)
        # C00 = M1 + M4 + M5 - M7  (W[0])
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = 1; W[0, 6] = -1
        # C01 = M3 + M5  (W[1])
        W[1, 2] = 1; W[1, 4] = 1
        # C10 = M2 + M4 + M6  (W[2])
        W[2, 1] = 1; W[2, 3] = 1; W[2, 5] = 1
        # C11 = M1 + M6  (W[3])
        W[3, 0] = 1; W[3, 5] = 1
        
        return U, V, W
    
    def build_strassen_3blocks():
        """
        Build rank-21 decomposition by applying Strassen to each of 3 independent 2x2x2 blocks.
        B is 2x6, so we split into 3 blocks of 2 columns each.
        Block r handles columns 2r, 2r+1 of B.
        """
        U_s, V_s, W_s = strassen_2x2x2()
        rank = 21
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                U[:, col] = U_s[:, k]
                # V: row index is j*p + k_local, where k_local = 2r + local_k
                for j in range(m):
                    for kk in range(2):
                        v_idx = j * p + (2*r + kk)
                        V[v_idx, col] = V_s[j*2+kk, k]
                # W: row index is k*n + i, where k_local = 2r + local_k
                for i in range(n):
                    for kk in range(2):
                        w_idx = (2*r + kk) * n + i
                        W[w_idx, col] = W_s[kk*2+i, k]
        
        return U, V, W
    
    def check_rank1_residual(residual):
        """
        Check if residual tensor is rank-1. If so, return (u, v, w) vectors.
        Otherwise return None.
        """
        # Mode-0 unfolding: (U_dim) x (V_dim * W_dim)
        R0 = residual.reshape(U_dim, -1)
        s0 = np.linalg.svd(R0, compute_uv=False)
        if len(s0) > 1 and s0[1] > 1e-7 * max(s0[0], 1e-15):
            return None
        
        # Mode-1 unfolding
        R1 = residual.transpose(1, 0, 2).reshape(V_dim, -1)
        s1 = np.linalg.svd(R1, compute_uv=False)
        if len(s1) > 1 and s1[1] > 1e-7 * max(s1[0], 1e-15):
            return None
        
        # Mode-2 unfolding
        R2 = residual.transpose(2, 0, 1).reshape(W_dim, -1)
        s2 = np.linalg.svd(R2, compute_uv=False)
        if len(s2) > 1 and s2[1] > 1e-7 * max(s2[0], 1e-15):
            return None
        
        # Extract vectors
        U_svd, s, Vt = np.linalg.svd(R0, full_matrices=False)
        if s[0] < 1e-15:
            return None
        u_vec = U_svd[:, 0] * np.sqrt(s[0])
        
        u_max = np.argmax(np.abs(u_vec))
        if abs(u_vec[u_max]) < 1e-12:
            return None
        v_w_vec = R0[u_max, :] / u_vec[u_max]
        
        M = v_w_vec.reshape(V_dim, W_dim)
        sM = np.linalg.svd(M, compute_uv=False)
        if len(sM) > 1 and sM[1] > 1e-7 * max(sM[0], 1e-15):
            return None
        
        if sM[0] < 1e-15:
            return None
        UM, sM_full, VM = np.linalg.svd(M, full_matrices=False)
        v_vec = UM[:, 0] * np.sqrt(sM_full[0])
        w_vec = VM[0, :] * np.sqrt(sM_full[0])
        
        # Verify
        if np.linalg.norm(np.outer(v_vec, w_vec) - M) > 1e-5:
            return None
        
        return u_vec, v_vec, w_vec
    
    def try_elimination(U_base, V_base, W_base, terms_to_remove, total_sum, rank1_tensors):
        """
        Try to replace a set of terms with a single rank-1 term.
        Returns (U_new, V_new, W_new) if successful, else None.
        """
        rank = U_base.shape[1]
        n_remove = len(terms_to_remove)
        
        # Residual = target - (total - sum of removed terms)
        residual = target - total_sum
        for t in terms_to_remove:
            residual += rank1_tensors[t]
        
        if np.linalg.norm(residual) < 1e-12:
            # All removed terms are redundant (shouldn't happen in Strassen)
            keep = [r for r in range(rank) if r not in terms_to_remove]
            return U_base[:, keep], V_base[:, keep], W_base[:, keep]
        
        # Check if residual is rank-1
        result = check_rank1_residual(residual)
        if result is None:
            return None
        
        u_vec, v_vec, w_vec = result
        keep = [r for r in range(rank) if r not in terms_to_remove]
        new_rank = len(keep) + 1
        
        U_new = np.zeros((U_dim, new_rank))
        V_new = np.zeros((V_dim, new_rank))
        W_new = np.zeros((W_dim, new_rank))
        
        for idx, r in enumerate(keep):
            U_new[:, idx] = U_base[:, r]
            V_new[:, idx] = V_base[:, r]
            W_new[:, idx] = W_base[:, r]
        
        U_new[:, new_rank-1] = u_vec
        V_new[:, new_rank-1] = v_vec
        W_new[:, new_rank-1] = w_vec
        
        # Verify
        if not verify_decomposition(U_new, V_new, W_new, tol=1e-6):
            return None
        
        return U_new, V_new, W_new
    
    # === Build base Strassen decomposition ===
    U_base, V_base, W_base = build_strassen_3blocks()
    
    # Verify base
    if not verify_decomposition(U_base, V_base, W_base, tol=1e-6):
        # Something is wrong with our construction, try to fix
        print("Base verification failed, attempting fix...")
        # Fallback: direct construction
        rank = U_dim * V_dim  # trivial upper bound
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        U_base, V_base, W_base = U, V, W
    
    rank21 = U_base.shape[1]
    
    # Precompute rank-1 tensors
    rank1_tensors = []
    for r in range(rank21):
        T_r = np.einsum('u,v,w->uvw', U_base[:, r], V_base[:, r], W_base[:, r])
        rank1_tensors.append(T_r)
    
    total_sum = np.zeros((U_dim, V_dim, W_dim))
    for T_r in rank1_tensors:
        total_sum += T_r
    
    best_result = None
    best_rank = rank21
    
    # Try to snap base to half-integers
    snapped = try_snap_and_verify(U_base, V_base, W_base)
    if snapped is not None:
        best_result = snapped
        best_rank = rank21
    
    # === Search for lower rank ===
    
    # Try removing 1 term (rank 20)
    if time.time() < deadline - 30:
        for t in range(rank21):
            if time.time() > deadline - 20:
                break
            result = try_elimination(U_base, V_base, W_base, [t], total_sum, rank1_tensors)
            if result is not None:
                U_new, V_new, W_new = result
                snapped = try_snap_and_verify(U_new, V_new, W_new)
                if snapped is not None:
                    best_result = snapped
                    best_rank = rank21 - 1
                    break
    
    # Try removing 2 terms (rank 19)
    if best_rank > 19 and time.time() < deadline - 15:
        from itertools import combinations
        found_19 = False
        for pair in combinations(range(rank21), 2):
            if time.time() > deadline - 10:
                break
            result = try_elimination(U_base, V_base, W_base, list(pair), total_sum, rank1_tensors)
            if result is not None:
                U_new, V_new, W_new = result
                snapped = try_snap_and_verify(U_new, V_new, W_new)
                if snapped is not None:
                    best_result = snapped
                    best_rank = rank21 - 2
                    found_19 = True
                    break
        # If not found via single replacement, try two separate rank-1 replacements
        if not found_19 and time.time() < deadline - 10:
            # Try: remove term a, replace with rank-1; then from that result remove term b, replace with rank-1
            # This is more complex; skip for now and rely on the above
            pass

    # Try removing 3 terms (rank 18) - only if we have time
    if best_rank > 18 and time.time() < deadline - 8:
        from itertools import combinations
        found_18 = False
        for triple in combinations(range(rank21), 3):
            if time.time() > deadline - 5:
                break
            result = try_elimination(U_base, V_base, W_base, list(triple), total_sum, rank1_tensors)
            if result is not None:
                U_new, V_new, W_new = result
                snapped = try_snap_and_verify(U_new, V_new, W_new)
                if snapped is not None:
                    best_result = snapped
                    best_rank = rank21 - 3
                    found_18 = True
                    break
    
    # === Algebraic approach: try to find a rank-r decomposition directly ===
    # For small r, we can try to solve the system
    # T = sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s
    # This is a system of polynomial equations. For r < 21, we need to search.
    
    # Alternative: use the fact that the tensor has special structure
    # The target tensor has T[i*m+j, j*p+k, k*n+i] = 1, which is 24 ones.
    
    # Try a structured approach: since p=6=3*2, maybe we can use a 2x2x6 specific structure
    # that's more efficient than 3 independent Strassen blocks.
    
    # Let's try: can we do 2x2x6 with rank < 21 using a different factorization?
    # Lower bound: max(U_dim, V_dim, W_dim) = 12, but that's weak.
    # Better: the tensor has 24 non-zero entries, so rank >= 24/7 ≈ 3.4 (very weak)
    # Actually, the "slice rank" gives better bounds.
    
    # For now, let's try a randomized search for lower rank
    if best_rank > 18 and time.time() < deadline - 5:
        # Try rank 18-20 with random initialization and local optimization
        # This is expensive, so only try if we have time
        pass
    
    # Return best result
    if best_result is not None:
        U_snap, V_snap, W_snap = best_result
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: return the base Strassen decomposition
    U_snap = np.round(U_base * 2) / 2
    V_snap = np.round(V_base * 2) / 2
    W_snap = np.round(W_base * 2) / 2
    
    # Final verification
    T_check = np.zeros((U_dim, V_dim, W_dim))
    for r in range(rank21):
        T_check += np.einsum('u,v,w->uvw', U_snap[:, r], V_snap[:, r], W_snap[:, r])
    
    if np.linalg.norm(T_check - target) > 1e-4:
        # Something went wrong, use the unsnapped version
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}
    
    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

# EVOLVE-BLOCK-END
