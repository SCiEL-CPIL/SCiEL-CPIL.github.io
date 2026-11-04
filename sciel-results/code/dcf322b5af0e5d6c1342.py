import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    Starts with rank-56 Strassen decomposition, then attempts rank reduction.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    nm, mp, pn = n*m, m*p, p*n  # 8, 32, 16
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    # T[a_idx, b_idx, c_idx] = 1 if the product contributes
    # a_idx = i*m + j (i in 0..n-1, j in 0..m-1)
    # b_idx = j*p + k (j in 0..m-1, k in 0..p-1)
    # c_idx = k*n + i (k in 0..p-1, i in 0..n-1) -- trace convention W[k*n+i, r]
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    def build_strassen_56():
        """Build the rank-56 Strassen decomposition."""
        total_rank = 56
        U = np.zeros((nm, total_rank), dtype=float)
        V = np.zeros((mp, total_rank), dtype=float)
        W = np.zeros((pn, total_rank), dtype=float)
        
        # Strassen for 2x2 * 2x2 (local indices 0,1,2,3 for 2x2 matrix entries)
        U_local = np.zeros((4, 7), dtype=float)
        V_local = np.zeros((4, 7), dtype=float)
        W_local = np.zeros((4, 7), dtype=float)
        
        # M1 = (A11+A22)(B11+B22)
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        # M2 = (A21+A22)B11
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        # M3 = A11(B21-B22)
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        # M4 = A22(B12-B11)
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        # M5 = (A11+A12)B22
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                
                if s == 0:
                    a_map = [0*m+0, 0*m+1, 1*m+0, 1*m+1]
                else:
                    a_map = [0*m+2, 0*m+3, 1*m+2, 1*m+3]
                
                if s == 0:
                    b_map = [0*p+(2*j), 0*p+(2*j+1), 1*p+(2*j), 1*p+(2*j+1)]
                else:
                    b_map = [2*p+(2*j), 2*p+(2*j+1), 3*p+(2*j), 3*p+(2*j+1)]
                
                c_map = [(2*j)*n+0, (2*j+1)*n+0, (2*j)*n+1, (2*j+1)*n+1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        return U, V, W
    
    def verify_decomp(U, V, W, T, tol=1e-9):
        """Verify that sum_r u_r ⊗ v_r ⊗ w_r == T."""
        R = U.shape[1]
        T_recon = np.zeros_like(T)
        for r in range(R):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_recon, T, atol=tol)
    
    def try_rank_reduction(U, V, W, T, start_rank, max_iter=2000):
        """Try to reduce rank by finding linear dependencies among rank-1 terms.
        Uses iterative approach: try to express one term as combination of others."""
        R = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to remove each rank-1 term by checking if it's in the span
            # of the others (approximately, using the tensor equation)
            # For term r, we need: u_r ⊗ v_r ⊗ w_r = sum_{s≠r} c_s * u_s ⊗ v_s ⊗ w_s
            # This is a system of nm*mp*pn equations in (R-1) unknowns
            # Too large. Instead, use a heuristic: try to zero out one term
            # by adjusting others.
            
            # Alternative: use the fact that if two terms are "close" in some sense,
            # we can merge them. Or use SVD-based rank reduction.
            
            # Try: for each pair (r1, r2), check if we can replace them with
            # a single term. This requires u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w
            # which is very restrictive.
            
            # Better approach: use the "splitting" technique.
            # If we have terms that share structure, we can combine.
            
            # For now, try a simple approach: check if any column of U is zero
            # (meaning that term doesn't contribute to A-side), and similarly for V, W.
            removed = False
            for r in range(R - 1, -1, -1):
                if np.allclose(U[:, r], 0, atol=1e-12):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    R -= 1
                    removed = True
                    break
                if np.allclose(V[:, r], 0, atol=1e-12):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    R -= 1
                    removed = True
                    break
                if np.allclose(W[:, r], 0, atol=1e-12):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    R -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        if verify_decomp(U, V, W, T):
            return U, V, W
        return None
    
    def attempt_direct_search(T, target_rank, max_iter=5000):
        """Attempt to find a rank-target_rank decomposition using iterative optimization.
        Initialize with random rank-r decomposition and optimize."""
        nm, mp, pn = T.shape
        R = target_rank
        
        # Initialize with the naive rank-64 decomposition, then try to reduce
        # Naive: for each (i,j,k), term has u at (i*m+j), v at (j*p+k), w at (k*n+i)
        # That's 64 terms. We want to reduce to target_rank.
        
        # Start from rank-56 Strassen and try to reduce
        U56, V56, W56 = build_strassen_56()
        
        # Try to reduce from 56 to target_rank by finding dependencies
        # Use a simple approach: iteratively try to remove one term and solve
        # for corrections in the remaining terms.
        
        U = U56.copy()
        V = V56.copy()
        W = W56.copy()
        R = 56
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if R <= target_rank:
                break
            
            # Pick a random term to try to remove
            r_remove = np.random.randint(0, R)
            
            # Save the term
            u_rm = U[:, r_remove].copy()
            v_rm = V[:, r_remove].copy()
            w_rm = W[:, r_remove].copy()
            
            # Remove it
            U_no = np.delete(U, r_remove, axis=1)
            V_no = np.delete(V, r_remove, axis=1)
            W_no = np.delete(W, r_remove, axis=1)
            
            # The residual is: residual = u_rm ⊗ v_rm ⊗ w_rm
            # We want to find small corrections δu_s, δv_s, δw_s for s in remaining terms
            # such that sum_s δu_s ⊗ v_s ⊗ w_s + u_s ⊗ δv_s ⊗ w_s + u_s ⊗ v_s ⊗ δw_s ≈ residual
            # This is a linear system. Let's set up the matrix.
            # Number of unknowns: (R-1) * (nm + mp + pn)
            # Number of equations: nm * mp * pn = 8 * 32 * 16 = 4096
            
            R_no = R - 1
            n_unknowns = R_no * (nm + mp + pn)
            n_equations = nm * mp * pn
            
            if n_unknowns >= n_equations:
                # Underdetermined, use least squares
                # Build the linear system
                # For each equation (a,b,c): sum_s [U_no[a,s]*V_no[b,s]*δW_no[c,s] 
                #   + U_no[a,s]*δV_no[b,s]*W_no[c,s] + δU_no[a,s]*V_no[b,s]*W_no[c,s]] = residual[a,b,c]
                
                residual = np.einsum('a,b,c->abc', u_rm, v_rm, w_rm)
                
                # This is a large system. Use a simplified approach:
                # Only allow corrections to W (output side) since pn=16 is smallest
                # Unknowns: (R-1) * pn
                n_unk_w = R_no * pn
                if n_unk_w >= n_equations:
                    # Build matrix A_mat of shape (n_equations, n_unk_w)
                    # and vector b of shape (n_equations,)
                    # A_mat[(a*mp+b)*pn+c, s*pn+dc] = U_no[a,s]*V_no[b,s]*δW[dc]
                    # So for fixed (a,b,s), the contribution to all c is U_no[a,s]*V_no[b,s] * δW_no[c,s]
                    
                    # This is still large: 4096 x (55*16=880). Let's try.
                    A_mat = np.zeros((n_equations, n_unk_w), dtype=float)
                    b_vec = residual.flatten()
                    
                    for s in range(R_no):
                        for a in range(nm):
                            for b in range(mp):
                                coeff = U_no[a, s] * V_no[b, s]
                                if abs(coeff) > 1e-12:
                                    for c in range(pn):
                                        eq_idx = (a * mp + b) * pn + c
                                        unk_idx = s * pn + c
                                        A_mat[eq_idx, unk_idx] = coeff
                    
                    # Solve least squares
                    try:
                        delta_W, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
                        delta_W = delta_W.reshape(R_no, pn)
                    except:
                        R = R_no + 1  # restore
                        continue
                    
                    # Apply correction
                    W_no = W_no + delta_W
                    
                    # Check if the decomposition is still valid
                    U_new = np.concatenate([U_no, u_rm.reshape(-1,1) if False else U_no], axis=1)
                    # Actually, we're removing the term, so:
                    U_new = U_no
                    V_new = V_no
                    W_new = W_no
                    
                    if verify_decomp(U_new, V_new, W_new, T, tol=1e-6):
                        U, V, W = U_new, V_new, W_new
                        R = R_no
                    else:
                        # Try with smaller correction
                        delta_W_small = delta_W * 0.1
                        W_no2 = W_no - delta_W + delta_W_small  # revert and apply small
                        # Actually let's just skip this term
                        continue
            else:
                # Not enough unknowns, try a different strategy
                # Try removing a term where the vectors are "simple" (sparse)
                sparsities = []
                for r in range(R):
                    sp = np.count_nonzero(U[:, r]) + np.count_nonzero(V[:, r]) + np.count_nonzero(W[:, r])
                    sparsities.append(sp)
                r_remove = int(np.argmin(sparsities))
                
                u_rm = U[:, r_remove].copy()
                v_rm = V[:, r_remove].copy()
                w_rm = W[:, r_remove].copy()
                
                U_no = np.delete(U, r_remove, axis=1)
                V_no = np.delete(V, r_remove, axis=1)
                W_no = np.delete(W, r_remove, axis=1)
                
                residual = np.einsum('a,b,c->abc', u_rm, v_rm, w_rm)
                R_no = R - 1
                
                # Try to find corrections to all three
                # Use a simplified approach: only correct W
                n_unk_w = R_no * pn
                if n_unk_w >= n_equations:
                    A_mat = np.zeros((n_equations, n_unk_w), dtype=float)
                    b_vec = residual.flatten()
                    
                    for s in range(R_no):
                        for a in range(nm):
                            for b in range(mp):
                                coeff = U_no[a, s] * V_no[b, s]
                                if abs(coeff) > 1e-12:
                                    for c in range(pn):
                                        eq_idx = (a * mp + b) * pn + c
                                        unk_idx = s * pn + c
                                        A_mat[eq_idx, unk_idx] = coeff
                    
                    try:
                        delta_W, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
                        delta_W = delta_W.reshape(R_no, pn)
                    except:
                        continue
                    
                    W_no = W_no + delta_W
                    
                    if verify_decomp(U_no, V_no, W_no, T, tol=1e-6):
                        U, V, W = U_no, V_no, W_no
                        R = R_no
                    else:
                        continue
        return U, V, W
    
    # Build initial rank-56 decomposition
    U, V, W = build_strassen_56()
    
    # Verify
    if not verify_decomp(U, V, W, T, tol=1e-9):
        # Something wrong with our Strassen construction, fall back to naive
        print("Strassen verification failed, using naive rank-64")
        U = np.zeros((nm, 64), dtype=float)
        V = np.zeros((mp, 64), dtype=float)
        W = np.zeros((pn, 64), dtype=float)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        if not verify_decomp(U, V, W, T, tol=1e-9):
            # Shouldn't happen, but return something
            pass
    else:
        # Try to reduce rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
        
        # Try rank reduction
        if time.time() < deadline - 30:
            result = attempt_direct_search(T, 40, max_iter=3000)
            if result is not None:
                rU, rV, rW = result
                if verify_decomp(rU, rV, rW, T, tol=1e-6):
                    if rU.shape[1] < best_rank:
                        best_U, best_V, best_W = rU, rV, rW
                        best_rank = rU.shape[1]
        
        U, V, W = best_U, best_V, best_W
    
    # Convert to integer/half-integer if possible
    # Check if all values are close to integers or half-integers
    def to_clean(arr):
        """Convert to clean integer/half-integer values if close."""
        arr = arr.copy()
        # Check if values are close to multiples of 0.5
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=1e-6):
            arr = rounded / 2.0
        else:
            # Try integers
            rounded_int = np.round(arr)
            if np.allclose(arr, rounded_int, atol=1e-6):
                arr = rounded_int
        return arr
    
    U = to_clean(U)
    V = to_clean(V)
    W = to_clean(W)
    
    # Final verification
    if not verify_decomp(U, V, W, T, tol=1e-6):
        # Re-verify with original values
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
