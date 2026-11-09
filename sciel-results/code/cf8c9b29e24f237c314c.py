"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p

    # Build the list of (i,j,k) triples and their indices
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                triples.append((idx_u, idx_v, idx_w))

    # Schoolbook solution as fallback
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    for r, (iu, iv, iw) in enumerate(triples):
        U_school[iu, r] = 1
        V_school[iv, r] = 1
        W_school[iw, r] = 1

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        R = U.shape[1]
        # Check all (iu, iv, iw) combinations
        # T[iu, iv, iw] = sum_r U[iu,r]*V[iv,r]*W[iw,r]
        # Should equal 1 if there exist i,j,k such that iu=i*m+j, iv=j*p+k, iw=k*n+i
        # Should equal 0 otherwise
        
        # Build the target tensor
        target = np.zeros((nm, mp, pn))
        for iu, iv, iw in triples:
            target[iu, iv, iw] = 1.0
        
        # Compute reconstruction
        # For each r, outer product U[:,r] @ V[:,r].T @ W[:,r].T ... 
        # Actually: T[iu,iv,iw] = sum_r U[iu,r]*V[iv,r]*W[iw,r]
        # This is a sum of rank-1 tensors
        recon = np.zeros((nm, mp, pn))
        for r in range(R):
            recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Check exact match (with tolerance for floating point)
        if not np.allclose(recon, target, atol=1e-10):
            return False
        return True

    def verify_fast(U, V, W):
        """Faster verification using the specific structure."""
        R = U.shape[1]
        # Only need to check:
        # 1) For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # 2) For all other (iu,iv,iw) not in the set: sum == 0
        
        # Check the 30 positive constraints
        for iu, iv, iw in triples:
            val = 0.0
            for r in range(R):
                val += U[iu, r] * V[iv, r] * W[iw, r]
            if abs(val - 1.0) > 1e-10:
                return False
        
        # Check zero constraints - sample or check all
        # Total entries: nm*mp*pn = 6*15*10 = 900
        # Positive entries: 30
        # Zero entries: 870
        # For exactness, we need to check all
        # But this is expensive. Let's use a smarter approach.
        
        # Actually, let's check all 900 entries
        for iu in range(nm):
            for iv in range(mp):
                for iw in range(pn):
                    val = 0.0
                    for r in range(R):
                        val += U[iu, r] * V[iv, r] * W[iw, r]
                    # Determine if this should be 1 or 0
                    # Find if (iu, iv, iw) corresponds to some (i,j,k)
                    found = False
                    for i in range(n):
                        if iu == i * m:  # j=0 case won't work generally
                            pass
                    # Better: check if there exists i,j,k with i*m+j==iu, j*p+k==iv, k*n+i==iw
                    valid = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i * m + j == iu and j * p + k == iv and k * n + i == iw:
                                    valid = True
                                    break
                            if valid:
                                break
                        if valid:
                            break
                    expected = 1.0 if valid else 0.0
                    if abs(val - expected) > 1e-10:
                        return False
        return True

    def try_rank_with_W(R, W_mat):
        """Given W (pn x R), try to find U and V such that the tensor is satisfied.
        This is a system of quadratic equations in U and V.
        Fix one of them and solve linearly for the other.
        
        Strategy: For each column r of W, we have contributions.
        The constraint is: for each (iu, iv, iw):
            sum_r U[iu,r] * V[iv,r] * W[iw,r] = target[iu,iv,iw]
        
        This is not linear in (U,V) jointly. But if we fix V, it's linear in U.
        Alternatively, we can use the fact that for each iw, the equation is:
            sum_r W[iw,r] * (U[:,r] * V[:,r].T) = slice_iw
        
        Let's try: fix V to be sparse (one non-zero per column), then solve for U.
        Or: use alternating minimization.
        
        Actually, a cleaner approach: the equations are linear in U if V is fixed, 
        and linear in V if U is fixed. Use alternating least squares or direct solve.
        
        For exact solution, let's try: assume V has a specific sparse structure.
        """
        if time.time() > deadline - 10:
            return None
        
        # Approach: For each possible support pattern of W, solve the linear system.
        # But W is given. Let's use alternating approach.
        
        # Initialize V randomly
        np.random.seed(42)
        V_curr = np.random.randn(mp, R) * 0.5
        
        for iteration in range(50):
            if time.time() > deadline - 10:
                return None
            
            # Solve for U given V: 
            # For each (iu, iw): sum_r U[iu,r] * (V[:,r] * W[iw,r])_iv... 
            # Actually: for each (iu, iv, iw): U[iu,:] . (V[iv,:] * W[iw,:]) = target[iu,iv,iw]
            # This is linear in U[iu,:] for each (iu, iv, iw).
            # For fixed iu: we have mp*pn equations in R unknowns (U[iu,0],...,U[iu,R-1])
            # The equation is: sum_r U[iu,r] * V[iv,r] * W[iw,r] = target[iu,iv,iw]
            # Let a_{iv,iw,r} = V[iv,r] * W[iw,r]
            # Then: U[iu,:] @ a_{iv,iw,:} = target[iu,iv,iw] for all iv,iw
            # This is a linear system: A_iu @ U[iu,:] = b_iu
            # where A_iu has shape (mp*pn, R) and b_iu has shape (mp*pn,)
            
            U_curr = np.zeros((nm, R))
            for iu in range(nm):
                # Build A_iu
                A = np.zeros((mp * pn, R))
                b = np.zeros(mp * pn)
                idx = 0
                for iv in range(mp):
                    for iw in range(pn):
                        A[idx, :] = V[iv, :] * W[iw, :]
                        # Determine target
                        # Check if (iu, iv, iw) is a valid triple
                        valid = False
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i * m + j == iu and j * p + k == iv and k * n + i == iw:
                                        valid = True
                                        break
                                    if valid:
                                        break
                                if valid:
                                    break
                        b[idx] = 1.0 if valid else 0.0
                        idx += 1
                
                # Solve least squares: A @ x = b
                # If R < mp*pn, this is overdetermined. Use lstsq.
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_curr[iu, :] = x
                except:
                    return None
            
            # Now solve for V given U
            V_new = np.zeros((mp, R))
            for iv in range(mp):
                A = np.zeros((nm * pn, R))
                b = np.zeros(nm * pn)
                idx = 0
                for iu in range(nm):
                    for iw in range(pn):
                        A[idx, :] = U[iu, :] * W[iw, :]
                        valid = False
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i * m + j == iu and j * p + k == iv and k * n + i == iw:
                                        valid = True
                                        break
                                    if valid:
                                        break
                                if valid:
                                    break
                        b[idx] = 1.0 if valid else 0.0
                        idx += 1
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[iv, :] = x
                except:
                    return None
            
            # Check convergence
            if np.allclose(V_new, V_curr, atol=1e-12):
                V_curr = V_new
                break
            V_curr = V_new
        
        # Final verification
        if verify_fast(U_curr, V_curr, W_mat):
            return (U_curr, V_curr, W_mat)
        return None

    # Try to find lower rank solutions
    best_result = None
    best_rank = rank_schoolbook

    # Strategy: Try different W patterns and solve for U, V
    # W has shape (pn, R) = (10, R)
    # For R < 30, W has 10*R entries.
    
    # Key insight: W[iw, r] only matters for iw values that appear in the triples.
    # All 10 values of iw appear (since k*n+i ranges over 0..9 for n=2, p=5).
    
    # Let's try ranks from 29 down to 25, with randomized W initializations
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 6, 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Try several random W patterns
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            W_rand = np.random.randn(pn, target_rank) * 0.5
            
            result = try_rank_with_W(target_rank, W_rand)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    # If we found a better result, use it; otherwise schoolbook
    if best_result is not None:
        U, V, W = best_result
        # Round to nearest half-integer for cleanliness (optional)
        # U = np.round(U * 2) / 2
        # V = np.round(V * 2) / 2
        # W = np.round(W * 2) / 2
        # Re-verify after rounding
        # if verify_fast(U, V, W):
        #     pass
        # else:
        #     # Don't round if it breaks exactness
        #     pass
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
