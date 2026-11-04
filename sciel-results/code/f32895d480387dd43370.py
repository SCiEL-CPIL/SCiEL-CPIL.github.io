"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 72)
    def schoolbook():
        U = np.zeros((nm, nm*mp//p))  # Actually rank = n*m*p = 72
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n*m*p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 140  # 140s limit, 10s safety margin
    
    def verify(U, V, W, tol=1e-8):
        T_hat = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.max(np.abs(T - T_hat)) < tol
    
    # Try rank 71
    target_rank = 71
    
    # Strategy: The rank-71 decomposition exists if and only if one of the 72 
    # schoolbook terms is a linear combination of the other 71.
    # We search for coefficients c_1,...,c_71 and a term t such that:
    # e_{t} = sum_{r != t} c_r * e_{r} (in the tensor sense)
    # This means: for all (alpha, beta, gamma):
    # T[t][alpha,beta,gamma] = sum_{r != t} c_r * T[r][alpha,beta,gamma]
    
    # Actually, more precisely: we want to find U, V, W of rank 71 such that
    # sum_r u_r ⊗ v_r ⊗ w_r = T
    # Let's parameterize: take 71 of the schoolbook terms and add one "correction" term.
    # Or: modify all 72 terms slightly.
    
    # Better approach: Use the fact that if rank 71 exists, then there's a non-trivial
    # linear relation among the 72 rank-1 tensors. Let's search for it.
    
    # The 72 rank-1 tensors are T_{ijk} = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # We want to find c_{ijk} not all zero such that sum c_{ijk} T_{ijk} = 0
    # i.e., sum_{i,j,k} c_{ijk} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} = 0
    
    # This is equivalent to: for all (alpha, beta, gamma),
    # sum_{i,j,k: i*m+j=alpha, j*p+k=beta, k*n+i=gamma} c_{ijk} = 0
    
    # Let's set up the linear system. The tensor has nm*mp*pn = 12*24*18 = 5184 entries.
    # But most are 0 or 1. The constraint is on the 72 positions where T=1.
    
    # Actually, the condition for linear dependency is:
    # There exist c_{ijk} such that sum_{i,j,k} c_{ijk} T_{ijk} = 0 as a tensor.
    # T_{ijk} has a 1 at position (i*m+j, j*p+k, k*n+i).
    # So the sum is zero iff for each position (a,b,c) in the tensor grid,
    # sum of c_{ijk} over all (i,j,k) with i*m+j=a, j*p+k=b, k*n+i=c equals 0.
    
    # Let's build this linear system.
    # Variables: c_{ijk} for i in 0..2, j in 0..3, k in 0..5 (72 variables)
    # Constraints: for each (a,b,c) where at least one (i,j,k) maps to it,
    #   sum c_{ijk} = 0
    
    # Build the matrix A where A[pos, (i,j,k)] = 1 if (i,j,k) maps to pos
    # pos = (a,b,c) -> index = a*mp*pn + b*pn + c
    
    mp_pn = mp * pn
    A = np.zeros((nm * mp_pn, n*m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                pos = a * mp_pn + b * pn + c
                col = i*m*p + j*p + k
                A[pos, col] = 1
    
    # Find null space of A
    # A is 5184 x 72. We want a non-zero vector c in null(A).
    # This requires rank(A) < 72.
    
    # Compute SVD or use null_space
    try:
        from scipy.linalg import null_space
        # Use economy SVD
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Null space is spanned by rows of Vt corresponding to zero singular values
        tol_svd = max(A.shape) * np.amax(s) * np.finfo(s.dtype).eps
        null_idx = np.where(s < tol_svd)[0]
        
        if len(null_idx) > 0:
            # There's a non-trivial null space
            # The null space vector gives coefficients c_{ijk}
            # We can use this to construct a rank-71 decomposition
            
            # The null vector c means: sum c_{ijk} T_{ijk} = 0
            # So T = sum T_{ijk} = sum T_{ijk} - sum c_{ijk} T_{ijk} + sum c_{ijk} T_{ijk}
            # This doesn't directly help. Let me think differently.
            
            # Actually, if sum c_{ijk} T_{ijk} = 0, then the 72 tensors are linearly dependent.
            # This means we can express one as a combination of the others, giving rank 71.
            
            # But we need the actual decomposition. Let's use the null vector to find it.
            
            # If c is in null(A), then sum c_{ijk} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} = 0
            # This means the tensor sum c_{ijk} T_{ijk} = 0.
            # We can write: T_{i0j0k0} = -sum_{(i,j,k) != (i0,j0,k0)} (c_{ijk}/c_{i0j0k0}) T_{ijk}
            # for any (i0,j0,k0) with c_{i0j0k0} != 0.
            
            # So we can merge that term into the others, giving a rank-71 decomposition.
            
            # Find a term with non-zero coefficient
            c_vec = Vt[null_idx[0], :]  # First null vector
            nonzero = np.where(np.abs(c_vec) > 1e-10)[0]
            
            if len(nonzero) >= 2:
                # Pick a term to eliminate
                idx_elim = nonzero[0]
                # Convert idx_elim back to (i,j,k)
                i0 = idx_elim // (m*p)
                j0 = (idx_elim % (m*p)) // p
                k0 = idx_elim % p
                
                # The decomposition: for each of the other 71 terms, keep it,
                # but scale the ones that are "absorbed" appropriately.
                
                # Actually, the rank-71 decomposition is:
                # T = sum_{r=1}^{71} u_r ⊗ v_r ⊗ w_r
                # where we take 71 of the schoolbook terms and adjust them.
                
                # Let's construct it explicitly.
                # We have: T_{i0j0k0} = -sum_{(i,j,k) != (i0,j0,k0)} (c_{ijk}/c_{i0j0k0}) T_{ijk}
                
                # So: T = sum_{all (i,j,k)} T_{ijk} 
                #     = sum_{(i,j,k) != (i0,j0,k0)} T_{ijk} + T_{i0j0k0}
                #     = sum_{(i,j,k) != (i0,j0,k0)} T_{ijk} - sum_{(i,j,k) != (i0,j0,k0)} (c_{ijk}/c_{i0j0k0}) T_{ijk}
                #     = sum_{(i,j,k) != (i0,j0,k0)} (1 - c_{ijk}/c_{i0j0k0}) T_{ijk}
                
                # Wait, that's not right either. Let me be more careful.
                
                # T = sum_{i,j,k} T_{ijk}
                # And sum_{i,j,k} c_{ijk} T_{ijk} = 0, so sum_{i,j,k} T_{ijk} = sum_{i,j,k} T_{ijk} + 0
                # = sum_{i,j,k} T_{ijk} + sum_{i,j,k} c_{ijk} T_{ijk}
                # = sum_{i,j,k} (1 + c_{ijk}) T_{ijk}  -- no, this doesn't help.
                
                # Let me think again. We want to write T as a sum of 71 rank-1 tensors.
                # Since the 72 tensors are linearly dependent, we can write one as a combination of the rest.
                # Let's say T_{i0j0k0} = sum_{r in R} d_r T_r where R is the set of other 71 indices.
                # Then T = sum_{r in R} T_r + T_{i0j0k0} = sum_{r in R} (1 + d_r) T_r
                # But this is still 71 terms only if all (1+d_r) are non-zero, which they might not be.
                
                # Actually, the issue is that T_{i0j0k0} = -sum_{r != i0j0k0} (c_r / c_{i0j0k0}) T_r
                # So T = sum_{r != i0j0k0} T_r + T_{i0j0k0}
                #     = sum_{r != i0j0k0} T_r - sum_{r != i0j0k0} (c_r / c_{i0j0k0}) T_r
                #     = sum_{r != i0j0k0} (1 - c_r / c_{i0j0k0}) T_r
                
                # This gives 71 terms, but the coefficients might be 0 for some r,
                # which is fine (we just skip those terms).
                
                # Let's construct this.
                U_new = np.zeros((nm, target_rank))
                V_new = np.zeros((mp, target_rank))
                W_new = np.zeros((pn, target_rank))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i*m*p + j*p + k
                            if idx == idx_elim:
                                continue
                            coeff = 1 - c_vec[idx] / c_vec[idx_elim]
                            if abs(coeff) > 1e-10:
                                U_new[i*m+j, col] = coeff
                                V_new[j*p+k, col] = 1
                                W_new[k*n+i, col] = 1
                                col += 1
                
                # We might have fewer than 71 non-zero terms, pad with zeros
                if col < target_rank:
                    # Pad with zero columns (which don't contribute)
                    pass
                
                U_use = U_new[:, :col]
                V_use = V_new[:, :col]
                W_use = W_new[:, :col]
                
                if verify(U_use, V_use, W_use):
                    # Success! Convert to half-integers if needed
                    # The coefficients should be rational, let's check
                    best_rank = col
                    best_U = U_use
                    best_V = V_use
                    best_W = W_use
                    
                    # Try to round to half-integers
                    U_round = np.round(U_use * 2) / 2
                    V_round = np.round(V_use * 2) / 2
                    W_round = np.round(W_use * 2) / 2
                    if verify(U_round, V_round, W_round):
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                    break
                    
        # If we found a solution, break
        if best_rank < n*m*p:
            pass
    except Exception as e:
        pass
    
    # If the linear algebra approach didn't work, try numerical optimization
    if best_rank == n*m*p and time.time() < deadline:
        # Try a few random restarts with better initialization
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initialize from schoolbook, drop one random column, perturb
            perm = np.random.permutation(n*m*p)
            keep = perm[:target_rank]
            
            U_init = U_sb[:, keep].copy()
            V_init = V_sb[:, keep].copy()
            W_init = W_sb[:, keep].copy()
            
            # Perturb
            scale = 0.5
            U_init += np.random.randn(nm, target_rank) * scale
            V_init += np.random.randn(mp, target_rank) * scale
            W_init += np.random.randn(pn, target_rank) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U_f = x[:target_rank*nm].reshape(nm, target_rank)
                V_f = x[target_rank*nm:2*target_rank*nm].reshape(mp, target_rank)
                W_f = x[2*target_rank*nm:].reshape(pn, target_rank)
                T_hat = np.einsum('ar,br,cr->abc', U_f, V_f, W_f, optimize=True)
                R = T - T_hat
                return np.sum(R*R)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                       'maxls': 50})
                x_opt = res.x
                U_opt = x_opt[:target_rank*nm].reshape(nm, target_rank)
                V_opt = x_opt[target_rank*nm:2*target_rank*nm].reshape(mp, target_rank)
                W_opt = x_opt[2*target_rank*nm:].reshape(pn, target_rank)
                
                if verify(U_opt, V_opt, W_opt, tol=1e-6):
                    best_rank = target_rank
                    best_U = U_opt
                    best_V = V_opt
                    best_W = W_opt
                    
                    # Try rounding
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    if verify(U_round, V_round, W_round, tol=1e-6):
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                    break
            except Exception:
                pass
    
    # Ensure correct shapes
    if best_U.shape[1] != best_rank:
        best_U = best_U[:, :best_rank]
    if best_V.shape[1] != best_rank:
        best_V = best_V[:, :best_rank]
    if best_W.shape[1] != best_rank:
        best_W = best_W[:, :best_rank]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
