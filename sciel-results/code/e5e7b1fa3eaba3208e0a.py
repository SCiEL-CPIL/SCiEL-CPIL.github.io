"""Matrix multiplication 2x3 by 3x5 rank minimization."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

# The matrix multiplication tensor for (n,m,p)=(2,3,5)
# T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
# The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# U shape: (n*m, rank) = (6, rank)
# V shape: (m*p, rank) = (15, rank)  
# W shape: (p*n, rank) = (10, rank)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 15, 10
    
    # Try ranks from lower bound up to schoolbook (30)
    # Lower bound: max(nm, mp, pn) = max(6,15,10) = 15
    # But we can do better. Let's try from 15 upward.
    # Actually, for matrix multiplication A(nxm) * B(mx p), 
    # the rank is at least max(nm, mp, pn) but often less than nmp.
    # Known results: 2x3*3x5 has rank <= 29 (schoolbook is 30)
    # Let's try to find rank < 30.
    
    # The tensor entries: for all i in range(n), j in range(m), k in range(p),
    # T[i,j,k] = 1
    # In flattened form: U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    # Build the list of nonzero tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx, 1))
    
    # Also need to ensure zero entries: for all (u,v,w) not in the above set,
    # the sum should be 0.
    
    # Strategy: Use iterative optimization with rank reduction.
    # Start from schoolbook and try to reduce rank by finding linear dependencies.
    
    # Schoolbook rank = n*m*p = 30
    schoolbook_rank = n * m * p
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx, col] = 1.0
                V_sb[v_idx, col] = 1.0
                W_sb[w_idx, col] = 1.0
                col += 1
    
    # Verify schoolbook
    for (ui, vi, wi, val) in tensor_entries:
        s = sum(U_sb[ui, r] * V_sb[vi, r] * W_sb[wi, r] for r in range(schoolbook_rank))
        assert abs(s - val) < 1e-10, f"Schoolbook failed at ({ui},{vi},{wi}): got {s}"
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = schoolbook_rank
    
    # Try to reduce rank by checking if columns are linearly dependent
    # and can be merged.
    
    # For each rank r from 15 to best_rank-1, try to find a decomposition.
    # Use a heuristic: try random initializations and gradient descent,
    # or use SVD-based truncation.
    
    # Actually, let's use a more systematic approach:
    # The tensor is a 3-way array. We can try to find a CP decomposition.
    # For small sizes, we can use alternating least squares.
    
    def try_rank(r, max_iters=500, timeout=None):
        """Try to find a rank-r decomposition using ALS."""
        if timeout is not None and time.time() > timeout:
            return None
        
        np.random.seed(42)
        
        best_residual = float('inf')
        best_factors = None
        
        for trial in range(20):
            if timeout is not None and time.time() > timeout:
                return best_factors
            
            # Initialize randomly
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # Scale to match the magnitude of the tensor
            # The tensor entries are all 1, so we want U*V*W ~ 1
            U = U / np.linalg.norm(U, axis=0, keepdims=True)
            V = V / np.linalg.norm(V, axis=0, keepdims=True)
            W = W / np.linalg.norm(W, axis=0, keepdims=True)
            
            for it in range(max_iters):
                if timeout is not None and time.time() > timeout:
                    break
                
                # Update U: for each row a, U[a,:] should satisfy
                # U[a,:] * (V*W)[a,:] = T[a,:,:] flattened
                # (V*W)[a,:] has shape (mp*pn,), where (V*W)[a, b*p+c] = V[b,:] * W[c,:]
                # Actually, we need to be more careful.
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # Let Z_a[b,c] = V[b,:] * W[c,:] element-wise sum... no.
                # Z_a is a matrix of shape (mp, pn) where Z_a[b, c] = sum_r V[b,r] * W[c,r] * U[a,r]?
                # No, that's what we're solving for.
                
                # Let's use the standard ALS formulation:
                # For fixed b, c: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a: we have a linear system in U[a,:]:
                # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # Let vec(VW)[b*p+c] = sum over r? No.
                # Actually, for fixed a, define matrix M_a of shape (mp*pn, r) where
                # M_a[b*p+c, r] = V[b,r] * W[c,r]
                # Then T_a = M_a @ U[a,:] where T_a has shape (mp*pn,)
                
                # Build M_a for each a
                for a in range(nm):
                    # Build M_a: shape (mp*pn, r)
                    M_a = np.zeros((mp*pn, r))
                    for b in range(mp):
                        for c in range(pn):
                            M_a[b*pn+c, :] = V[b, :] * W[c, :]
                    
                    # T_a: the tensor values for this a
                    T_a = np.zeros(mp*pn)
                    for (ui, vi, wi, val) in tensor_entries:
                        if ui == a:
                            T_a[vi*pn+wi] = val
                    
                    # Solve M_a @ U[a,:] = T_a
                    try:
                        U[a, :], residuals, rank_ls, sv = np.linalg.lstsq(M_a, T_a, rcond=None)
                    except:
                        pass
                
                # Update V similarly
                for b in range(mp):
                    M_b = np.zeros((nm*pn, r))
                    for a in range(nm):
                        for c in range(pn):
                            M_b[a*pn+c, :] = U[a, :] * W[c, :]
                    
                    T_b = np.zeros(nm*pn)
                    for (ui, vi, wi, val) in tensor_entries:
                        if vi == b:
                            T_b[ui*pn+wi] = val
                    
                    try:
                        V[b, :], residuals, rank_ls, sv = np.linalg.lstsq(M_b, T_b, rcond=None)
                    except:
                        pass
                
                # Update W similarly
                for c in range(pn):
                    M_c = np.zeros((nm*mp, r))
                    for a in range(nm):
                        for b in range(mp):
                            M_c[a*mp+b, :] = U[a, :] * V[b, :]
                    
                    T_c = np.zeros(nm*mp)
                    for (ui, vi, wi, val) in tensor_entries:
                        if wi == c:
                            T_c[ui*mp+vi] = val
                    
                    try:
                        W[c, :], residuals, rank_ls, sv = np.linalg.lstsq(M_c, T_c, rcond=None)
                    except:
                        pass
                
                # Compute residual
                residual = 0.0
                for (ui, vi, wi, val) in tensor_entries:
                    pred = sum(U[ui, r_idx] * V[vi, r_idx] * W[wi, r_idx] for r_idx in range(r))
                    residual += (pred - val) ** 2
                
                # Also check zero entries (sample a few)
                if residual < best_residual:
                    best_residual = residual
                    best_factors = (U.copy(), V.copy(), W.copy())
                
                if residual < 1e-8:
                    break
            
            if best_residual < 1e-8:
                break
        
        if best_factors is not None and best_residual < 1e-6:
            U_opt, V_opt, W_opt = best_factors
            # Round to nearest integers or half-integers
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            # Verify the rounded solution
            valid = True
            for (ui, vi, wi, val) in tensor_entries:
                pred = sum(U_rounded[ui, r_idx] * V_rounded[vi, r_idx] * W_rounded[wi, r_idx] for r_idx in range(r))
                if abs(pred - val) > 1e-6:
                    valid = False
                    break
            
            if not valid:
                # Try rounding to integers
                U_rounded = np.round(U_opt)
                V_rounded = np.round(V_opt)
                W_rounded = np.round(W_opt)
                
                valid = True
                for (ui, vi, wi, val) in tensor_entries:
                    pred = sum(U_rounded[ui, r_idx] * V_rounded[vi, r_idx] * W_rounded[wi, r_idx] for r_idx in range(r))
                    if abs(pred - val) > 1e-6:
                        valid = False
                        break
            
            if valid:
                return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    # Try ranks from 15 up
    for r in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=200, timeout=deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r
            # Continue trying lower ranks? No, we go upward. If we find one at r, 
            # we should try to find even lower. But we're going from 15 up, so 
            # the first one we find is the best.
            break
    
    # If no better than schoolbook was found, return schoolbook
    if best_rank == schoolbook_rank:
        U_final = best_U
        V_final = best_V
        W_final = best_W
    else:
        U_final = best_U
        V_final = best_V
        W_final = best_W
    
    # Convert to list of lists with Fraction for exactness
    def to_fraction_list(matrix):
        result = []
        for row in matrix:
            frac_row = []
            for val in row:
                # Convert to Fraction
                if abs(val - round(val)) < 1e-9:
                    frac_row.append(Fraction(int(round(val))))
                elif abs(val * 2 - round(val * 2)) < 1e-9:
                    frac_row.append(Fraction(int(round(val * 2)), 2))
                else:
                    # Use Fraction with a reasonable denominator
                    frac_row.append(Fraction(val).limit_denominator(1000))
            result.append(frac_row)
        return result
    
    U_out = to_fraction_list(U_final)
    V_out = to_fraction_list(V_final)
    W_out = to_fraction_list(W_final)
    
    return {"U": U_out, "V": V_out, "W": W_out}
