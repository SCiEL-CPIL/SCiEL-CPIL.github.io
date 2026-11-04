import numpy as np
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c]
    # a in 0..5, b in 0..11, c in 0..7
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    lower_bound = max(nm, mp, pn)
    max_rank = n * m * p
    
    def schoolbook():
        rank = max_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def is_half_integer(arr, tol=1e-9):
        # Check if arr is close to multiples of 0.5
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.all(np.abs(doubled - rounded) < tol):
            return True
        return False

    def verify(U, V, W, T, tol=1e-9):
        # U: nm x R, V: mp x R, W: pn x R
        # Check T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        R = U.shape[1]
        # Efficient check using einsum
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this in chunks or just loop over c if R is small
        for c in range(pn):
            # For each c, we need to check if U * V (elementwise sum over r) matches T[:,:,c]
            # Let M[a,b] = sum_r U[a,r] V[b,r]
            # This is U @ V.T? No.
            # M = sum_r U[:,r] V[:,r].T
            # M = U.T @ V ? No.
            # M[a,b] = sum_r U[a,r] V[b,r]
            # This is the element-wise product sum.
            # Can be computed as:
            # Temp = U * V.T ? No, U is nm x R, V is mp x R.
            # We want M (nm x mp).
            # M = U @ V.T gives (nm x mp) where M[a,b] = sum_r U[a,r] V.T[r,b] = sum_r U[a,r] V[b,r].
            # So M = U @ V.T
            M = U @ V.T
            # Now check M * W[c,:] ? No.
            # We need sum_r (U[a,r] V[b,r]) W[c,r]
            # This is not simply M * W[c,:].
            # It is the inner product of the vector (U[:,r] * V[:,r])_flattened with W[c,:]?
            # No, W[c,r] multiplies the r-th term.
            # So for each a,b: val = sum_r M[a,b,r] W[c,r] where M[a,b,r] = U[a,r]V[b,r].
            # This is equivalent to:
            # Let X be nm x mp x R. X[a,b,r] = U[a,r] V[b,r].
            # Then T_recon[a,b,c] = sum_r X[a,b,r] W[c,r].
            # This is a tensor contraction.
            # We can reshape X to (nm*mp, R) and W to (R, pn).
            # Then T_recon_flat = X_flat @ W.T
            # But building X is expensive.
            
            # Alternative:
            # T_recon[:, :, c] = sum_r (U[:, r] * V[:, r].T) * W[c, r]
            # This is a sum of rank-1 matrices.
            # We can just loop over r if R is small (< 20)
            recon = np.zeros((nm, mp))
            for r in range(R):
                recon += (U[:, r] * V[:, r].T) * W[c, r]
            if not np.allclose(recon, T[:, :, c], atol=tol):
                return False
        return True

    def solve_for_rank(R, num_trials=1000):
        for _ in range(num_trials):
            # Random integers in {-1, 0, 1}
            # Scale by 2 later? No, just use integers for now.
            # The problem allows half-integers.
            # Let's generate integers for U and V.
            U = np.random.randint(-1, 2, size=(nm, R)).astype(float)
            V = np.random.randint(-1, 2, size=(mp, R)).astype(float)
            
            # Solve for W
            # For each c, solve Y W_c = b_c
            # Y is nm x R, Y[a,b] is not a matrix, Y is formed by U and V.
            # The equation is: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # Let's construct the system matrix A_c of size (nm*mp) x R?
            # No, the unknowns are W[c, 0..R-1].
            # The equations are for all a, b.
            # There are nm*mp = 6*12 = 72 equations.
            # There are R unknowns.
            # If R < 72, it's overdetermined.
            # If R >= 72, it's underdetermined.
            # For R=12, it is overdetermined (72 eqs, 12 unknowns).
            # We can use least squares.
            
            # Construct A: shape (nm*mp, R)
            # A[idx(a,b), r] = U[a,r] * V[b,r]
            A = np.zeros((nm * mp, R))
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    for r in range(R):
                        A[idx, r] = U[a, r] * V[b, r]
                    idx += 1
            
            # b vector: shape (nm*mp)
            b = T.flatten()
            
            # Solve A W = b for W (vector of length R)
            # Wait, W is different for each c.
            # The equation T[a,b,c] = sum_r ... W[c,r]
            # So for each c, we have a different RHS b_c.
            # b_c[a,b] = T[a,b,c]
            
            W = np.zeros((pn, R))
            valid = True
            for c in range(pn):
                b_c = T[:, :, c].flatten()
                # Solve A x = b_c
                # Since A is same for all c, we can precompute pseudoinverse?
                # But A depends on U, V which are random.
                try:
                    # Least squares solution
                    x, residuals, rank, sv = np.linalg.lstsq(A, b_c, rcond=None)
                    W[c, :] = x
                except:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if W is half-integer
            if is_half_integer(W):
                # Snap to half-integers
                W_snap = np.round(W * 2) / 2
                U_snap = np.round(U * 2) / 2 # U was integer, so this is exact
                V_snap = np.round(V * 2) / 2
                
                # Verify
                if verify(U_snap, V_snap, W_snap, T):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    
        return None

    # Try ranks from lower bound up
    # Lower bound is 12.
    # Schoolbook is 24.
    # We try 12, 13, 14...
    
    # To increase chances, we can also try randomizing which matrix is solved for.
    # But solving for W is symmetric in complexity.
    
    # Optimization: Pre-compute the structure of A?
    # A depends on U and V.
    
    # Let's run the search.
    # Limit time? The loop is fast. 1000 trials * 8 c's * lstsq(72x12) is very fast.
    
    for R in range(lower_bound, max_rank + 1):
        # Increase trials for lower R?
        # Or just fixed trials.
        sol = solve_for_rank(R, num_trials=500)
        if sol:
            return sol
            
    return schoolbook()
