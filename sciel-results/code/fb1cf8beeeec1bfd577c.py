import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_dim = nm * mp * pn
    
    # Build the target tensor T
    # T is a tensor of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
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
    
    def check_decomposition(U, V, W):
        # Verify U @ V @ W == T
        # This is expensive, so we only check non-zero entries of T
        # T has 30 non-zero entries.
        # Sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We can compute the full tensor and compare
        # Tensor product: T_rec[i, v, w] = sum_r U[i, r] V[v, r] W[w, r]
        # This is (nm x R) @ (R x mp) -> nm x mp? No.
        # We need to form the 3D tensor.
        # T_rec = np.einsum('ir,vr,wr->ivw', U, V, W)
        # This might be slow if R is large, but R is small (<=30).
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        # Check if T_rec matches T
        # T has 1s at specific indices, 0 elsewhere.
        # We check if max abs diff is small
        diff = np.max(np.abs(T_rec - T))
        return diff < 1e-6

    def try_rank(R, time_limit=10.0):
        start_time = time.time()
        best_U, best_V, best_W = None, None, None
        
        # Initialize with schoolbook terms
        # If R < schoolbook rank, we can't just take a subset.
        # We initialize U, V, W with random values or from schoolbook.
        # Let's try to initialize from schoolbook by merging?
        # Or just random.
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize U, V, W
            # Use schoolbook columns as a starting point if R <= schoolbook
            if R <= rank_schoolbook:
                # Take first R schoolbook columns
                U = U_sb[:, :R].copy()
                V = V_sb[:, :R].copy()
                W = W_sb[:, :R].copy()
                # Add some noise to escape local minima?
                # Or keep them and try to adjust.
            else:
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
                
            # Alternating Minimization
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Fix U, V, solve for W
                # T[i, v, w] = sum_r U[i, r] V[v, r] W[w, r]
                # For each w, T[:, :, w] = sum_r W[w, r] (U[:, r] outer V[:, r])
                # This is a linear system for W[:, r]
                # Let A_w be a matrix of size (nm*mp) x R
                # A_w[i*mp+v, r] = U[i, r] * V[v, r]
                # b_w[i*mp+v] = T[i, v, w]
                # Solve A_w W[:, r] = b_w for all w?
                # No, W is (pn, R).
                # We can solve for each w independently?
                # The system is coupled because W[:, r] is shared across w?
                # No, W[w, r] is the variable.
                # The equation is:
                # sum_r W[w, r] * (U[:, r] outer V[:, r]) = T[:, :, w]
                # This is a linear system for the vector W[:, r]?
                # No, for a fixed w, we have:
                # sum_r W[w, r] M_r = T_w, where M_r = U[:, r] outer V[:, r]
                # This is a linear combination of R matrices M_r to form T_w.
                # We can vectorize:
                # vec(T_w) = sum_r W[w, r] vec(M_r)
                # Let B be a matrix of size (nm*mp) x R, where B[:, r] = vec(M_r)
                # Then B W[w, :]^T = vec(T_w)
                # We can solve for W[w, :] using least squares.
                
                B = np.zeros((nm * mp, R))
                for r in range(R):
                    M_r = np.outer(U[:, r], V[:, r])
                    B[:, r] = M_r.flatten()
                
                # Solve for W
                W_new = np.zeros((pn, R))
                for w in range(pn):
                    b_w = T[:, :, w].flatten()
                    # Solve B x = b_w
                    x, residuals, rank, sv = np.linalg.lstsq(B, b_w, rcond=None)
                    W_new[w, :] = x
                
                # Check convergence or just update
                W = W_new
                
                # Similarly for V and U
                # Fix V, W, solve for U
                # T[i, v, w] = sum_r U[i, r] (V[:, r] outer W[:, r])
                # For each i, T[i, :, :] = sum_r U[i, r] (V[:, r] outer W[:, r])
                # vec(T[i, :, :]) = sum_r U[i, r] vec(V[:, r] outer W[:, r])
                # Let C be (mp*pn) x R, C[:, r] = vec(V[:, r] outer W[:, r])
                # C U[i, :]^T = vec(T[i, :, :])
                
                C = np.zeros((mp * pn, R))
                for r in range(R):
                    M_r = np.outer(V[:, r], W[:, r])
                    C[:, r] = M_r.flatten()
                
                U_new = np.zeros((nm, R))
                for i in range(nm):
                    b_i = T[i, :, :].flatten()
                    x, residuals, rank, sv = np.linalg.lstsq(C, b_i, rcond=None)
                    U_new[i, :] = x
                U = U_new
                
                # Fix U, W, solve for V
                # T[i, v, w] = sum_r V[v, r] (U[:, r] outer W[:, r])
                # For each v, T[:, v, :] = sum_r V[v, r] (U[:, r] outer W[:, r])
                # vec(T[:, v, :]) = sum_r V[v, r] vec(U[:, r] outer W[:, r])
                # Let D be (nm*pn) x R, D[:, r] = vec(U[:, r] outer W[:, r])
                # D V[v, :]^T = vec(T[:, v, :])
                
                D = np.zeros((nm * pn, R))
                for r in range(R):
                    M_r = np.outer(U[:, r], W[:, r])
                    D[:, r] = M_r.flatten()
                
                V_new = np.zeros((mp, R))
                for v in range(mp):
                    b_v = T[:, v, :].flatten()
                    x, residuals, rank, sv = np.linalg.lstsq(D, b_v, rcond=None)
                    V_new[v, :] = x
                V = V_new
                
            # After AM, check if we found a solution
            if check_decomposition(U, V, W):
                # Try to snap to integers/half-integers
                # Round to nearest half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_decomposition(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
                else:
                    # Return the floating point solution?
                    # The task requires integer/half-integer entries.
                    # If the floating point solution is very close to integers, we return it.
                    # Otherwise, this attempt failed.
                    pass
                    
        return None

    # Try ranks from 29 down to 1
    for R in range(rank_schoolbook - 1, 0, -1):
        result = try_rank(R, time_limit=10.0)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
