import numpy as np
import time

_START = time.time()
_LIMIT = 130.0
DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n
    R0 = n * m * p
    
    # Schoolbook fallback
    U0 = np.zeros((Nm, R0))
    V0 = np.zeros((Mp, R0))
    W0 = np.zeros((Pn, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]

    # Target tensor is all ones
    T = np.ones((n, m, p))
    
    # Try to find a lower rank decomposition
    # We will try ranks from 83 down to 40
    # If we find one, we return it.
    
    def check_exact(U, V, W, R):
        # Reconstruct tensor
        # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Efficiently:
        # U is (Nm, R), V is (Mp, R), W is (Pn, R)
        # We can compute the tensor by summing outer products
        # But R is small, so we can loop
        T_recon = np.zeros((n, m, p))
        for r in range(R):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # Outer product u (n,m) x v (m,p) x w (p,n)
            # This is not a standard outer product.
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        # Check if T_recon is close to 1
        if np.allclose(T_recon, T, atol=1e-9):
            return True
        return False

    def snap_half_int(arr):
        # Snap to nearest half integer
        return np.round(arr * 2) / 2

    best_result = None
    best_rank = R0
    
    # We will try a few ranks
    # To save time, we will try a limited number of iterations for each rank
    ranks_to_try = list(range(83, 39, -1))
    
    for R in ranks_to_try:
        if time.time() - _START > _LIMIT - 10:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() - _START > _LIMIT - 5:
                break
                
            # Random initialization
            U = np.random.randn(Nm, R)
            V = np.random.randn(Mp, R)
            W = np.random.randn(Pn, R)
            
            # ALS optimization
            # Minimize || T - sum u_r v_r w_r ||^2
            # This is non-convex. We use simple gradient steps or ALS.
            # ALS: Fix V, W, update U. Fix U, W, update V. Fix U, V, update W.
            
            for it in range(50):
                if time.time() - _START > _LIMIT - 2:
                    break
                    
                # Update U
                # For each i,j, U[i*m+j, :] should minimize sum_k (T[i,j,k] - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])^2
                # This is linear in U[i*m+j, :]
                # Let A[i,j] be the vector of U[i*m+j, :]
                # Term is sum_r A[r] * (sum_k V[j*p+k,r] W[k*n+i,r])
                # Let B[i,j,r] = sum_k V[j*p+k,r] W[k*n+i,r]
                # We want A to minimize || T[i,j,:] - A B^T ||^2? No.
                # T[i,j,k] is a scalar.
                # Residual R[i,j,k] = T[i,j,k] - sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
                # We want to find U[i*m+j, :] = a such that sum_k (T[i,j,k] - sum_r a_r V[j*p+k,r] W[k*n+i,r])^2 is minimized.
                # This is a least squares problem: a M = T[i,j,:]
                # Where M[k, r] = V[j*p+k, r] W[k*n+i, r]
                # So a = T[i,j,:] M^T (M M^T)^-1
                
                for i in range(n):
                    for j in range(m):
                        # Build M (p x R)
                        M_mat = np.zeros((p, R))
                        for k in range(p):
                            M_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve a M_mat^T = T[i,j,:]
                        # a = T[i,j,:] M_mat (M_mat^T M_mat)^-1 ? No.
                        # Minimize || T_vec - M_mat^T a ||^2
                        # M_mat^T is (R x p).
                        # Let X = M_mat^T. We want X a = T_vec.
                        # a = (X^T X)^-1 X^T T_vec
                        try:
                            X = M_mat.T # R x p
                            XtX = X.T @ X # R x R
                            XtT = X.T @ T[i, j, :] # R x 1
                            # Add small regularization
                            XtX += 1e-6 * np.eye(R)
                            a = np.linalg.solve(XtX, XtT)
                            U[i * m + j, :] = a
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, R))
                        for i in range(n):
                            M_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        X = M_mat.T # R x n
                        XtX = X.T @ X
                        XtT = X.T @ T[:, j, k]
                        XtX += 1e-6 * np.eye(R)
                        try:
                            a = np.linalg.solve(XtX, XtT)
                            V[j * p + k, :] = a
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        M_mat = np.zeros((m, R))
                        for j in range(m):
                            M_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        X = M_mat.T # R x m
                        XtX = X.T @ X
                        XtT = X.T @ T[i, :, k]
                        XtX += 1e-6 * np.eye(R)
                        try:
                            a = np.linalg.solve(XtX, XtT)
                            W[k * n + i, :] = a
                        except:
                            pass

            # Check exactness
            if check_exact(U, V, W, R):
                # Snap to half integers
                U_snapped = snap_half_int(U)
                V_snapped = snap_half_int(V)
                W_snapped = snap_half_int(W)
                
                # Verify snapped version
                if check_exact(U_snapped, V_snapped, W_snapped, R):
                    best_result = {"U": _pack(U_snapped), "V": _pack(V_snapped), "W": _pack(W_snapped)}
                    best_rank = R
                    break # Found a valid one for this R, try lower R?
                    # Actually, we want the MINIMAL rank. So we should continue to lower R.
                    # But we are iterating from high to low.
                    # So we found a valid one at R. We should check if lower R is possible.
                    # But we are going down. So we can just break and return this one?
                    # No, we want the minimal. So we should continue the loop to lower R.
                    # But we break the attempt loop.
                    # We should not break the rank loop.
                    # Let's store it and continue.
                    # But if we find a valid one at R, any lower R is better.
                    # So we continue.
                else:
                    # Snapped version failed, try without snap?
                    # The problem requires integer/half-integer.
                    # So we must use snapped.
                    pass
            
            if best_result is not None and best_rank < R:
                # We already found a better one in a previous iteration? No, we are going down.
                # So if we found one at R, we continue to R-1.
                pass

    if best_result is not None:
        return best_result
    else:
        return {"U": _pack(U0), "V": _pack(V0), "W": _pack(W0)}

