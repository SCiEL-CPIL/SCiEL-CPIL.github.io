import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 50s margin
    
    # Target tensor
    # T[i, j, k] = 1 for all i, j, k
    # Actually, the tensor is defined by the multiplication.
    # The condition is sum_mu U[im+j, mu] * V[jp+k, mu] * W[kn+i, mu] == 1
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find a lower rank decomposition
    # Known rank for 2x3 is 11
    for r in range(11, rank_schoolbook):
        if time.time() > deadline - 5:
            break
        # Try a few random initializations
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS loop
            for it in range(100):
                # Update U
                # Build system for U
                # For each (i, j, k), sum_mu U[im+j, mu] * V[jp+k, mu] * W[kn+i, mu] = 1
                # This is linear in U.
                # Let A be the matrix where A[(i,j,k), (i',j',mu)] = delta_{i,i'} delta_{j,j'} V[j'p+k, mu] W[kn+i', mu]
                # But we can solve for each (i,j) separately?
                # No, U[im+j, mu] appears in all k for that (i,j).
                # For a fixed (i,j), the equations are:
                # sum_mu U[im+j, mu] * V[jp+k, mu] * W[kn+i, mu] = 1 for all k.
                # This is a system of p equations in r unknowns (for each i,j).
                # Wait, U has n*m entries.
                # For each (i,j), we have a system:
                # sum_mu U[im+j, mu] * (V[jp+k, mu] * W[kn+i, mu]) = 1
                # Let M[i,j] be a p x r matrix where M[i,j][k, mu] = V[jp+k, mu] * W[kn+i, mu]
                # Then M[i,j] @ U[:, im+j] = ones(p)
                # We can solve this least squares problem.
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        M_ij = np.zeros((p, r))
                        for k in range(p):
                            for mu in range(r):
                                M_ij[k, mu] = V[j*p+k, mu] * W[k*n+i, mu]
                        # Solve M_ij @ U[:, im+j] = ones(p)
                        sol, _, _, _ = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                        U_new[:, i*m+j] = sol
                        
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # Equations: sum_mu U[im+j, mu] * V[jp+k, mu] * W[kn+i, mu] = 1 for all i
                        # For fixed (j,k), unknown is V[jp+k, :].
                        # sum_mu V[jp+k, mu] * (U[im+j, mu] * W[kn+i, mu]) = 1
                        # Let M_jk be n x r where M_jk[i, mu] = U[i*m+j, mu] * W[k*n+i, mu]
                        # M_jk @ V[jp+k, :]^T = ones(n)
                        M_jk = np.zeros((n, r))
                        for i in range(n):
                            for mu in range(r):
                                M_jk[i, mu] = U[i*m+j, mu] * W[k*n+i, mu]
                        sol, _, _, _ = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = sol
                        
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # Equations: sum_mu U[im+j, mu] * V[jp+k, mu] * W[kn+i, mu] = 1 for all j
                        # For fixed (k,i), unknown is W[kn+i, :].
                        # sum_mu W[kn+i, mu] * (U[im+j, mu] * V[jp+k, mu]) = 1
                        # Let M_ki be m x r where M_ki[j, mu] = U[i*m+j, mu] * V[j*p+k, mu]
                        # M_ki @ W[kn+i, :]^T = ones(m)
                        M_ki = np.zeros((m, r))
                        for j in range(m):
                            for mu in range(r):
                                M_ki[j, mu] = U[i*m+j, mu] * V[j*p+k, mu]
                        sol, _, _, _ = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = sol
                        
                U, V, W = U_new, V_new, W_new
                
                # Check error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += abs(val - 1.0)
                if err < 1e-9:
                    # Found a solution
                    # Snap to half-integers?
                    # The problem asks for integer/half-integer entries.
                    # If the solution is close to half-integers, snap.
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    err_snap = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_snapped[i*m+j, :] * V_snapped[j*p+k, :] * W_snapped[k*n+i, :])
                                err_snap += abs(val - 1.0)
                    if err_snap < 1e-9:
                        best_rank = r
                        best_U = U_snapped
                        best_V = V_snapped
                        best_W = W_snapped
                        break
                    else:
                        # If snapping fails, we might still accept the floating point solution?
                        # The prompt says "integer/half-integer entries".
                        # So we MUST snap.
                        # If we can't snap, this solution is invalid.
                        pass
            if best_rank == r:
                break
                
    # Convert to list of lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
