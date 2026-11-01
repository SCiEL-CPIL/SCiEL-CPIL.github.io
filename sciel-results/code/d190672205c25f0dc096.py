import numpy as np
import time

def construct():
        n, m, p = 2, 3, 3
        # Schoolbook fallback
        def schoolbook():
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}
        
        start_time = time.time()
        deadline = start_time + 100 # 100s margin
        
        # Target tensor T[i, j1, j2, k] = delta[j1, j2] if indices match?
        # Condition: sum_c W[k, i, c] U[i, j1, c] V[j2, k, c] = delta[j1, j2)
        # Note: i, k are free indices. j1, j2 are summed indices.
        
        # We want to find U, V, W of rank r.
        
        def solve_als(r, max_iter=1000, tol=1e-8):
            # Initialize
            np.random.seed(42)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Reshape for easier indexing
            # U[i*m+j, c]
            # V[j*p+k, c]
            # W[k*n+i, c]
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U
                # For each i, j1:
                # sum_c W[k, i, c] U[i, j1, c] V[j2, k, c] = delta[j1, j2)
                # This is not quite right. The equation is for all i, k, j1, j2.
                # Let's fix i, j1.
                # We have equations for all k, j2.
                # sum_c (W[k*n+i, c] * V[j2*p+k, c]) * U[i*m+j1, c] = delta[j1, j2)
                
                # Let A_{k, j2, c} = W[k*n+i, c] * V[j2*p+k, c]
                # Then sum_c A_{k, j2, c} * U_{i, j1, c} = delta[j1, j2)
                
                # We can vectorize this.
                # For fixed i, j1:
                # We have a system of equations for U[i*m+j1, :]
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j1 in range(m):
                        # Build matrix M of size (p*m) x r
                        # Rows indexed by (k, j2)
                        # M[(k, j2), c] = W[k*n+i, c] * V[j2*p+k, c]
                        M = np.zeros((p * m, r))
                        for k in range(p):
                            for j2 in range(m):
                                # M row index
                                row_idx = k * m + j2
                                # W[k*n+i, c] * V[j2*p+k, c]
                                M[row_idx, :] = W[k * n + i, :] * V[j2 * p + k, :]
                        
                        # Target vector b of size p*m
                        # b[k*m+j2] = delta[j1, j2)
                        b = np.zeros(p * m)
                        for k in range(p):
                            for j2 in range(m):
                                if j1 == j2:
                                    b[k * m + j2] = 1.0
                                else:
                                    b[k * m + j2] = 0.0
                        
                        # Solve M * U_vec = b
                        # Least squares
                        U_vec, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                        U_new[i * m + j1, :] = U_vec
                
                U = U_new
                
                # Update V
                V_new = np.zeros_like(V)
                for j2 in range(m):
                    for k in range(p):
                        # Fix j2, k.
                        # Equation: sum_c W[k*n+i, c] U[i*m+j1, c] V[j2*p+k, c] = delta[j1, j2)
                        # For fixed j2, k:
                        # sum_c (W[k*n+i, c] * U[i*m+j1, c]) * V[j2*p+k, c] = delta[j1, j2)
                        # Variables: i, j1.
                        # M row index (i, j1)
                        M = np.zeros((n * m, r))
                        for i in range(n):
                            for j1 in range(m):
                                row_idx = i * m + j1
                                M[row_idx, :] = W[k * n + i, :] * U[i * m + j1, :]
                        
                        b = np.zeros(n * m)
                        for i in range(n):
                            for j1 in range(m):
                                if j1 == j2:
                                    b[i * m + j1] = 1.0
                                else:
                                    b[i * m + j1] = 0.0
                            
                        V_vec, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                        V_new[j2 * p + k, :] = V_vec
                        
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # Fix k, i.
                        # Equation: sum_c W[k*n+i, c] U[i*m+j1, c] V[j2*p+k, c] = delta[j1, j2)
                        # For fixed k, i:
                        # sum_c (U[i*m+j1, c] * V[j2*p+k, c]) * W[k*n+i, c] = delta[j1, j2)
                        # Variables: j1, j2.
                        M = np.zeros((m * m, r))
                        for j1 in range(m):
                            for j2 in range(m):
                                row_idx = j1 * m + j2
                                M[row_idx, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                        
                        b = np.zeros(m * m)
                        for j1 in range(m):
                            for j2 in range(m):
                                if j1 == j2:
                                    b[j1 * m + j2] = 1.0
                                else:
                                    b[j1 * m + j2] = 0.0
                            
                        W_vec, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                        W_new[k * n + i, :] = W_vec
                        
                W = W_new
                
                # Check convergence
                # Calculate error
                err = 0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                val = 0
                                for c in range(r):
                                    val += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                                target = 1.0 if j1 == j2 else 0.0
                                err += (val - target) ** 2
                if err < tol:
                    break
            
            # Snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Verify
            err_snap = 0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            val = 0
                            for c in range(r):
                                val += W_snapped[k * n + i, c] * U_snapped[i * m + j1, c] * V_snapped[j2 * p + k, c]
                            target = 1.0 if j1 == j2 else 0.0
                            err_snap += (val - target) ** 2
            
            if err_snap < 1e-9:
                return U_snapped, V_snapped, W_snapped
            else:
                return None
        
        for r in range(14, 18):
            if time.time() > deadline:
                break
            res = solve_als(r)
            if res:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return schoolbook()
