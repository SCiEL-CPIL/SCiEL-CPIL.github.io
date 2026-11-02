"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matmul tensor T[i,j,k]=1."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(R, max_seeds=100, max_iter=2000):
        """Try to find a rank-R half-integer decomposition via numerical optimization."""
        if time.time() > deadline - 20:
            return None

        # Reshape for vectorized computation
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        def residuals(params):
            U = params[:n*m*R].reshape(n*m, R)
            V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = params[n*m*R + m*p*R:].reshape(p*n, R)
            
            # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We can do this with einsum
            # U_flat[i*m+j, r] -> U[i, j, r]
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # Sum over r
            # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
            # einsum: 'ijr,jkr,ki r->ijk'
            try:
                T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            except Exception:
                # Fallback to loops if einsum fails (shouldn't)
                T = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
            
            return (T - 1.0).flatten()

        try:
            from scipy.optimize import least_squares
        except ImportError:
            return None

        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                return None

            np.random.seed(seed)
            # Initialize with small random values
            scale = 1.0
            x0 = np.random.randn(n*m*R + m*p*R + p*n*R) * scale
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=max_iter)
            except Exception:
                continue

            if result.cost < 1e-10:
                U_sol = result.x[:n*m*R].reshape(n*m, R)
                V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = result.x[n*m*R + m*p*R:].reshape(p*n, R)

                # Snap to half-integers
                U_rounded = np.round(U_sol * 2) / 2.0
                V_rounded = np.round(V_sol * 2) / 2.0
                W_rounded = np.round(W_sol * 2) / 2.0

                U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(n*m)]
                V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(m*p)]
                W_list = [[float(W_rounded[i, r]) for r in range(R)] for i in range(p*n)]

                if verify_decomposition(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If not exact, try local search on half-integer grid
                # Only if close
                if result.cost < 1e-6:
                    # Perturb each column independently?
                    # This is expensive, so only do it for small R
                    if R <= 25:
                        for _ in range(100):
                            if time.time() > deadline - 5:
                                break
                            # Pick a random column
                            r_idx = np.random.randint(R)
                            # Pick a random matrix
                            mat_idx = np.random.randint(3)
                            # Perturb entry
                            row_idx = np.random.randint(n*m if mat_idx==0 else m*p if mat_idx==1 else p*n)
                            
                            # Get current value
                            if mat_idx == 0:
                                cur = U_rounded[row_idx, r_idx]
                            elif mat_idx == 1:
                                cur = V_rounded[row_idx, r_idx]
                            else:
                                cur = W_rounded[row_idx, r_idx]
                            
                            # Try neighbors
                            for delta in [-0.5, 0.5]:
                                new_val = cur + delta
                                if mat_idx == 0:
                                    U_rounded[row_idx, r_idx] = new_val
                                elif mat_idx == 1:
                                    V_rounded[row_idx, r_idx] = new_val
                                else:
                                    W_rounded[row_idx, r_idx] = new_val
                                
                                U_l = [[float(U_rounded[i, r]) for r in range(R)] for i in range(n*m)]
                                V_l = [[float(V_rounded[i, r]) for r in range(R)] for i in range(m*p)]
                                W_l = [[float(W_rounded[i, r]) for r in range(R)] for i in range(p*n)]
                                
                                if verify_decomposition(U_l, V_l, W_l, R):
                                    return {"U": U_l, "V": V_l, "W": W_l}
                                
                                # Revert
                                if mat_idx == 0:
                                    U_rounded[row_idx, r_idx] = cur
                                elif mat_idx == 1:
                                    V_rounded[row_idx, r_idx] = cur
                                else:
                                    W_rounded[row_idx, r_idx] = cur

        return None

    # Try ranks from lower bound up to schoolbook
    for R in range(21, 43):
        if time.time() > deadline - 5:
            break
        result = try_rank(R)
        if result is not None:
            return result

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
