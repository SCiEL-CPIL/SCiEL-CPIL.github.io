"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    A, B, C = T.shape
    
    def _als_decompose(T, R, seed, max_iter=300, tol=1e-10):
        """ALS for tensor decomposition with structured sparse initialization."""
        rng = np.random.RandomState(seed)
        
        # Structured initialization: exploit the known sparsity of matmul tensor
        # Each rank component should be sparse with values in {-1, 0, 1}
        U = np.zeros((A, R), dtype=np.float64)
        V = np.zeros((B, R), dtype=np.float64)
        W = np.zeros((C, R), dtype=np.float64)
        
        for r in range(R):
            # Sparse random initialization with bias toward zeros
            U[:, r] = rng.choice([0.0, 1.0, -1.0], size=A, p=[0.6, 0.2, 0.2])
            V[:, r] = rng.choice([0.0, 1.0, -1.0], size=B, p=[0.6, 0.2, 0.2])
            W[:, r] = rng.choice([0.0, 1.0, -1.0], size=C, p=[0.6, 0.2, 0.2])
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Solve for U: fix V, W, solve least squares for each row of U
            # U[a, :] = argmin ||T[a,:,:] - sum_r U[a,r] V[:,r] W[:,r]||
            VW = np.einsum('br,cr->bc r', V, W)  # shape (B, C, R)
            for a in range(A):
                target = T[a, :, :].reshape(-1)
                # Build design matrix: each column r is V[:,r] * W[:,r] flattened
                design = np.einsum('br,cr->rbc', V, W).reshape(R, -1)
                U[a, :] = np.linalg.lstsq(design, target, rcond=None)[0]
            
            # Solve for V: fix U, W
            UW = np.einsum('ar,cr->ac r', U, W)
            for b in range(B):
                target = T[:, b, :].reshape(-1)
                design = np.einsum('ar,cr->rac', U, W).reshape(R, -1)
                V[b, :] = np.linalg.lstsq(design, target, rcond=None)[0]
            
            # Solve for W: fix U, V
            UV = np.einsum('ar,br->ab r', U, V)
            for c in range(C):
                target = T[:, :, c].reshape(-1)
                design = np.einsum('ar,br->rab', U, V).reshape(R, -1)
                W[c, :] = np.linalg.lstsq(design, target, rcond=None)[0]
            
            # Check convergence
            if it % 10 == 0 or it == max_iter - 1:
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(recon - T))
                if err < tol:
                    return U, V, W, err
                if err < prev_err * 0.99 and it > 50:
                    # Converged locally
                    break
                prev_err = err
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(recon - T))
        if err < tol:
            return U, V, W, err
        return None, None, None, err
    
    def _snap_to_half_int(U, V, W, T, tol=1e-8):
        """Try to snap to half-integers while preserving validity."""
        # Try different rounding strategies
        for mult in [2, 4, 8, 16]:
            U_r = np.round(U * mult) / mult
            V_r = np.round(V * mult) / mult
            W_r = np.round(W * mult) / mult
            recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.max(np.abs(recon - T)) < tol:
                return U_r, V_r, W_r
        return None
    
    # Try ranks from 20 down to 18 (known min is 20 for 2x3x4)
    best_result = None
    for rank in [20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts for each rank
        num_restarts = 5 if rank == 20 else 3
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            U, V, W, err = _als_decompose(T, rank, seed=42 + seed)
            if U is not None and err < 1e-8:
                # Try to snap to half-integers
                snapped = _snap_to_half_int(U, V, W, T)
                if snapped is not None:
                    U_s, V_s, W_s = snapped
                    # Final verification
                    recon = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
                    if np.max(np.abs(recon - T)) < 1e-8:
                        best_result = (U_s, V_s, W_s, rank)
                        break
                else:
                    # Use the float solution as-is (should be close to half-integers)
                    recon = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(recon - T)) < 1e-8:
                        best_result = (U, V, W, rank)
                        break
        
        if best_result is not None:
            break
    
    # If we found a lower rank, return it; otherwise fall back to schoolbook
    if best_result is not None:
        U, V, W, rank = best_result
        # Convert to lists of lists with float values (evaluator accepts integer/half-integer)
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition of rank n*m*p = 24
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
