"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c] for c in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _als_decompose_vectorized(target_rank, deadline, seed=42, max_iters=500):
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    rank = target_rank
    # Initialize with small random values
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                # A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]
                # V_j = V[j*p:(j+1)*p, :] shape (p, rank)
                # W_ki = W.reshape(p, n, rank)[:, i, :] shape (p, rank)
                V_j = V[j*p:(j+1)*p, :]
                W_ki = W.reshape(p, n, rank)[:, i, :]
                A_mat = V_j * W_ki # shape (p, rank)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                # B_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                # U_ij = U.reshape(n, m, rank)[:, j, :] shape (n, rank)
                # W_ki = W.reshape(p, n, rank)[k, :, :] shape (n, rank)
                U_ij = U.reshape(n, m, rank)[:, j, :]
                W_ki = W.reshape(p, n, rank)[k, :, :]
                B_mat = U_ij * W_ki # shape (n, rank)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(B_mat, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                # C_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # U_ij = U.reshape(n, m, rank)[i, :, :] shape (m, rank)
                # V_jk = V.reshape(m, p, rank)[:, k, :] shape (m, rank)
                U_ij = U.reshape(n, m, rank)[i, :, :]
                V_jk = V.reshape(m, p, rank)[:, k, :]
                C_mat = U_ij * V_jk # shape (m, rank)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(C_mat, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Calculate error
        try:
            T_approx = np.einsum('ijc,jkc,kic->ijk', 
                                 U.reshape(n, m, rank), 
                                 V.reshape(m, p, rank), 
                                 W.reshape(p, n, rank))
            err = np.sum((T_approx - 1.0)**2)
        except:
            err = float('inf')
            
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-12:
            break
            
    if best_err < 1e-9:
        # Snap to half-integers
        U_list = [[round(x * 2) / 2 for x in row] for row in best_U]
        V_list = [[round(x * 2) / 2 for x in row] for row in best_V]
        W_list = [[round(x * 2) / 2 for x in row] for row in best_W]
        
        if _verify(U_list, V_list, W_list, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Try integer rounding
            U_list = [[round(x) for x in row] for row in best_U]
            V_list = [[round(x) for x in row] for row in best_V]
            W_list = [[round(x) for x in row] for row in best_W]
            if _verify(U_list, V_list, W_list, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
            return None
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Try rank 20 with multiple seeds
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        result = _als_decompose_vectorized(20, deadline, seed=seed, max_iters=200)
        if result is not None:
            return result
            
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
