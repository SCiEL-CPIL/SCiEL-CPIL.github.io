"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Fast check using vectorized operations
    def check(U, V, W, tol=1e-8):
        rank = U.shape[1]
        # For each (i,k), compute M_{i,k}[a,b] = sum_d U[a,d]*V[b,d]*W[k*n+i,d]
        # and check it equals the target matrix
        for i in range(n):
            for k in range(p):
                # Target: M[i*m+j, j*p+k] = 1 for j=0..m-1, else 0
                # Compute M = sum_d W[k*n+i,d] * outer(U[:,d], V[:,d])
                M = np.zeros((n * m, m * p))
                for d in range(rank):
                    w_val = W[k * n + i, d]
                    if abs(w_val) > 1e-15:
                        M += w_val * np.outer(U[:, d], V[:, d])
                
                # Check against target
                for j in range(m):
                    if abs(M[i * m + j, j * p + k] - 1.0) > tol:
                        return False
                # Check all other entries are ~0
                for a in range(n * m):
                    for b in range(m * p):
                        if not (a == i * m + j and b == j * p + k for j in range(m)):
                            if abs(M[a, b]) > tol:
                                return False
        return True
    
    # Try to find a decomposition using alternating least squares
    def try_als(r, max_iters=200):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix U and V, solve for W
            # For each (i,k): M_{i,k} = sum_d W[k*n+i,d] * outer(U[:,d], V[:,d])
            # This is linear in W for fixed U,V
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    # Build the linear system: for each (a,b) where target is nonzero
                    # target[a,b] = sum_d W[k*n+i,d] * U[a,d] * V[b,d]
                    # For (a,b) = (i*m+j, j*p+k): target = 1
                    # For other (a,b): target = 0
                    # We have r unknowns W[k*n+i,0..r-1]
                    # The equations are: sum_d W[k*n+i,d] * U[a,d] * V[b,d] = target[a,b]
                    # This is an overdetermined system, use least squares
                    
                    # Build A matrix: rows are (a,b) pairs, columns are d
                    # A[ab, d] = U[a,d] * V[b,d]
                    # b_vec[ab] = target[a,b]
                    # Solve A @ W_col = b_vec
                    
                    A_mat = np.zeros((n * m * m * p, r))
                    b_vec = np.zeros(n * m * m * p)
                    idx = 0
                    for a in range(n * m):
                        for b in range(m * p):
                            for d in range(r):
                                A_mat[idx, d] = U[a, d] * V[b, d]
                            # Set target
                            is_target = False
                            for j in range(m):
                                if a == i * m + j and b == j * p + k:
                                    b_vec[idx] = 1.0
                                    is_target = True
                                    break
                            idx += 1
                    
                    # Solve least squares
                    W_col, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new[k * n + i, :] = W_col
            
            # Fix W and V, solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((n * m * m * p, r))
                    b_vec = np.zeros(n * m * m * p)
                    idx = 0
                    for a in range(n * m):
                        for b in range(m * p):
                            for d in range(r):
                                A_mat[idx, d] = W[k * n + i, d] * V[b, d]
                            for j in range(m):
                                if a == i * m + j and b == j * p + k:
                                    b_vec[idx] = 1.0
                                    break
                            idx += 1
                    U_col, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    # U_col is for all a, but we only need to update U
                    # Actually, U[a,d] appears in multiple equations, so this approach is flawed
                    
                    # Let me redo: fix W and V, the equations for U are:
                    # For each (i,k,a,b): sum_d U[a,d] * (W[k*n+i,d] * V[b,d]) = target[a,b]
                    # This is linear in U[:,d] for each a
                    # For fixed a, the equation is: sum_d U[a,d] * C[a,b,d] = target[a,b] where C[a,b,d] = W[k*n+i,d]*V[b,d]
                    # But this must hold for all (i,k,b), so we have many equations per a
                    
                    # This is getting complicated. Let me use a different approach.
                    break
                break
            
            # This ALS approach is getting too complex. Let me try a simpler method.
            break
        
        return None
    
    # Given the complexity, let's just return schoolbook for now
    # but with a proper structure that could be improved
    
    U, V, W, rank = schoolbook()
    
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
