import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    def check_decomposition(U, V, W):
        """
        Check if sum_l U[i*m+j, l] V[j*p+k, l] W[k*n+i, l] == 1 for all i,j,k
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        """
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # T[i, j, k] = sum_l U[i, j, l] V[j, k, l] W[k, i, l]
        # Using einsum: U is i,j,l; V is j,k,l; W is k,i,l
        T = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T, 1.0, atol=1e-9)

    def refine(U, V, W, max_iters=20):
        """
        ALS refinement step.
        """
        r = U.shape[1]
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            # For each i, j, we need to solve A x = 1 where A[k, l] = V[j*p+k, l] * W[k*n+i, l]
            # A is (p, r). x is U[i*m+j, :]
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    V_j = V[j*p:(j+1)*p, :] # (p, r)
                    # W_i_mat[k, :] = W[k*n + i, :]
                    W_i_mat = np.array([W[k*n + i, :] for k in range(p)]) # (p, r)
                    A = V_j * W_i_mat # (p, r)
                    b = np.ones(p)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        U_new[i*m + j, :] = x
                    except:
                        U_new[i*m + j, :] = U[i*m + j, :]
            U = U_new
            
            # Update V
            # For each j, k, we need to solve A x = 1 where A[i, l] = U[i*m+j, l] * W[k*n+i, l]
            # A is (n, r). x is V[j*p+k, :]
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        V_new[j*p + k, :] = x
                    except:
                        V_new[j*p + k, :] = V[j*p + k, :]
            V = V_new
            
            # Update W
            # For each k, i, we need to solve A x = 1 where A[j, l] = U[i*m+j, l] * V[j*p+k, l]
            # A is (m, r). x is W[k*n+i, :]
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        W_new[k*n + i, :] = x
                    except:
                        W_new[k*n + i, :] = W[k*n + i, :]
            W = W_new
            
            # Check convergence
            if check_decomposition(U, V, W):
                # Try to snap to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # Try integers
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if check_decomposition(U_int, V_int, W_int):
                        return U_int, V_int, W_int
                    else:
                        # If rounding fails, we can't return float if strict half-integer check is applied
                        # But the prompt says "integer/half-integer entries".
                        # If we can't satisfy it, we might fail.
                        # Let's return None to try another rank/init.
                        return None
        return None

    # Search for low rank
    # Start from a reasonable lower bound. 
    # Lower bound for 3x4x8 is not known to be much less than 96.
    # Let's try 80 to 95.
    # We will use multiple random restarts for each rank.
    
    best_result = None
    
    # We can also try to construct a decomposition using block structures if possible,
    # but random search with ALS is a good general approach.
    
    for r in range(80, rank_sb):
        if time.time() > deadline - 10:
            break
            
        found = False
        for trial in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(r * 100 + trial)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            result = refine(U, V, W, max_iters=30)
            if result is not None:
                U_res, V_res, W_res = result
                return {
                    "U": U_res.tolist(),
                    "V": V_res.tolist(),
                    "W": W_res.tolist()
                }
                found = True
                break
        
    # Fallback to schoolbook
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
