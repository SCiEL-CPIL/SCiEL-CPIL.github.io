import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s limit
    
    # Target ranks to try
    target_ranks = [23, 24, 25, 26, 27, 28, 29, 30]
    
    # Fallback: Schoolbook
    def get_schoolbook():
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
    
    # ALS Solver
    def try_als(rank, max_iters=2000):
        # Initialize
        np.random.seed(42)
        U = np.random.rand(n * m, rank) * 2 - 1
        V = np.random.rand(m * p, rank) * 2 - 1
        W = np.random.rand(p * n, rank) * 2 - 1
        
        # Precompute indices
        # U[i*m+j], V[j*p+k], W[k*n+i]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A (p x rank) and b (p x 1)
                    A = np.zeros((p, rank))
                    b = np.zeros(p)
                    for k in range(p):
                        # W index: k*n + i
                        w_row = W[k * n + i]
                        # V index: j*p + k
                        v_row = V[j * p + k]
                        A[:, :] = v_row * w_row # Element-wise product
                        b[k] = 1.0
                    # Solve A u = b
                    # A is p x rank. p=4, rank=23.
                    # Underdetermined. Use least squares.
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    U[i * m + j] = u_sol
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    b = np.zeros(n)
                    for i in range(n):
                        u_row = U[i * m + j]
                        w_row = W[k * n + i]
                        A[:, :] = u_row * w_row
                        b[i] = 1.0
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    V[j * p + k] = v_sol
            
            # Update W
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, rank))
                    b = np.zeros(m)
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        A[:, :] = u_row * v_row
                        b[j] = 1.0
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        continue
                    W[k * n + i] = w_sol
            
            # Check residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        residual += (val - 1.0) ** 2
            if residual < 1e-8:
                # Round to integers or halves
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_int[i * m + j], V_int[j * p + k] * W_int[k * n + i])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Convert to list of lists
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        return None

    # Try ranks
    for r in target_ranks:
        res = try_als(r)
        if res:
            return res
            
    # Fallback
    return get_schoolbook()
