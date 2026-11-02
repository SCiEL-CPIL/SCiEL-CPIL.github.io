import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Build the rank-56 Strassen decomposition (baseline)
    rank_56 = 56
    U56 = np.zeros((n * m, rank_56))
    V56 = np.zeros((m * p, rank_56))
    W56 = np.zeros((p * n, rank_56))
    
    u_vecs = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    v_vecs = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    w_vecs = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            U56[:, col] = u_vecs[s]
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            W56[4*r:4*r+4, col] = w_vecs[s]
            # V indices: k*p+j for k in {0,1}, j in {2r, 2r+1}
            V56[v_indices, col] = v_vecs[s]
    
    # Verify rank-56 solution
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = np.sum(U56[i*m+k] * V56[k*p+j] * W56[j*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to naive rank 64
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * m * p + k * p + j
                    U[i*m+k, idx] = 1
                    V[k*p+j, idx] = 1
                    W[j*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a lower-rank decomposition
    # Strategy: check if any column can be removed (rank 55)
    for col_to_remove in range(rank_56):
        if time.time() > deadline:
            break
        
        U_try = np.delete(U56, col_to_remove, axis=1)
        V_try = np.delete(V56, col_to_remove, axis=1)
        W_try = np.delete(W56, col_to_remove, axis=1)
        
        max_err = 0.0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = np.sum(U_try[i*m+k] * V_try[k*p+j] * W_try[j*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    # Try rank 54: remove two columns
    for col1 in range(rank_56):
        if time.time() > deadline:
            break
        for col2 in range(col1+1, rank_56):
            if time.time() > deadline:
                break
            
            cols_to_remove = [col1, col2]
            U_try = np.delete(U56, cols_to_remove, axis=1)
            V_try = np.delete(V56, cols_to_remove, axis=1)
            W_try = np.delete(W56, cols_to_remove, axis=1)
            
            max_err = 0.0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        val = np.sum(U_try[i*m+k] * V_try[k*p+j] * W_try[j*n+i])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    # Return the rank-56 solution
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
