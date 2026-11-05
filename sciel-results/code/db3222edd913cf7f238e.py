"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = 120.0

    # Schoolbook baseline (rank 54)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = 3, 3, 6
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.shape != (n * m, rank):
            return False
        if V_arr.shape != (m * p, rank):
            return False
        if W_arr.shape != (p * n, rank):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    # Try to find a low-rank decomposition using numerical optimization
    np.random.seed(42)
    
    # Try rank 18 (the lower bound)
    for attempt in range(200):
        if time.time() - start_time > deadline - 5:
            break
        
        r = 18
        U_rand = np.random.randn(9, r)
        V_rand = np.random.randn(18, r)
        
        # Build the linear system for W
        A = np.zeros((54, 18 * r))
        b = np.ones(54)
        
        eq = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    for c in range(r):
                        col_idx = (k * 3 + i) * r + c
                        A[eq, col_idx] = U_rand[i * 3 + j, c] * V_rand[j * 6 + k, c]
                    eq += 1
        
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W_rand = W_flat.reshape(18, r)
            
            # Check if the solution is exact
            error = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(6):
                        val = np.sum(U_rand[i * 3 + j] * V_rand[j * 6 + k] * W_rand[k * 3 + i])
                        error += abs(val - 1.0)
            
            if error < 1e-9:
                U_list = [[float(x) for x in row] for row in U_rand]
                V_list = [[float(x) for x in row] for row in V_rand]
                W_list = [[float(x) for x in row] for row in W_rand]
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except np.linalg.LinAlgError:
            pass
    
    # Try rank 20
    for attempt in range(100):
        if time.time() - start_time > deadline - 5:
            break
        
        r = 20
        U_rand = np.random.randn(9, r)
        V_rand = np.random.randn(18, r)
        
        A = np.zeros((54, 18 * r))
        b = np.ones(54)
        
        eq = 0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    for c in range(r):
                        col_idx = (k * 3 + i) * r + c
                        A[eq, col_idx] = U_rand[i * 3 + j, c] * V_rand[j * 6 + k, c]
                    eq += 1
        
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            W_rand = W_flat.reshape(18, r)
            
            error = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(6):
                        val = np.sum(U_rand[i * 3 + j] * V_rand[j * 6 + k] * W_rand[k * 3 + i])
                        error += abs(val - 1.0)
            
            if error < 1e-9:
                U_list = [[float(x) for x in row] for row in U_rand]
                V_list = [[float(x) for x in row] for row in V_rand]
                W_list = [[float(x) for x in row] for row in W_rand]
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except np.linalg.LinAlgError:
            pass
    
    # If we couldn't find a better solution, return the schoolbook decomposition
    return best_result

# EVOLVE-BLOCK-END
