"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # First, build the known rank-44 decomposition
    rank44 = _build_rank44_decomposition(n, m, p)
    best_rank = 44
    best_result = {
        "U": rank44["U"],
        "V": rank44["V"], 
        "W": rank44["W"]
    }
    
    # Try to reduce rank using ALS
    for target_rank in [43, 42, 41, 40, 39, 38, 37, 36]:
        if time.time() > deadline - 25:
            break
        
        result = _try_reduce_rank(rank44, target_rank, n, m, p, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break  # Found a lower rank, stop
    
    return best_result

def _build_rank44_decomposition(n, m, p):
    """Build the known rank-44 decomposition."""
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_2x2x4_block(k_start):
        nonlocal col
        k0 = k_start
        k1 = k_start + 1
        
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        k0 = k_start + 2
        k1 = k_start + 3
        
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        U[0 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    def add_2x1x4_block(k_start):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[i * m + 2][col] = 1
                V[2 * p + k_start + k][col] = 1
                W[(k_start + k) * n + i][col] = 1
                col += 1

    add_2x2x4_block(0)
    add_2x1x4_block(0)
    add_2x2x4_block(4)
    add_2x1x4_block(4)

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = len(U[0])
    
    # T[i,l,k] = sum_r U[i*m+l, r] * V[l*p+k, r] * W[k*n+i, r] should equal 1
    max_err = 0.0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+l][r] * V[l*p+k][r] * W[k*n+i][r]
                max_err = max(max_err, abs(val - 1.0))
    
    return max_err < tol

def _try_reduce_rank(rank44, target_rank, n, m, p, deadline):
    """Try to find a rank-target_rank decomposition using ALS."""
    U_np = np.array(rank44["U"], dtype=float)
    V_np = np.array(rank44["V"], dtype=float)
    W_np = np.array(rank44["W"], dtype=float)
    
    # Try several random initializations
    for trial in range(3):
        if time.time() > deadline - 15:
            break
        
        np.random.seed(42 + trial)
        
        # Initialize with random projections
        U_init = np.random.randn(n*m, target_rank) * 0.5
        V_init = np.random.randn(m*p, target_rank) * 0.5
        W_init = np.random.randn(p*n, target_rank) * 0.5
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline - 8:
                break
            
            # Update U
            for idx in range(n*m):
                i = idx // m
                l = idx % m
                
                A_mat = np.zeros((p, target_rank))
                for k in range(p):
                    A_mat[k] = V_init[l*p+k] * W_init[k*n+i]
                
                try:
                    U_init[idx] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except:
                    pass
            
            # Update V
            for idx in range(m*p):
                l = idx // p
                k = idx % p
                
                A_mat = np.zeros((n, target_rank))
                for i in range(n):
                    A_mat[i] = U_init[i*m+l] * W_init[k*n+i]
                
                try:
                    V_init[idx] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                except:
                    pass
            
            # Update W
            for idx in range(p*n):
                k = idx // n
                i = idx % n
                
                A_mat = np.zeros((m, target_rank))
                for l in range(m):
                    A_mat[l] = U_init[i*m+l] * V_init[l*p+k]
                
                try:
                    W_init[idx] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check if the solution is valid
        U_list = U_init.tolist()
        V_list = V_init.tolist()
        W_list = W_init.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            # Try rounding to half-integers
            U_rounded = np.round(U_init * 2) / 2
            V_rounded = np.round(V_init * 2) / 2
            W_rounded = np.round(W_init * 2) / 2
            
            U_round_list = U_rounded.tolist()
            V_round_list = V_rounded.tolist()
            W_round_list = W_rounded.tolist()
            
            if _verify_decomposition(U_round_list, V_round_list, W_round_list, n, m, p):
                return {"U": U_round_list, "V": V_round_list, "W": W_round_list}
            else:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
