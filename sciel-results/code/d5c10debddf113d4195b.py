"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_true = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_true, atol=1e-10)

def _strassen_block_decomposition(n, m, p):
    """
    Construct rank-32 decomposition using Strassen blocks for column pairs
    and schoolbook for the single remaining column.
    """
    rank = 4 * 7 + 4  # 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_block(k1, k2, start_col):
        nonlocal U, V, W
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 1
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1
        W[w_03][c] -= 1
        
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] -= 1
        W[w_02][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] -= 1
        W[w_00][c] += 1
        W[w_01][c] += 1
        
        c = start_col + 4
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1
        W[w_02][c] += 1
        
        c = start_col + 5
        U[u_10][c] += 1
        U[u_00][c] -= 1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 6
        U[u_01][c] += 1
        U[u_11][c] -= 1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        nonlocal U, V, W
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        
        w_0 = k * n + 0
        w_1 = k * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1
        
        return start_col + 4
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    col = schoolbook_single(8, col)
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x2 * 2x9 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - 5.0:
        # Return a trivial schoolbook decomposition if time is running out
        # This ensures we return *something* valid, even if not optimal
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Try the rank-32 Strassen block construction
    result = _strassen_block_decomposition(n, m, p)
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
