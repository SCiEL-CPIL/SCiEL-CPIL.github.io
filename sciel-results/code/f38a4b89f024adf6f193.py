import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _als_decompose(T, n, m, p, rank, deadline, max_sweeps=200, seed=0):
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = T.shape
    
    # Initialize with small random values to avoid large numbers
    U = rng.randn(dim1, rank) * 0.1
    V = rng.randn(dim2, rank) * 0.1
    W = rng.randn(dim3, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 5.0:
            break
        
        # Update U
        # T1[i, j, k] = U[i, c] V[j, c] W[k, c]
        # For fixed V, W, T1 is linear in U.
        # Reshape T to (dim1, dim2*dim3)
        # Reshape V, W to (dim2*dim3, rank)
        VW = np.einsum('jc,kc->jkc', V, W).reshape(-1, rank)
        T1 = T.reshape(dim1, -1)
        try:
            U_new, _, _, _ = np.linalg.lstsq(VW, T1.T, rcond=None)
            U = U_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update V
        UW = np.einsum('ic,kc->ikc', U, W).reshape(-1, rank)
        T2 = np.transpose(T, (1, 0, 2)).reshape(dim2, -1)
        try:
            V_new, _, _, _ = np.linalg.lstsq(UW, T2.T, rcond=None)
            V = V_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Update W
        UV = np.einsum('ic,jc->ijc', U, V).reshape(-1, rank)
        T3 = np.transpose(T, (2, 0, 1)).reshape(dim3, -1)
        try:
            W_new, _, _, _ = np.linalg.lstsq(UV, T3.T, rcond=None)
            W = W_new.T
        except np.linalg.LinAlgError:
            pass
        
        # Compute loss
        reconstructed = np.einsum('ic,jc,kc->ijk', U, V, W)
        loss = np.sum((reconstructed - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        U_list = [[round(float(x), 8) for x in row] for row in best_U]
        V_list = [[round(float(x), 8) for x in row] for row in best_V]
        W_list = [[round(float(x), 8) for x in row] for row in best_W]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank(T, n, m, p, rank, deadline, max_trials=50):
    for seed in range(max_trials):
        if time.time() > deadline - 10.0:
            break
        result = _als_decompose(T, n, m, p, rank, deadline, max_sweeps=100, seed=seed)
        if result is not None:
            return result
    return None

def _construct_2x3x6_from_2x3x3(res_2x3x3):
    """
    Construct 2x3x6 decomposition from 2x3x3 decomposition.
    res_2x3x3: dict with U (6xR), V (9xR), W (6xR) for 2x3x3.
    We need U (6x2R), V (18x2R), W (12x2R) for 2x3x6.
    
    2x3x3: A(2x3), B(3x3), C(2x3)
    2x3x6: A(2x3), B(3x6), C(2x6)
    
    B = [B1, B2] where B1, B2 are 3x3.
    C = [C1, C2] where C1, C2 are 2x3.
    
    If A*B1 = sum U_r V_r W_r (indices for 2x3x3)
    And A*B2 = sum U_r V'_r W'_r
    
    Then A*B = sum U_r V_r W_r (for C1 part) + sum U_r V'_r W'_r (for C2 part)
    
    The U factors are the same.
    The V factors: first R columns are V (for B1), next R columns are V' (for B2).
    The W factors: first R columns are W (for C1), next R columns are W' (for C2).
    
    Wait, the indices for V and W in the 2x3x3 case are:
    V: (m*p) x R = 9 x R. Indices j*p+k.
    W: (p*n) x R = 6 x R. Indices k*n+i.
    
    For 2x3x6:
    V: (m*p) x 2R = 18 x 2R.
    W: (p*n) x 2R = 12 x 2R.
    
    For the first R terms (corresponding to B1, C1):
    V[0:9, 0:R] = V_2x3x3
    W[0:6, 0:R] = W_2x3x3
    
    For the next R terms (corresponding to B2, C2):
    V[9:18, R:2R] = V_2x3x3 (assuming B2 is treated identically)
    W[6:12, R:2R] = W_2x3x3
    
    U[0:6, 0:2R] = [U_2x3x3, U_2x3x3]
    """
    U_233 = np.array(res_2x3x3['U'])
    V_233 = np.array(res_2x3x3['V'])
    W_233 = np.array(res_2x3x3['W'])
    
    R = U_233.shape[1]
    
    U_new = np.hstack([U_233, U_233])
    V_new = np.zeros((18, 2*R))
    V_new[0:9, 0:R] = V_233
    V_new[9:18, R:2*R] = V_233
    
    W_new = np.zeros((12, 2*R))
    W_new[0:6, 0:R] = W_233
    W_new[6:12, R:2*R] = W_233
    
    U_list = [[round(float(x), 8) for x in row] for row in U_new]
    V_list = [[round(float(x), 8) for x in row] for row in V_new]
    W_list = [[round(float(x), 8) for x in row] for row in W_new]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook_decomposition(n, m, p):
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Strategy 1: Try to find rank 11 for 2x3x3, then construct rank 22 for 2x3x6
    # This is a known upper bound.
    T_233 = _build_tensor(2, 3, 3)
    res_233 = _try_rank(T_233, 2, 3, 3, 11, deadline, max_trials=50)
    
    if res_233 is not None:
        # Verify the 2x3x3 solution
        if _verify_decomposition(res_233['U'], res_233['V'], res_233['W'], 2, 3, 3, tol=1e-6):
            res_236 = _construct_2x3x6_from_2x3x3(res_233)
            if _verify_decomposition(res_236['U'], res_236['V'], res_236['W'], 2, 3, 6, tol=1e-6):
                return res_236
    
    # Strategy 2: Direct search for 2x3x6 with ranks 18-22
    T_236 = _build_tensor(2, 3, 6)
    for rank in range(18, 23):
        if time.time() > deadline - 20.0:
            break
        res = _try_rank(T_236, 2, 3, 6, rank, deadline, max_trials=20)
        if res is not None:
            return res
            
    # Fallback: Schoolbook
    return _schoolbook_decomposition(2, 3, 6)
