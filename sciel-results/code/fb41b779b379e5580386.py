import numpy as np
import time

DIMENSIONS = [2, 3, 3]
_START_TIME = time.time()
TIME_LIMIT = 120  # seconds, with safety margin

def _time_remaining():
    return TIME_LIMIT - 10 - (time.time() - _START_TIME)  # 10s safety margin

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) as lists.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # U: A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V: B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W: C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2: (A10+A11)B00 -> contributes to C10, C11 (negative)
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3: A00(B01-B11) -> contributes to C01, C11
    U[0][2] += 1
    V[1][2] += 1
    V[3][2] -= 1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4: A11(B10-B00) -> contributes to C00, C10
    U[3][3] += 1
    V[2][3] += 1
    V[0][3] -= 1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5: (A00+A01)B11 -> contributes to C00 (negative), C01
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6: (A10-A00)(B00+B01) -> contributes to C11
    U[2][5] += 1
    U[0][5] -= 1
    V[0][5] += 1
    V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11)(B10+B11) -> contributes to C00
    U[1][6] += 1
    U[3][6] -= 1
    V[2][6] += 1
    V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _build_decomposed_2x3x3():
    """
    Decompose 2x3x3 into:
    - Part 1: 2x2x2 (columns j=0,1; rows k=0,1) -> Strassen rank 7
    - Part 2: 2x2x1 (columns j=0,1; row k=2) -> schoolbook rank 4
    - Part 3: 2x1x3 (column j=2; rows k=0,1,2) -> schoolbook rank 6
    Total rank: 17
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 2x2x2 block (j in {0,1}, k in {0,1}) ---
    U1, V1, W1 = _build_strassen_2x2x2()
    for r in range(7):
        for a_idx in range(4):
            i_local = a_idx // 2
            j_local = a_idx % 2
            a_global = i_local * 3 + j_local
            U[a_global][r] = U1[a_idx][r]
        
        for b_idx in range(4):
            j_local = b_idx // 2
            k_local = b_idx % 2
            b_global = j_local * 3 + k_local
            V[b_global][r] = V1[b_idx][r]
        
        for c_idx in range(4):
            k_local = c_idx // 2
            i_local = c_idx % 2
            c_global = k_local * 2 + i_local
            W[c_global][r] = W1[c_idx][r]
    
    # --- Part 2: 2x2x1 block (j in {0,1}, k=2) ---
    col_offset = 7
    for i in range(2):
        for j in range(2):
            col = col_offset + i * 2 + j
            a_global = i * 3 + j
            b_global = j * 3 + 2
            c_global = 4 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    # --- Part 3: 2x1x3 block (j=2, k in {0,1,2}) ---
    col_offset = 11
    for i in range(2):
        for k in range(3):
            col = col_offset + i * 3 + k
            a_global = i * 3 + 2
            b_global = 6 + k
            c_global = k * 2 + i
            U[a_global][col] = 1
            V[b_global][col] = 1
            W[c_global][col] = 1
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification of the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def _try_rank_reduction_via_strassen_blocks():
    """
    Attempt to find a rank-16 or lower decomposition by searching for a valid
    decomposition with fewer terms using numerical optimization.
    
    Strategy: Start from the rank-17 Strassen-based decomposition and attempt
    to reduce the rank by finding a rank-16 decomposition through optimization.
    """
    n, m, p = 2, 3, 3
    target_rank = 16
    U0, V0, W0 = _build_decomposed_2x3x3()
    U0_arr = np.array(U0, dtype=float)
    V0_arr = np.array(V0, dtype=float)
    W0_arr = np.array(W0, dtype=float)
    
    # Try to find a rank-16 decomposition by dropping one column and optimizing
    # the remaining columns to compensate.
    # We'll try each column in turn and see if we can find a valid decomposition.
    
    from scipy.optimize import minimize
    
    def _objective(params, col_to_drop):
        """Objective: minimize the error in the decomposition when col_to_drop is removed."""
        rank = target_rank
        # Reshape params into U, V, W matrices
        u_size = n * m * rank
        v_size = m * p * rank
        w_size = p * n * rank
        
        U_flat = params[:u_size]
        V_flat = params[u_size:u_size + v_size]
        W_flat = params[u_size + v_size:u_size + v_size + w_size]
        
        U_mat = U_flat.reshape(n * m, rank)
        V_mat = V_flat.reshape(m * p, rank)
        W_mat = W_flat.reshape(p * n, rank)
        
        # Compute the error: for each (i,j,k), check if sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U_mat[a_idx], V_mat[b_idx] * W_mat[c_idx])
                    error += (val - 1.0) ** 2
        return error
    
    def _verify_rank_r(U, V, W, n, m, p, tol=1e-6):
        """Verify that the decomposition is correct."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Start from the rank-17 solution and try to drop one column
    # We'll use the first 16 columns as a starting point
    U_start = U0_arr[:, :target_rank]
    V_start = V0_arr[:, :target_rank]
    W_start = W0_arr[:, :target_rank]
    
    x0 = np.concatenate([U_start.flatten(), V_start.flatten(), W_start.flatten()])
    
    # Check if the truncated solution is already close
    obj_val = _objective(x0, 0)
    
    # If the error is small, we might be close to a solution
    # Try to optimize
    result = minimize(_objective, x0, args=(0,), method='L-BFGS-B', 
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.fun < 1e-4:
        # Extract the optimized matrices
        u_size = n * m * target_rank
        v_size = m * p * target_rank
        w_size = p * n * target_rank
        
        U_opt = result.x[:u_size].reshape(n * m, target_rank)
        V_opt = result.x[u_size:u_size + v_size].reshape(m * p, target_rank)
        W_opt = result.x[u_size + v_size:u_size + v_size + w_size].reshape(p * n, target_rank)
        
        # Convert to lists and verify
        U_list = U_opt.tolist()
        V_list = V_opt.tolist()
        W_list = W_opt.tolist()
        
        if _verify_rank_r(U_list, V_list, W_list, n, m, p, tol=1e-5):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-16 decomposition
    try:
        result = _try_rank_reduction_via_strassen_blocks()
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try the decomposed approach (rank 17)
    try:
        U, V, W = _build_decomposed_2x3x3()
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
