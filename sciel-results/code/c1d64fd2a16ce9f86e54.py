import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Try to construct rank 41 solution
    try:
        U, V, W = construct_rank_41()
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to schoolbook rank 45
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct_rank_41():
    """
    Constructs a rank-41 decomposition by combining a rank-23 decomposition
    for the 3x3x3 block and a rank-18 schoolbook decomposition for the 3x3x2 block.
    """
    n, m, p = 3, 3, 5
    
    # Part 1: 3x3x3 (Rank 23)
    U1, V1, W1 = _solve_rank_23_block()
    
    # Part 2: 3x3x2 (Rank 18)
    U2, V2, W2 = _construct_rank_18_block()
    
    # Concatenate
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Snap to nearest half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    return U, V, W

def _solve_rank_23_block():
    """
    Solves for a rank-23 decomposition of the 3x3x3 matrix multiplication tensor.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    # Target tensor T[a, b, c]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()
    
    def residuals(params):
        U = params[:rank*9].reshape(rank, 9)
        V = params[rank*9:2*rank*9].reshape(rank, 9)
        W = params[2*rank*9:].reshape(rank, 9)
        T_calc = np.einsum('ra,rb,rc->abc', U, V, W)
        return (T_calc - T).flatten()
    
    np.random.seed(42)
    best_params = None
    min_cost = float('inf')
    
    # Try multiple random starts
    for start in range(10):
        U0 = np.random.randn(rank, 9)
        V0 = np.random.randn(rank, 9)
        W0 = np.random.randn(rank, 9)
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residuals, params0, method='lm', max_nfev=5000)
            if result.cost < min_cost:
                min_cost = result.cost
                best_params = result.x
                if result.cost < 1e-12:
                    break
        except Exception:
            continue
            
    if best_params is None or min_cost > 1e-6:
        raise ValueError("Failed to converge to rank 23 solution")
        
    U = best_params[:rank*9].reshape(rank, 9)
    V = best_params[rank*9:2*rank*9].reshape(rank, 9)
    W = best_params[2*rank*9:].reshape(rank, 9)
    
    return U, V, W

def _construct_rank_18_block():
    """
    Constructs the rank-18 schoolbook decomposition for the 3x3x2 part.
    """
    n, m, p = 3, 3, 5
    rank = 18
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 3
    row_offset = 3
    
    idx = 0
    for k_local in range(2):
        global_k = col_offset + k_local
        global_row_k = row_offset + k_local
        
        for i in range(n):
            for j in range(m):
                a = i * m + j
                b = j * p + global_k
                c = global_row_k * n + i
                
                U[a, idx] = 1.0
                V[b, idx] = 1.0
                W[c, idx] = 1.0
                idx += 1
                
    return U, V, W

def verify(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    C = A @ B
    C_target = np.zeros(p*n)
    for i in range(n):
        for k in range(p):
            C_target[k*n + i] = C[i, k]
            
    u = U_np @ A_flat
    v = V_np @ B_flat
    c = W_np @ (u * v)
    
    if not np.allclose(c, C_target, atol=1e-5):
        return False
    return True
