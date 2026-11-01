"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np

def _check_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # Build index arrays for vectorized check
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = np.dot(U[i*m + j, :], V[jp*p + k, :] * W[k*n + i, :])
                    if abs(val - target) > tol:
                        return False
    return True

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the squared error of the decomposition (vectorized)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = np.dot(U[i*m + j, :], V[jp*p + k, :] * W[k*n + i, :])
                    error += (val - target) ** 2
    return error

def _try_rank(n, m, p, rank, max_attempts=50000):
    """Try to find a rank-r decomposition via structured sparse random search."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    candidates = [-2, -1, 0, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        # Generate sparse matrices: each column has few non-zero entries
        U = np.zeros(U_shape)
        V = np.zeros(V_shape)
        W = np.zeros(W_shape)
        
        for c in range(rank):
            n_u = np.random.randint(1, 4)
            n_v = np.random.randint(1, 4)
            n_w = np.random.randint(1, 4)
            
            u_idx = np.random.choice(n*m, size=n_u, replace=False)
            v_idx = np.random.choice(m*p, size=n_v, replace=False)
            w_idx = np.random.choice(p*n, size=n_w, replace=False)
            
            U[u_idx, c] = np.random.choice(candidates, size=n_u)
            V[v_idx, c] = np.random.choice(candidates, size=n_v)
            W[w_idx, c] = np.random.choice(candidates, size=n_w)
        
        if _check_decomposition(U, V, W, n, m, p, rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_half(n, m, p, rank, max_attempts=30000):
    """Try with half-integer candidates."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    candidates = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        U = np.zeros(U_shape)
        V = np.zeros(V_shape)
        W = np.zeros(W_shape)
        
        for c in range(rank):
            n_u = np.random.randint(1, 4)
            n_v = np.random.randint(1, 4)
            n_w = np.random.randint(1, 4)
            
            u_idx = np.random.choice(n*m, size=n_u, replace=False)
            v_idx = np.random.choice(m*p, size=n_v, replace=False)
            w_idx = np.random.choice(p*n, size=n_w, replace=False)
            
            U[u_idx, c] = np.random.choice(candidates, size=n_u)
            V[v_idx, c] = np.random.choice(candidates, size=n_v)
            W[w_idx, c] = np.random.choice(candidates, size=n_w)
        
        if _check_decomposition(U, V, W, n, m, p, rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank_gradient(n, m, p, rank, max_attempts=500):
    """Try random init then gradient descent to refine."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        np.random.seed(42 + attempt)
        U = np.random.uniform(-1, 1, size=U_shape)
        V = np.random.uniform(-1, 1, size=V_shape)
        W = np.random.uniform(-1, 1, size=W_shape)
        
        # Gradient descent
        lr = 0.01
        for step in range(200):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            target = 1.0 if j == jp else 0.0
                            val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[k*n+i, :])
                            diff = val - target
                            dU[i*m+j, :] += diff * V[jp*p+k, :] * W[k*n+i, :]
                            dV[jp*p+k, :] += diff * U[i*m+j, :] * W[k*n+i, :]
                            dW[k*n+i, :] += diff * U[i*m+j, :] * V[jp*p+k, :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Snap to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _check_decomposition(U_r, V_r, W_r, n, m, p, rank):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
        # Also try integer snap
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if _check_decomposition(U_i, V_i, W_i, n, m, p, rank):
            return U_i.tolist(), V_i.tolist(), W_i.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try ranks from 13 upward
    for rank in range(13, schoolbook_rank + 1):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # For lower ranks, spend more time
        if rank <= 15:
            attempts_sparse = 20000
            attempts_half = 10000
            attempts_grad = 200
        elif rank <= 17:
            attempts_sparse = 10000
            attempts_half = 5000
            attempts_grad = 100
        else:
            attempts_sparse = 5000
            attempts_half = 2000
            attempts_grad = 50
        
        # Try sparse integer search
        result = _try_rank(n, m, p, rank, max_attempts=attempts_sparse)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try sparse half-integer search
        result = _try_rank_half(n, m, p, rank, max_attempts=attempts_half)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try gradient-based search
        result = _try_rank_gradient(n, m, p, rank, max_attempts=attempts_grad)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
    
    # Fall back to schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
