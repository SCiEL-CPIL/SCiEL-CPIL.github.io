"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if the output index matches."""
    # T is of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    reconstructed = np.zeros_like(T)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        reconstructed += np.einsum('a,b,d->abd', uc, vc, wc)
    return np.allclose(reconstructed, T, atol=tol)

def _try_rank(n, m, p, rank, max_iter=5000, time_limit=None):
    """Try to find a rank-rank decomposition using iterative optimization."""
    if time_limit is not None:
        start = time.time()
        if time.time() - start > time_limit:
            return None
    
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Initialize U, V, W randomly
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_loss = float('inf')
    best_state = (U.copy(), V.copy(), W.copy())
    
    for iteration in range(max_iter):
        if time_limit is not None:
            if time.time() - start > time_limit:
                break
        
        # Compute current tensor
        T_current = np.zeros((n * m, m * p, p * n))
        for c in range(rank):
            T_current += np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
        
        loss = np.sum((T_current - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_state = (U.copy(), V.copy(), W.copy())
        
        if best_loss < 1e-15:
            break
        
        # Gradient-based update
        lr = 0.01 / (1 + iteration * 0.001)
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(rank):
            # dT/dc = outer product of U[:,c], V[:,c], W[:,c]
            dT_dc = np.einsum('a,b,d->abd', U[:, c], V[:, c], W[:, c])
            err = dT_dc - T
            
            # Gradient for U[:, c]
            dU[:, c] = np.einsum('abd,bd->a', err, V[:, c] * W[:, c])
            # Gradient for V[:, c]
            dV[:, c] = np.einsum('abd,ad->b', err, U[:, c] * W[:, c])
            # Gradient for W[:, c]
            dW[:, c] = np.einsum('abd,ab->d', err, U[:, c] * V[:, c])
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found a valid decomposition
    U_best, V_best, W_best = best_state
    if _verify_decomposition(U_best, V_best, W_best, n, m, p):
        return U_best, V_best, W_best
    
    return None

def _snap_to_rational(U, V, W, max_denom=10):
    """Try to snap values to rational numbers with small denominators."""
    U_snapped = np.zeros_like(U)
    V_snapped = np.zeros_like(V)
    W_snapped = np.zeros_like(W)
    
    for mat, mat_snapped in [(U, U_snapped), (V, V_snapped), (W, W_snapped)]:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if abs(val) < 1e-10:
                    mat_snapped[i, j] = 0.0
                else:
                    # Try to find a rational approximation
                    snapped = False
                    for denom in range(1, max_denom + 1):
                        num = round(val * denom)
                        if abs(val - num / denom) < 1e-10:
                            mat_snapped[i, j] = num / denom
                            snapped = True
                            break
                    if not snapped:
                        mat_snapped[i, j] = val
    
    return U_snapped, V_snapped, W_snapped

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s margin for the 150s limit
    
    # Try ranks from lower bound up
    # Lower bound for matrix multiplication rank: max(n*m, m*p, p*n)
    lower_bound = max(n * m, m * p, p * n)
    schoolbook_rank = n * m * p
    
    best_result = None
    
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=2000, time_limit=deadline - time.time())
            if result is not None:
                U, V, W = result
                # Try to snap to rationals
                U_snapped, V_snapped, W_snapped = _snap_to_rational(U, V, W)
                if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p):
                    best_result = (U_snapped, V_snapped, W_snapped, rank)
                    found = True
                    break
                else:
                    best_result = (U, V, W, rank)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
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
    
    U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
