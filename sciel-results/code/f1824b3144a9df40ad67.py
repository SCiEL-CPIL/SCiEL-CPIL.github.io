"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def _make_schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-8, n_checks=200, seed=42):
    """Fast verification using random sampling."""
    np.random.seed(seed)
    for _ in range(n_checks):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
        if abs(val - 1.0) > tol:
            return False
    return True

def _structured_init(n, m, p, rank, seed=0):
    """Create structured initialization inspired by block decomposition."""
    np.random.seed(seed)
    # Use a mix of structured and random initialization
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First, try to use a partial schoolbook structure
    # Distribute the rank among different (i,j,k) patterns
    used = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if used < rank:
                    U[i*m+j, used] = 1.0
                    V[j*p+k, used] = 1.0
                    W[k*n+i, used] = 1.0
                    used += 1
    
    # If rank < n*m*p, add some structured randomness to remaining columns
    for c in range(used, rank):
        # Random sparse initialization
        U[:, c] = np.random.randn(n * m) * 0.5
        V[:, c] = np.random.randn(m * p) * 0.5
        W[:, c] = np.random.randn(p * n) * 0.5
    
    return U, V, W

def _gradient_descent(U, V, W, n, m, p, max_steps=200, lr=0.05, tol=1e-10, deadline=None):
    """Perform gradient descent to minimize decomposition error."""
    rank = U.shape[1]
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute all (i,j,k) indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        
        # Compute predictions for all entries
        preds = np.zeros(len(indices))
        for idx, (i, j, k) in enumerate(indices):
            preds[idx] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
        
        loss = np.mean((preds - 1.0) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        d_loss_d_pred = 2 * (preds - 1.0) / len(indices)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for idx, (i, j, k) in enumerate(indices):
            factor = d_loss_d_pred[idx]
            u_row = U[i*m+j, :]
            v_row = V[j*p+k, :]
            w_row = W[k*n+i, :]
            
            grad_U[i*m+j, :] += factor * v_row * w_row
            grad_V[j*p+k, :] += factor * u_row * w_row
            grad_W[k*n+i, :] += factor * u_row * v_row
        
        # Adaptive learning rate
        current_lr = lr * (0.95 ** (step // 50))
        
        U -= current_lr * grad_U
        V -= current_lr * grad_V
        W -= current_lr * grad_W
    
    return best_U, best_V, best_W, best_loss

def _snap_to_half_int(arr, max_denom=2):
    """Snap values to nearest half-integer or integer."""
    result = arr.copy()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            val = result[i, j]
            # Try 0, ±0.5, ±1, ±1.5, etc. up to some bound
            best_val = val
            best_err = float('inf')
            for k in range(-20, 21):
                candidate = k / max_denom
                err = abs(candidate - val)
                if err < best_err:
                    best_err = err
                    best_val = candidate
            result[i, j] = best_val
    return result

def _try_rank(n, m, p, target_rank, deadline, max_restarts=3):
    """Try to find a valid decomposition of the given target rank."""
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None, None, None, False
        
        # Structured initialization
        U, V, W = _structured_init(n, m, p, target_rank, seed=restart * 100 + 42)
        
        # Gradient descent
        U, V, W, loss = _gradient_descent(U, V, W, n, m, p, max_steps=150, lr=0.03, deadline=deadline)
        
        if loss < 1e-8:
            # Verify fully
            if _verify_full(U, V, W, n, m, p, tol=1e-9):
                return U, V, W, True
        
        # Try snapping to half-integers
        U_snapped = _snap_to_half_int(U)
        V_snapped = _snap_to_half_int(V)
        W_snapped = _snap_to_half_int(W)
        if _verify_full(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
            return U_snapped, V_snapped, W_snapped, True
    
    return None, None, None, False

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as guaranteed fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try progressively lower ranks
    # Based on known bounds, try from 65 down to 55
    for target_rank in range(65, 54, -1):
        if time.time() > deadline - 10:
            break
        
        U, V, W, success = _try_rank(n, m, p, target_rank, deadline)
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid decomposition at this rank
    
    # Convert to lists of lists with float values
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
