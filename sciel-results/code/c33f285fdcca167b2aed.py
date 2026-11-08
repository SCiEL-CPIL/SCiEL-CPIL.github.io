"""Program for matrix multiplication tensor decomposition (3x4, 4x6)."""
import time
import numpy as np

# The dimensions: n x m * m x p = 3x4 * 4x6
n_dim = 3
m_dim = 4
p_dim = 6

def _verify_fast(U, V, W, tensor, tol=1e-9):
    """Fast verification using vectorization."""
    rank = U.shape[1]
    n, m, p = tensor.shape
    
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        err = np.max(np.abs(reconstructed - tensor))
        return err < tol
    except Exception:
        return False

def _als_step(U, V, W, tensor, n, m, p, rank):
    """Perform one step of Alternating Least Squares."""
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # 1. Solve for U
    U_new = np.zeros((n, m, rank))
    for i in range(n):
        for j in range(m):
            A = V_reshaped[j, :, :] * W_reshaped[:, i, :] # (p, rank)
            b = tensor[i, j, :]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i, j, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    U = U_new.reshape(n * m, rank)
    U_reshaped = U.reshape(n, m, rank)
    
    # 2. Solve for V
    V_new = np.zeros((m, p, rank))
    for j in range(m):
        for k in range(p):
            A = U_reshaped[:, j, :] * W_reshaped[k, :, :] # (n, rank)
            b = tensor[:, j, k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j, k, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    V = V_new.reshape(m * p, rank)
    V_reshaped = V.reshape(m, p, rank)
    
    # 3. Solve for W
    W_new = np.zeros((p, n, rank))
    for k in range(p):
        for i in range(n):
            A = U_reshaped[i, :, :] * V_reshaped[:, k, :] # (m, rank)
            b = tensor[i, :, k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k, i, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    W = W_new.reshape(p * n, rank)
    
    return U, V, W

def _polish(U, V, W, tensor, n, m, p, rank, max_iters=50, tol=1e-12):
    """Polish the solution to higher precision using more iterations and tighter tolerance."""
    prev_err = np.inf
    for _ in range(max_iters):
        U, V, W = _als_step(U, V, W, tensor, n, m, p, rank)
        
        # Check error
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        err = np.max(np.abs(reconstructed - tensor))
        
        if err < tol:
            return U, V, W, True
        if abs(err - prev_err) < 1e-14:
            break
        prev_err = err
    return U, V, W, False

def construct():
    deadline = 145  # seconds, safe margin before 150s limit
    start_time = time.time()
    
    n = n_dim
    m = m_dim
    p = p_dim
    
    # Create the standard tensor: tensor[i,j,k] = 1 for all i,j,k
    tensor = np.ones((n, m, p), dtype=np.float64)
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    best_rank = rank_schoolbook
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb
    
    rng = np.random.default_rng(42)
    
    # Try ranks from low to high
    # Lower bound for rank of T_{3,4,6} is max(n,m,p)=6, but likely higher.
    # Known lower bounds for matrix multiplication are often close to n*m*p/sqrt(max(n,m,p)) or similar,
    # but let's just search from a reasonable lower bound.
    # For 3x4x6, a rank of ~20-25 is plausible.
    # Let's try from 15 up to schoolbook.
    
    ranks_to_try = list(range(15, rank_schoolbook + 1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        # More attempts for lower ranks
        max_attempts = 10 if target_rank < 30 else 5
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = rng.standard_normal((n * m, target_rank))
            V = rng.standard_normal((m * p, target_rank))
            W = rng.standard_normal((p * n, target_rank))
            
            # Run ALS for a few iterations
            success = False
            for iter in range(30):
                if time.time() > deadline - 2:
                    break
                U, V, W = _als_step(U, V, W, tensor, n, m, p, target_rank)
                
                # Check for convergence/exactness
                if _verify_fast(U, V, W, tensor):
                    # Polish to ensure high precision
                    U, V, W, polished = _polish(U, V, W, tensor, n, m, p, target_rank)
                    if polished or _verify_fast(U, V, W, tensor):
                        best_rank = target_rank
                        best_U = U
                        best_V = V
                        best_W = W
                        success = True
                        break
            
            if success:
                found = True
                break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
