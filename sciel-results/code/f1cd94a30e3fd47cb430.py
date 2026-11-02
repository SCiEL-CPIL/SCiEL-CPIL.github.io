"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _compute_loss_and_grads(U, V, W, n, m, p):
    """Compute loss and analytical gradients efficiently using vectorization."""
    rank = U.shape[1]
    loss = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # Precompute products for efficiency
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i * m + j]
                vj = V[j * p + k]
                wk = W[k * n + i]
                prod = ui * vj * wk
                s = np.sum(prod)
                diff = s - 1.0
                loss += diff * diff
                # Gradient: d/dU = 2*diff * V*W, similarly for V and W
                grad_U[i * m + j] += 2 * diff * vj * wk
                grad_V[j * p + k] += 2 * diff * ui * wk
                grad_W[k * n + i] += 2 * diff * ui * vj
    
    return loss, grad_U, grad_V, grad_W

def _optimize_decomposition(n, m, p, rank, U_init, V_init, W_init, max_iter=5000, lr=0.05, timeout=30.0):
    """Optimize decomposition using analytical gradient descent with momentum."""
    start = time.time()
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    
    # Momentum
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    momentum = 0.9
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start > timeout:
            break
        
        loss, gU, gV, gW = _compute_loss_and_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
        
        # Update with momentum
        vU = momentum * vU + (1 - momentum) * gU
        vV = momentum * vV + (1 - momentum) * gV
        vW = momentum * vW + (1 - momentum) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
    
    # Final check
    loss, _, _, _ = _compute_loss_and_grads(best_U, best_V, best_W, n, m, p)
    if loss < 1e-8:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist(),
            "loss": loss
        }
    return None

def _polish_decomposition(n, m, p, rank, U_init, V_init, W_init, timeout=20.0):
    """High-precision polish pass with adaptive learning rate and tighter tolerance."""
    start = time.time()
    U = np.array(U_init, dtype=np.float64).copy()
    V = np.array(V_init, dtype=np.float64).copy()
    W = np.array(W_init, dtype=np.float64).copy()
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    momentum = 0.95
    lr = 0.01
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(20000):
        if time.time() - start > timeout:
            break
        
        loss, gU, gV, gW = _compute_loss_and_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-14:
            break
        
        # Adaptive learning rate
        norm_g = np.linalg.norm(gU) + np.linalg.norm(gV) + np.linalg.norm(gW)
        if norm_g > 0:
            lr = max(0.001, min(0.05, 0.01 / (norm_g + 1e-8)))
        
        # Update with momentum
        vU = momentum * vU + (1 - momentum) * gU
        vV = momentum * vV + (1 - momentum) * gV
        vW = momentum * vW + (1 - momentum) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
    
    loss, _, _, _ = _compute_loss_and_grads(best_U, best_V, best_W, n, m, p)
    if loss < 1e-12:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist(),
            "loss": loss
        }
    return None

def _try_near_schoolbook_init(n, m, p, rank, seed=None):
    """Initialize near schoolbook with random perturbations for lower rank."""
    if seed is not None:
        np.random.seed(seed)
    
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Create lower-rank initialization by combining schoolbook terms
    U = np.random.normal(0, 0.5, (n * m, rank))
    V = np.random.normal(0, 0.5, (m * p, rank))
    W = np.random.normal(0, 0.5, (p * n, rank))
    
    # Add structured components based on schoolbook
    for c in range(rank):
        # Randomly assign which (i,j,k) triples this column handles
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if np.random.random() < 0.3:
                        U[i * m + j, c] += 1.0
                        V[j * p + k, c] += 1.0
                        W[k * n + i, c] += 1.0
    
    return U, V, W

def _try_random_initializations(n, m, p, rank, timeout_per_attempt=15.0, num_attempts=5):
    """Try multiple random initializations."""
    start = time.time()
    for attempt in range(num_attempts):
        if time.time() - start > timeout_per_attempt * num_attempts:
            break
        
        # Random initialization
        U = np.random.uniform(-1, 1, (n * m, rank))
        V = np.random.uniform(-1, 1, (m * p, rank))
        W = np.random.uniform(-1, 1, (p * n, rank))
        
        result = _optimize_decomposition(n, m, p, rank, U, V, W, 
                                         max_iter=3000, lr=0.05, 
                                         timeout=timeout_per_attempt)
        if result is not None:
            return result
    
    return None

def _try_near_schoolbook_search(n, m, p, rank, timeout_per_attempt=10.0, num_attempts=5):
    """Try initializations near schoolbook structure."""
    start = time.time()
    for attempt in range(num_attempts):
        if time.time() - start > timeout_per_attempt * num_attempts:
            break
        
        U, V, W = _try_near_schoolbook_init(n, m, p, rank, seed=attempt * 1000)
        
        result = _optimize_decomposition(n, m, p, rank, U, V, W, 
                                         max_iter=5000, lr=0.03, 
                                         timeout=timeout_per_attempt)
        if result is not None:
            return result
    
    return None

def _try_polish_from_near_schoolbook(n, m, p, rank, timeout=15.0, num_attempts=3):
    """Try to polish near-schoolbook initializations to high precision."""
    start = time.time()
    for attempt in range(num_attempts):
        if time.time() - start > timeout * num_attempts:
            break
        
        U, V, W = _try_near_schoolbook_init(n, m, p, rank, seed=attempt * 2000)
        
        # First do a rough optimization
        result_rough = _optimize_decomposition(n, m, p, rank, U, V, W, 
                                               max_iter=2000, lr=0.05, 
                                               timeout=timeout * 0.5)
        
        if result_rough is not None:
            # Then polish
            result_polished = _polish_decomposition(n, m, p, rank, 
                                                    result_rough["U"], result_rough["V"], result_rough["W"],
                                                    timeout=timeout * 0.5)
            if result_polished is not None:
                return result_polished
        else:
            # Try polishing directly from near-schoolbook init
            result_polished = _polish_decomposition(n, m, p, rank, 
                                                    U.tolist(), V.tolist(), W.tolist(),
                                                    timeout=timeout)
            if result_polished is not None:
                return result_polished
    
    return None

def _try_structured_decomposition(n, m, p):
    """
    Try a structured decomposition based on block structure.
    For 3x3 * 3x4, we can potentially share computations.
    
    Strategy: Use the fact that we can decompose into:
    - 2x2 * 2x4 block (can use Strassen-like methods)
    - Cross terms with the remaining row/column
    
    This is a heuristic approach that may find rank < 36.
    """
    # This is complex; for now, try a few specific structured patterns
    # that might allow rank reduction.
    
    # Pattern 1: Try to merge some schoolbook terms
    # If we can find coefficients such that multiple (i,j,k) triples
    # share the same column c, we reduce rank.
    
    # For now, return None and rely on optimization
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Known lower bounds and targets for 3x3x4
    # Try ranks from 35 down to 25
    target_ranks = list(range(35, 24, -1))
    
    for rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        # Method 1: Near-schoolbook structured initialization
        result = _try_near_schoolbook_search(n, m, p, rank, 
                                             timeout_per_attempt=8.0, 
                                             num_attempts=3)
        if result is not None and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best = result
            best_rank = rank
            break
        
        # Method 2: Polish from near-schoolbook
        if time.time() > deadline - 10:
            break
        result = _try_polish_from_near_schoolbook(n, m, p, rank, 
                                                   timeout=8.0, 
                                                   num_attempts=2)
        if result is not None and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best = result
            best_rank = rank
            break
        
        # Method 3: Pure random search
        if time.time() > deadline - 5:
            break
        result = _try_random_initializations(n, m, p, rank, 
                                             timeout_per_attempt=5.0, 
                                             num_attempts=2)
        if result is not None and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best = result
            best_rank = rank
            break
    
    # If we found an improvement, try to push further
    if best_rank < n * m * p:
        for rank in range(best_rank - 1, max(best_rank - 5, 20), -1):
            if time.time() > deadline - 10:
                break
            
            result = _try_near_schoolbook_search(n, m, p, rank,
                                                 timeout_per_attempt=5.0,
                                                 num_attempts=2)
            if result is not None and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = rank
                break
    
    return best
