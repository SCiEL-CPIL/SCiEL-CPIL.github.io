"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(decomp, n, m, p):
    """Verify that the tensor decomposition exactly multiplies the matrices."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    r = U.shape[1]
    
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Check if T is close to 1 for all i,j,k
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    half_ints = np.round(arr * 2.0) / 2.0
    if np.allclose(arr, half_ints, atol=tol):
        return half_ints
    return arr

def _polish_local_search(U, V, W, n, m, p, max_iters=5000, deadline=None):
    """Local search on half-integer values to refine the solution."""
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    def compute_error():
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.sum((T - 1.0) ** 2)
    
    current_error = compute_error()
    candidates = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if current_error < 1e-12:
            break
        
        # Pick a random matrix, row, column
        mat_idx = random.randint(0, 2)
        if mat_idx == 0:
            row = random.randint(0, n * m - 1)
            col = random.randint(0, rank - 1)
            mat = U
        elif mat_idx == 1:
            row = random.randint(0, m * p - 1)
            col = random.randint(0, rank - 1)
            mat = V
        else:
            row = random.randint(0, p * n - 1)
            col = random.randint(0, rank - 1)
            mat = W
        
        old_val = mat[row, col]
        new_val = random.choice(candidates)
        if abs(new_val - old_val) < 1e-10:
            continue
        
        mat[row, col] = new_val
        new_error = compute_error()
        
        if new_error < current_error - 1e-14:
            current_error = new_error
        else:
            mat[row, col] = old_val
    
    return U, V, W, current_error

def _try_rank(n, m, p, rank, max_iter=800, seed=42, deadline=None):
    """Try to find a rank-r decomposition using gradient descent with snapping."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Try multiple initializations
    for init_seed in range(3):
        init_rng = np.random.RandomState(seed + init_seed * 1000)
        
        # Initialize with small random values
        U = init_rng.randn(num_u, rank) * 0.3
        V = init_rng.randn(num_v, rank) * 0.3
        W = init_rng.randn(num_w, rank) * 0.3
        
        lr = 0.05
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Compute loss and gradients
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            diff = T - 1.0
            loss = np.sum(diff * diff)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-14:
                break
            
            # Gradients
            grad_U_t = 2.0 * np.einsum('ijk,jkr,kir->ijr', diff, V_t, W_t)
            grad_V_t = 2.0 * np.einsum('ijk,ijr,kir->jkr', diff, U_t, W_t)
            grad_W_t = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U_t, V_t)
            
            grad_U = grad_U_t.reshape(num_u, rank)
            grad_V = grad_V_t.reshape(num_v, rank)
            grad_W = grad_W_t.reshape(num_w, rank)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if it % 100 == 99:
                lr *= 0.85
        
        if best_loss < 1e-14:
            break
    
    if best_U is None or best_loss > 1e-10:
        return None
    
    # Snap to half-integers
    U_snapped = _snap_to_half_integer(best_U)
    V_snapped = _snap_to_half_integer(best_V)
    W_snapped = _snap_to_half_integer(best_W)
    
    # Check if snapped version is valid
    test_decomp = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
    if _verify(test_decomp, n, m, p):
        return test_decomp
    
    # Try local search polish on the snapped values
    U_polished, V_polished, W_polished, polish_error = _polish_local_search(
        U_snapped, V_snapped, W_snapped, n, m, p, max_iters=3000, deadline=deadline
    )
    
    if polish_error < 1e-12:
        test_decomp = {"U": U_polished.tolist(), "V": V_polished.tolist(), "W": W_polished.tolist()}
        if _verify(test_decomp, n, m, p):
            return test_decomp
    
    # Try local search on the original (unsnapped) values
    U_polished, V_polished, W_polished, polish_error = _polish_local_search(
        best_U, best_V, best_W, n, m, p, max_iters=3000, deadline=deadline
    )
    
    if polish_error < 1e-12:
        # Snap the polished result
        U_final = _snap_to_half_integer(U_polished)
        V_final = _snap_to_half_integer(V_polished)
        W_final = _snap_to_half_integer(W_polished)
        test_decomp = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        if _verify(test_decomp, n, m, p):
            return test_decomp
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from 14 upwards (known lower bound region)
    for rank in range(14, 31):
        if time.time() > deadline:
            break
        
        # Try a few seeds for each rank
        for seed in range(4):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=600, seed=seed + 100, deadline=deadline)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    break
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
