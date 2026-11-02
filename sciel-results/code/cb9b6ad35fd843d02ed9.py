import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5

def _validate(U, V, W, tol=1e-9):
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_vectors():
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

def _structured_85():
    """Build the known rank-85 structured decomposition."""
    n, m, p = N, M, P
    R = 85
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    c = 0
    
    s2 = _strassen_2x2_vectors()
    
    def map_2x2_to_4x4(vec, row_off, col_off):
        out = [0.0] * 16
        for r in range(2):
            for c in range(2):
                out[(row_off + r) * 4 + (col_off + c)] = vec[r * 2 + c]
        return out

    calls = [
        ([ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ]),
        ([ (2,0,1), (2,2,1) ], [ (0,0,1) ], [ (2,0,1), (2,2,-1) ]),
        ([ (0,0,1) ], [ (0,2,1), (2,2,-1) ], [ (0,2,1), (2,2,1) ]),
        ([ (2,2,1) ], [ (2,0,1), (0,0,-1) ], [ (0,0,1), (2,0,1) ]),
        ([ (0,0,1), (0,2,1) ], [ (2,2,1) ], [ (0,0,-1), (0,2,1) ]),
        ([ (2,0,1), (0,0,-1) ], [ (0,0,1), (0,2,1) ], [ (2,2,1) ]),
        ([ (0,2,1), (2,2,-1) ], [ (2,0,1), (2,2,1) ], [ (0,0,1) ])
    ]
    
    for u_blocks, v_blocks, w_blocks in calls:
        for u_vec, v_vec, w_vec in s2:
            u_full = [0.0] * 16
            v_full = [0.0] * 16
            w_full = [0.0] * 16
            
            for row_off, col_off, sign in u_blocks:
                mapped = map_2x2_to_4x4(u_vec, row_off, col_off)
                for i in range(16):
                    u_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in v_blocks:
                mapped = map_2x2_to_4x4(v_vec, row_off, col_off)
                for i in range(16):
                    v_full[i] += sign * mapped[i]
                    
            for row_off, col_off, sign in w_blocks:
                mapped = map_2x2_to_4x4(w_vec, row_off, col_off)
                for i in range(16):
                    w_full[i] += sign * mapped[i]
            
            for i in range(4):
                for j in range(4):
                    u_idx = i * 5 + j
                    u_val = u_full[i * 4 + j]
                    if u_val != 0:
                        U[u_idx, c] += u_val
                        
            for j in range(4):
                for k in range(4):
                    v_idx = j * 5 + k
                    v_val = v_full[j * 4 + k]
                    if v_val != 0:
                        V[v_idx, c] += v_val
                        
            for k in range(4):
                for i in range(4):
                    w_idx = k * 4 + i
                    w_val = w_full[k * 4 + i]
                    if w_val != 0:
                        W[w_idx, c] += w_val
                        
            c += 1
            
    # Part 2: A11 B112 (4x4x1) Rank 16
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0
            c += 1
            
    # Part 3: A12 B21 (4x1x5) Rank 20
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4, c] = 1.0
            V[4 * 5 + k, c] = 1.0
            W[k * 4 + i, c] = 1.0
            c += 1
            
    return U, V, W

def _gradient_search(U, V, W, max_iter=2000, lr=0.01, tol=1e-8, deadline=None):
    """Try to reduce the residual of the decomposition via gradient descent."""
    n, m, p = N, M, P
    R = U.shape[1]
    
    # Compute residual and gradient
    def compute_loss_and_grad(U, V, W):
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # residual = 1 - sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    prod = V[j*p+k, :] * W[k*n+i, :]
                    s = np.dot(U[i*m+j, :], prod)
                    res = 1.0 - s
                    loss += res * res
                    # d/dU[i*m+j,r] = -2 * res * V[j*p+k,r] * W[k*n+i,r]
                    dU[i*m+j, :] += -2.0 * res * prod
                    # d/dV[j*p+k,r] = -2 * res * U[i*m+j,r] * W[k*n+i,r]
                    dV[j*p+k, :] += -2.0 * res * U[i*m+j, :] * W[k*n+i, :]
                    # d/dW[k*n+i,r] = -2 * res * U[i*m+j,r] * V[j*p+k,r]
                    dW[k*n+i, :] += -2.0 * res * U[i*m+j, :] * V[j*p+k, :]
        
        return loss, dU, dV, dW
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    best_loss = np.inf
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        loss, dU, dV, dW = compute_loss_and_grad(U_curr, V_curr, W_curr)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        if best_loss < tol:
            break
        
        # Adaptive learning rate
        if it > 100 and best_loss > 1e-4:
            lr *= 0.95
        if lr < 1e-6:
            break
        
        U_curr -= lr * dU
        V_curr -= lr * dV
        W_curr -= lr * dW
    
    return best_U, best_V, best_W, best_loss

def _try_lower_rank(rank, deadline):
    """Try to find a valid decomposition at the given rank using gradient search."""
    n, m, p = N, M, P
    
    # Initialize with random values, then try to refine
    best_loss = np.inf
    best_U = best_V = best_W = None
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(seed=42 + restart * 1000)
        U = rng.standard_normal((n*m, rank)) * 0.5
        V = rng.standard_normal((m*p, rank)) * 0.5
        W = rng.standard_normal((p*n, rank)) * 0.5
        
        # Quick gradient search
        U_ref, V_ref, W_ref, loss = _gradient_search(
            U, V, W, 
            max_iter=500, 
            lr=0.05, 
            tol=1e-8,
            deadline=deadline
        )
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U_ref, V_ref, W_ref
        
        if best_loss < 1e-8:
            break
    
    if best_loss < 1e-8 and best_U is not None:
        if _validate(best_U, best_V, best_W, tol=1e-7):
            return best_U, best_V, best_W
    
    return None

def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # First, try to find a lower rank via gradient-based search
    # Lower bound is max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    lower_bound = max(n*m, m*p, p*n)
    
    best_rank = 85
    best_U, best_V, best_W = _structured_85()
    
    # Try ranks from lower_bound+1 up to 84, but limit total search time
    search_deadline = t0 + DEADLINE_S - 15  # Leave 15s safety margin
    
    # Try a few promising ranks
    ranks_to_try = []
    # Start from just above lower bound, but skip too many
    for r in range(lower_bound + 1, min(85, lower_bound + 30)):
        ranks_to_try.append(r)
    
    # Sort by likelihood of success (higher ranks easier)
    ranks_to_try.sort(reverse=True)
    
    for rank in ranks_to_try:
        if time.time() > search_deadline:
            break
        
        result = _try_lower_rank(rank, search_deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            # Found a valid lower rank, try even lower
            break
    
    # If no lower rank found, use the structured 85
    if best_rank == 85:
        best_U, best_V, best_W = _structured_85()
    
    # Final validation
    if not _validate(best_U, best_V, best_W, tol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
