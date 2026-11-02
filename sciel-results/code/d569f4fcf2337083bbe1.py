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

def _build_base_85():
    """Build the base rank-85 decomposition using Strassen for 4x4x4 block."""
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

def _try_reduce_rank(U, V, W, target_rank, max_iter=500, lr=0.01, time_limit=30.0):
    """Try to reduce rank by optimizing with a penalty for rank reduction."""
    t0 = time.time()
    n, m, p = N, M, P
    R = U.shape[1]
    
    # We'll try to find a valid decomposition of rank target_rank
    # by randomly initializing and using gradient descent on the constraint
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    
    # This is a non-convex problem, so we use random restarts
    best_loss = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(10):
        if time.time() - t0 > time_limit:
            break
        
        U_r = np.random.randn(n*m, target_rank) * 0.5
        V_r = np.random.randn(m*p, target_rank) * 0.5
        W_r = np.random.randn(p*n, target_rank) * 0.5
        
        for it in range(max_iter):
            if time.time() - t0 > time_limit:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U_r)
            grad_V = np.zeros_like(V_r)
            grad_W = np.zeros_like(W_r)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # s = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        s = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                        err = s - 1.0
                        loss += err * err
                        
                        # Gradient with respect to U[i*m+j, r]
                        # d/dU_r [err^2] = 2*err * V[j*p+k,r] * W[k*n+i,r]
                        grad_U[i*m+j, :] += 2.0 * err * V_r[j*p+k, :] * W_r[k*n+i, :]
                        grad_V[j*p+k, :] += 2.0 * err * U_r[i*m+j, :] * W_r[k*n+i, :]
                        grad_W[k*n+i, :] += 2.0 * err * U_r[i*m+j, :] * V_r[j*p+k, :]
            
            # Gradient descent
            U_r -= lr * grad_U
            V_r -= lr * grad_V
            W_r -= lr * grad_W
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            if loss < 1e-12:
                break
    
    if best_loss < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    t0 = time.time()
    n, m, p = N, M, P
    
    # Start with the base rank-85 decomposition
    U, V, W = _build_base_85()
    
    if not _validate(U, V, W):
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
    
    # Try to reduce rank from 85 downward
    best_U, best_V, best_W = U, V, W
    best_rank = 85
    
    for target_rank in range(84, 70, -1):
        if time.time() - t0 > DEADLINE_S - 30:
            break
        
        result = _try_reduce_rank(U, V, W, target_rank, max_iter=300, lr=0.005, time_limit=25.0)
        if result is not None:
            rU, rV, rW = result
            if _validate(rU, rV, rW):
                best_U, best_V, best_W = rU, rV, rW
                best_rank = target_rank
                break  # Found a valid lower rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
