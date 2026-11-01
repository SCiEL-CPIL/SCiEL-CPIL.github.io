import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def _target_tensor(n, m, p):
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                key = (i*m+j, j*p+k, k*n+i)
                T[key] = T.get(key, 0) + 1
    return T

def _from_columns(cols, n, m, p, rank):
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for r, (Ac, Bc, Cc) in enumerate(cols):
        for (i, j), v in Ac.items():
            U[i*m+j][r] += v
        for (j, k), v in Bc.items():
            V[j*p+k][r] += v
        for (i, k), v in Cc.items():
            W[k*n+i][r] += v
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    target = _target_tensor(n, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                s = 0
                for r in range(rank):
                    s += U[a][r]*V[b][r]*W[c][r]
                if abs(s - target.get((a, b, c), 0)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _build_rank24(n, m, p):
    """
    Construct a rank-24 decomposition for 3x3x4 matrix multiplication.
    Part 1: Rows 0-1 of A (2x3) times B (3x4) -> C[0:2, 0:4] (2x4), schoolbook rank 24
    Part 2: Row 2 of A (1x3) times B (3x4) -> C[2, 0:4] (1x4), schoolbook rank 12
    Total: 36... wait, let me recalculate.
    
    Actually for 2x3 x 3x4: schoolbook is 2*3*4 = 24
    For 1x3 x 3x4: schoolbook is 1*3*4 = 12
    Total: 36. That's not 24.
    
    Let me just do full schoolbook: 3*3*4 = 36.
    """
    cols = []
    def add_col(Ac, Bc, Cc):
        cols.append((dict(Ac), dict(Bc), dict(Cc)))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_col({(i,j):1}, {(j,k):1}, {(i,k):1})
    
    rank = len(cols)
    return _from_columns(cols, n, m, p, rank)

def _try_optimize_rank(target_rank, n, m, p, deadline, seed=0):
    """
    Try to find a decomposition of given target_rank using numerical optimization.
    Uses the fact that we need to solve for U, V, W such that
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all (a,b,c)
    
    We parameterize with random initializations and use gradient descent.
    """
    np.random.seed(seed)
    
    nm = n*m
    mp = m*p
    pn = p*n
    
    # Build target tensor as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                T[a, b, c] = 1.0
    
    # Initialize with random values
    U = np.random.randn(nm, target_rank) * 0.5
    V = np.random.randn(mp, target_rank) * 0.5
    W = np.random.randn(pn, target_rank) * 0.5
    
    # Compute error
    def compute_error(U, V, W):
        # T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum for efficiency
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return T_rec - T
    
    def loss(U, V, W):
        err = compute_error(U, V, W)
        return np.sum(err**2)
    
    # Try to reduce loss using simple gradient descent with restarts
    best_loss = loss(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Adam-like optimizer
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    t = 0
    max_iters = 5000
    t0 = time.time()
    
    while t < max_iters and (time.time() - t0) < deadline - 5:
        t += 1
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        # Compute gradients
        err = compute_error(U, V, W)
        
        # dL/dU[a,r] = sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
        # = einsum('abc,br,cr->ar', err, V, W)
        gU = np.einsum('abc,br,cr->ar', err, V, W)
        gV = np.einsum('abc,ar,cr->br', err, U, W)
        gW = np.einsum('abc,ar,br->cr', err, U, V)
        
        # Adam update
        m_U = beta1*m_U + (1-beta1)*gU
        v_U = beta2*v_U + (1-beta2)*gU**2
        m_U_hat = m_U / (1 - beta1**t)
        v_U_hat = v_U / (1 - beta2**t)
        U -= lr_t * m_U_hat / (np.sqrt(v_U_hat) + eps)
        
        m_V = beta1*m_V + (1-beta1)*gV
        v_V = beta2*v_V + (1-beta2)*gV**2
        m_V_hat = m_V / (1 - beta1**t)
        v_V_hat = v_V / (1 - beta2**t)
        V -= lr_t * m_V_hat / (np.sqrt(v_V_hat) + eps)
        
        m_W = beta1*m_W + (1-beta1)*gW
        v_W = beta2*v_W + (1-beta2)*gW**2
        m_W_hat = m_W / (1 - beta1**t)
        v_W_hat = v_W / (1 - beta2**t)
        W -= lr_t * m_W_hat / (np.sqrt(v_W_hat) + eps)
        
        cur_loss = loss(U, V, W)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Early stop if loss is very small
        if best_loss < 1e-10:
            break
    
    if best_loss < 1e-6:
        # Try to snap to rational values
        U_snap = np.round(best_U * 100) / 100
        V_snap = np.round(best_V * 100) / 100
        W_snap = np.round(best_W * 100) / 100
        snap_loss = loss(U_snap, V_snap, W_snap)
        if snap_loss < 1e-6:
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), snap_loss
        else:
            return best_U.tolist(), best_V.tolist(), W_snap.tolist(), best_loss
    
    return None, None, None, best_loss

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 115.0
    
    # Start with schoolbook (rank 36)
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    best_result = (U_school, V_school, W_school, n*m*p)
    
    # Try ranks from 20 down to 15 (theoretical lower bound considerations)
    # For 3x3x4, known results suggest rank could be as low as ~24 or less
    for target_rank in [24, 22, 20, 18, 16]:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Try multiple random seeds
        for seed in range(3):
            if time.time() - t0 > DEADLINE - 10:
                break
            
            U, V, W, loss_val = _try_optimize_rank(
                target_rank, n, m, p, 
                deadline=time.time() - t0 + (DEADLINE - 10),
                seed=seed
            )
            
            if U is not None and loss_val < 1e-6:
                # Verify
                if _verify(U, V, W, n, m, p, target_rank):
                    best_result = (U, V, W, target_rank)
                    break  # Found valid decomposition at this rank, try lower
        
        # If we found a valid decomposition, try to go lower
        if best_result[3] < n*m*p:
            continue
        else:
            # If we couldn't beat schoolbook, stop trying
            break
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
