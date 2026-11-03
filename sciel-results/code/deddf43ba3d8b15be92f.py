import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 140  # safety margin

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition exactly."""
    rank = U.shape[1]
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_recon[i, j, k] = s
    T_target = np.ones((n, m, p), dtype=float)
    return np.max(np.abs(T_recon - T_target)) < tol

def tensor_loss(U, V, W, n, m, p):
    """Smooth L2 loss for the decomposition."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    T_target = np.ones((n, m, p), dtype=float)
    diff = T - T_target
    return np.sum(diff * diff)

def tensor_loss_grad(U, V, W, n, m, p):
    """Compute loss and gradients w.r.t. U, V, W."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    T_target = np.ones((n, m, p), dtype=float)
    E = T - T_target
    loss = np.sum(E * E)
    
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                d = 2.0 * E[i, j, k]
                gU[i*m+j, :] += d * V[j*p+k, :] * W[k*n+i, :]
                gV[j*p+k, :] += d * U[i*m+j, :] * W[k*n+i, :]
                gW[k*n+i, :] += d * U[i*m+j, :] * V[j*p+k, :]
    return loss, gU, gV, gW

def optimize_small(n, m, p, rank, max_iter=5000, lr=0.05, time_budget=30):
    """Optimize a small tensor decomposition."""
    np.random.seed(42)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    start_t = time.time()
    current_loss = tensor_loss(U, V, W, n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = current_loss
    patience = 0
    
    for it in range(max_iter):
        if time.time() - start_t > time_budget:
            break
        loss, gU, gV, gW = tensor_loss_grad(U, V, W, n, m, p)
        
        grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
        if grad_norm > 1e-12:
            target_step = 0.1 * np.sqrt(loss) if loss > 1e-6 else 0.01
            lr_current = min(lr * 2, max(1e-6, target_step / (grad_norm + 1e-12)))
        else:
            lr_current = lr
            
        U -= lr_current * gU
        V -= lr_current * gV
        W -= lr_current * gW
        
        new_loss = tensor_loss(U, V, W, n, m, p)
        if new_loss < best_loss:
            best_loss = new_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
        else:
            patience += 1
            if patience > 100:
                break
                
        if best_loss < 1e-9:
            break
            
    return best_U, best_V, best_W, best_loss

def snap_to_half_integer(A, tol=1e-3):
    """Snap values to nearest 0.5 or 1 if close."""
    A_snapped = np.round(A * 2) / 2
    # Check if snapping was good
    if np.max(np.abs(A - A_snapped)) < tol:
        return A_snapped
    return A

def construct():
    n, m, p = 2, 4, 4
    
    # Try to find rank 24 decomposition via two rank 12 (2x4x2) blocks
    # Block 1: C[:, 0:2]
    # Block 2: C[:, 2:4]
    
    # Optimize rank 12 for 2x4x2
    n_sub, m_sub, p_sub = 2, 4, 2
    rank_sub = 12
    
    U1, V1, W1, loss1 = optimize_small(n_sub, m_sub, p_sub, rank_sub, max_iter=10000, lr=0.1, time_budget=40)
    
    if loss1 < 1e-8:
        # Assemble rank 24
        U = np.zeros((n*m, 24))
        V = np.zeros((m*p, 24))
        W = np.zeros((p*n, 24))
        
        # Fill first 12 columns
        U[:, 0:12] = U1
        # V1 is (8, 12) corresponding to B rows 0-3, cols 0-1
        # V is (16, 24). V[j, k] index is j*4+k.
        # For k=0,1, V[j, 0], V[j, 1] should be V1[j, *]
        for j in range(4):
            V[j*4+0, 0:12] = V1[j, :]
            V[j*4+1, 0:12] = V1[j+4, :] # Wait, V1 index is j*p_sub + k_sub. p_sub=2.
            # V1 shape (m_sub*p_sub, rank) = (8, 12).
            # V1[j*2 + 0] corresponds to B[j, 0].
            # V1[j*2 + 1] corresponds to B[j, 1].
            # So V[j*4+0, 0:12] = V1[j*2, 0:12]
            # V[j*4+1, 0:12] = V1[j*2+1, 0:12]
            V[j*4+0, 0:12] = V1[j*2, :]
            V[j*4+1, 0:12] = V1[j*2+1, :]
            
        # W1 is (4, 12) corresponding to C rows 0-1, cols 0-1
        # W index is k*n + i. p_sub=2, n=2.
        # W1[k*2 + i].
        # W is (8, 24). W[k*2 + i].
        # For k=0,1, W[0:4, 0:12] = W1
        W[0:4, 0:12] = W1
        
        # Fill next 12 columns (Block 2)
        # Same U
        U[:, 12:24] = U1
        
        # V2: B rows 0-3, cols 2-3
        # V[j*4+2, 12:24] = V1[j*2, :]
        # V[j*4+3, 12:24] = V1[j*2+1, :]
        for j in range(4):
            V[j*4+2, 12:24] = V1[j*2, :]
            V[j*4+3, 12:24] = V1[j*2+1, :]
            
        # W2: C rows 0-1, cols 2-3
        # W[4:8, 12:24] = W1
        W[4:8, 12:24] = W1
        
        # Snap to half integers
        U = snap_to_half_integer(U)
        V = snap_to_half_integer(V)
        W = snap_to_half_integer(W)
        
        if verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Strassen Rank 28
    # Re-use the logic from the previous program or implement simple Strassen
    rank = 28
    U_s = np.zeros((n*m, rank))
    V_s = np.zeros((m*p, rank))
    W_s = np.zeros((p*n, rank))
    
    strassen_steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def place_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            u_dict, v_dict, w_dict = strassen_steps[s]
            col = col_offset + s
            for li, coeff in u_dict.items():
                lr, lc = li // 2, li % 2
                U_s[lr * m + (a_col_start + lc), col] += coeff
            for li, coeff in v_dict.items():
                lr, lc = li // 2, li % 2
                V_s[(b_row_start + lr) * p + (b_col_start + lc), col] += coeff
            for li, coeff in w_dict.items():
                lr, lc = li // 2, li % 2
                W_s[(c_col_start + lc) * n + lr, col] += coeff

    place_block(0, 0, 0, 0, 0)
    place_block(0, 0, 2, 2, 7)
    place_block(2, 2, 0, 0, 14)
    place_block(2, 2, 2, 2, 21)
    
    if verify_decomposition(U_s, V_s, W_s, n, m, p, tol=1e-9):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
    raise ValueError("No valid decomposition found")
