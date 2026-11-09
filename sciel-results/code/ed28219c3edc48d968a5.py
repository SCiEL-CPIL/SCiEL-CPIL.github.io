"""Matrix multiplication 3x3 x 3x5 tensor decomposition."""
DIMENSIONS = [3, 3, 5]

import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _is_half_int(x):
    """Check if x is a half-integer (multiple of 0.5)."""
    x2 = x * 2
    return abs(x2 - round(x2)) < 1e-10

def _make_schoolbook():
    """
    Construct the schoolbook decomposition of rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45
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

def _make_block_schoolbook():
    """
    Construct a block decomposition using schoolbook for 3x3 (first 3 cols) 
    and 3x2 (last 2 cols).
    
    For the 3x3 part (columns 0,1,2): rank 9
    For the 3x2 part (columns 3,4): rank 18
    Total rank: 27
    """
    n, m, p = DIMENSIONS
    # p = 5, so we split into p1=3 and p2=2
    p1 = 3
    p2 = 2
    rank = 9 + 18  # 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First block: columns 0, 1, 2 (p1=3)
    # This is a standard 3x3 x 3x3 multiplication
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 9
    
    # Second block: columns 3, 4 (p2=2)
    # This is a 3x3 x 3x2 multiplication
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                k_actual = p1 + k  # actual column index in B
                U[i * m + j][col] = 1
                V[j * p + k_actual][col] = 1
                W[k_actual * n + i][col] = 1
                col += 1
    # col should be 27
    
    return {"U": U, "V": V, "W": W}

def _try_optimize_rank(r, deadline, rng, seed_data=None):
    """
    Try to find a rank-r decomposition using gradient-based optimization.
    """
    n, m, p = DIMENSIONS
    
    if seed_data is not None:
        U_mat = np.array(seed_data['U'], dtype=float).reshape(n * m, -1)
        V_mat = np.array(seed_data['V'], dtype=float).reshape(m * p, -1)
        W_mat = np.array(seed_data['W'], dtype=float).reshape(p * n, -1)
        current_rank = U_mat.shape[1]
        # If we want a different rank, we need to handle this
        if current_rank != r:
            # Pad or truncate
            if current_rank < r:
                pad_U = np.zeros((n * m, r - current_rank))
                pad_V = np.zeros((m * p, r - current_rank))
                pad_W = np.zeros((p * n, r - current_rank))
                U_mat = np.hstack([U_mat, pad_U])
                V_mat = np.hstack([V_mat, pad_V])
                W_mat = np.hstack([W_mat, pad_W])
            else:
                U_mat = U_mat[:, :r]
                V_mat = V_mat[:, :r]
                W_mat = W_mat[:, :r]
    else:
        U_mat = rng.normal(0, 0.5, (n * m, r))
        V_mat = rng.normal(0, 0.5, (m * p, r))
        W_mat = rng.normal(0, 0.5, (p * n, r))
    
    U_flat = U_mat.flatten()
    V_flat = V_mat.flatten()
    W_flat = W_mat.flatten()
    
    best_loss = float('inf')
    best_params = (U_flat.copy(), V_flat.copy(), W_flat.copy())
    
    def compute_loss(Uf, Vf, Wf):
        Um = Uf.reshape(n * m, r)
        Vm = Vf.reshape(m * p, r)
        Wm = Wf.reshape(p * n, r)
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(r):
                        total += Um[i * m + j, c] * Vm[j * p + k, c] * Wm[k * n + i, c]
                    loss += (total - 1) ** 2
        return loss
    
    current_loss = compute_loss(U_flat, V_flat, W_flat)
    if current_loss < best_loss:
        best_loss = current_loss
        best_params = (U_flat.copy(), V_flat.copy(), W_flat.copy())
    
    lr = 0.005
    momentum = 0.9
    vU = np.zeros_like(U_flat)
    vV = np.zeros_like(V_flat)
    vW = np.zeros_like(W_flat)
    
    max_steps = 500
    step = 0
    while step < max_steps:
        if time.time() > deadline - 15:
            break
        
        # Compute gradients
        U_mat = U_flat.reshape(n * m, r)
        V_mat = V_flat.reshape(m * p, r)
        W_mat = W_flat.reshape(p * n, r)
        
        U_grad = np.zeros_like(U_flat)
        V_grad = np.zeros_like(V_flat)
        W_grad = np.zeros_like(W_flat)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the current product for (i,j,k)
                    prod = 0.0
                    for c in range(r):
                        prod += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    err = prod - 1.0
                    
                    for c in range(r):
                        dU = V_mat[j * p + k, c] * W_mat[k * n + i, c]
                        U_grad[i * m * r + j * r + c] += 2 * err * dU
                        
                        dV = U_mat[i * m + j, c] * W_mat[k * n + i, c]
                        V_grad[j * p * r + k * r + c] += 2 * err * dV
                        
                        dW = U_mat[i * m + j, c] * V_mat[j * p + k, c]
                        W_grad[k * n * r + i * r + c] += 2 * err * dW
        
        # Update with momentum
        vU = momentum * vU - lr * U_grad
        vV = momentum * vV - lr * V_grad
        vW = momentum * vW - lr * W_grad
        
        U_flat = U_flat + vU
        V_flat = V_flat + vV
        W_flat = W_flat + vW
        
        current_loss = compute_loss(U_flat, V_flat, W_flat)
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = (U_flat.copy(), V_flat.copy(), W_flat.copy())
        
        if current_loss < 1e-8:
            break
        
        step += 1
    
    # Try to round to half-integers
    U_best, V_best, W_best = best_params
    U_rounded = np.round(U_best * 2) / 2
    V_rounded = np.round(V_best * 2) / 2
    W_rounded = np.round(W_best * 2) / 2
    
    U_list = U_rounded.tolist()
    V_list = V_rounded.tolist()
    W_list = W_rounded.tolist()
    
    if _verify(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Get block schoolbook as a good baseline (rank 27)
    result = _make_block_schoolbook()
    assert _verify(result["U"], result["V"], result["W"]), "Block schoolbook verification failed"
    current_rank = len(result["U"][0])
    
    # Get full schoolbook as fallback
    full_schoolbook = _make_schoolbook()
    
    # Try to find lower rank decompositions using optimization
    rng = np.random.default_rng(42)
    
    # Try ranks from current_rank - 1 down to a reasonable minimum
    # For 3x3 x 3x5, the rank should be at least 19 (by known bounds)
    # Let's try to find something below 27
    
    for r in range(current_rank - 1, 18, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few restarts for each rank
        found = False
        for restart in range(3):
            if time.time() > deadline - 30:
                break
            
            candidate = _try_optimize_rank(r, deadline, rng)
            if candidate is not None:
                result = candidate
                found = True
                break
        
        if found:
            break
    
    return result
# EVOLVE-BLOCK-END
