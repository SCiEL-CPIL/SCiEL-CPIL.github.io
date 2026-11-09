"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

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
    return {"U": U, "V": V, "W": W}

def _compute_error_vectorized(U, V, W):
    """Compute the max error using vectorized operations."""
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    U_sel = U_np[idx_u, :]
    V_sel = V_np[idx_v, :]
    W_sel = W_np[idx_w, :]
    
    prod = U_sel * V_sel * W_sel
    vals = prod.sum(axis=1)
    
    err = np.abs(vals - 1.0)
    return float(err.max()) if err.size > 0 else 0.0

def _err_fast(U, V, W, idx_u, idx_v, idx_w):
    """Fast error computation."""
    U_sel = U[idx_u, :]
    V_sel = V[idx_v, :]
    W_sel = W[idx_w, :]
    vals = (U_sel * V_sel * W_sel).sum(axis=1)
    return float(np.max(np.abs(vals - 1.0)))

def _structured_init(target_rank, n, m, p, attempt):
    """Generate structured initializations that leverage Kronecker/block structure."""
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    if attempt == 0:
        # Block-structured: group by i, then j, then k
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n*m)) % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
    elif attempt == 1:
        # Kronecker-inspired: separate row and column structure
        # Think of U as (I_n ⊗ something), V as (something ⊗ I_p)
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n*m)) % p
            # Scale by block structure
            U[i*m+j, c] = 1.0 + 0.5*((j % 2))
            V[j*p+k, c] = 1.0 + 0.5*((k % 2))
            W[k*n+i, c] = 1.0
    elif attempt == 2:
        # Diagonal blocks: emphasize diagonal entries
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n*m)) % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
            # Add off-diagonal contributions
            if i + 1 < n:
                U[(i+1)*m+j, c] += 0.3
            if j + 1 < m:
                U[i*m+j+1, c] += 0.3
            if k + 1 < p:
                V[j*p+k+1, c] += 0.3
    elif attempt == 3:
        # Two-scale Kronecker: mix fine and coarse structure
        half = target_rank // 2
        for c in range(half):
            i = c % n
            j = (c // n) % m
            k = (c // (n*m)) % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        for c in range(half, target_rank):
            c2 = c - half
            i = c2 % n
            j = (c2 // n) % m
            k = (c2 // (n*m)) % p
            # Coarser grouping
            U[i*m+j, c] = 0.7
            V[j*p+k, c] = 0.7
            W[k*n+i, c] = 0.7
            # Cross terms
            if i < n-1:
                U[(i+1)*m+j, c] += 0.2
            if j < m-1:
                U[i*m+j+1, c] += 0.2
    elif attempt == 4:
        # Random structured: random assignment but with block bias
        rng = np.random.RandomState(attempt * 1000 + target_rank)
        for c in range(target_rank):
            # Bias towards structured positions
            r = rng.random()
            if r < 0.6:
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            else:
                ui = rng.randint(0, n*m)
                vi = rng.randint(0, m*p)
                wi = rng.randint(0, p*n)
                U[ui, c] = 1.0
                V[vi, c] = 1.0
                W[wi, c] = 1.0
    elif attempt == 5:
        # Symmetric block: try to exploit symmetry
        for c in range(target_rank):
            i = c % n
            j = (c // n) % m
            k = (c // (n*m)) % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
            # Add symmetric partner
            if c + n < target_rank:
                i2 = (c + n) % n
                j2 = ((c + n) // n) % m
                k2 = ((c + n) // (n*m)) % p
                U[i2*m+j2, c] += 0.4
                V[j2*p+k2, c] += 0.4
                W[k2*n+i2, c] += 0.4
    else:
        # Pure random
        rng = np.random.RandomState(attempt * 7919 + target_rank * 31)
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5
    
    return U, V, W

def _polish_decomposition(U, V, W, idx_u, idx_v, idx_w, deadline=None, max_iter=300):
    """High-precision polish pass with adaptive step size."""
    cur_err = _err_fast(U, V, W, idx_u, idx_v, idx_w)
    lr = 0.05
    patience = 0
    best_err = cur_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        if cur_err < 1e-13:
            break
        
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        vals = (U_sel * V_sel * W_sel).sum(axis=1)
        res = vals - 1.0
        
        VW = V_sel * W_sel
        UW = U_sel * W_sel
        UV = U_sel * V_sel
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        np.add.at(dU, idx_u, res[:, None] * VW)
        np.add.at(dV, idx_v, res[:, None] * UW)
        np.add.at(dW, idx_w, res[:, None] * UV)
        
        improved = False
        for alpha in [lr, lr*0.7, lr*0.5, lr*0.35, lr*0.25, lr*0.15, lr*0.1]:
            U_new = U - alpha * dU
            V_new = V - alpha * dV
            W_new = W - alpha * dW
            new_err = _err_fast(U_new, V_new, W_new, idx_u, idx_v, idx_w)
            if new_err < cur_err * 0.999:
                U, V, W = U_new, V_new, W_new
                cur_err = new_err
                improved = True
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not improved:
            lr *= 0.5
            patience += 1
            if lr < 1e-10 or patience > 10:
                break
        else:
            patience = 0
            if cur_err < 0.1 * best_err:
                lr = min(lr * 1.1, 0.2)
        
        if it % 40 == 0 and it > 0:
            lr *= 0.9
    
    return best_U, best_V, best_W, best_err

def _try_rank_decomposition(target_rank, num_attempts=10, max_iter=400, deadline=None):
    """Try to find a rank decomposition using structured initializations and gradient descent."""
    n, m, p = DIMENSIONS
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 15:
            return None
        
        U, V, W = _structured_init(target_rank, n, m, p, attempt)
        
        # Add small random perturbation to break symmetry
        rng = np.random.RandomState(attempt * 12345 + target_rank * 67)
        U += rng.randn(n*m, target_rank) * 0.1
        V += rng.randn(m*p, target_rank) * 0.1
        W += rng.randn(p*n, target_rank) * 0.1
        
        lr = 0.03
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if deadline and time.time() > deadline - 15:
                break
            
            U_sel = U[idx_u, :]
            V_sel = V[idx_v, :]
            W_sel = W[idx_w, :]
            
            prod = U_sel * V_sel * W_sel
            vals = prod.sum(axis=1)
            residuals = vals - 1.0
            
            VW = V_sel * W_sel
            UW = U_sel * W_sel
            UV = U_sel * V_sel
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, idx_u, residuals[:, None] * VW)
            np.add.at(dV, idx_v, residuals[:, None] * UW)
            np.add.at(dW, idx_w, residuals[:, None] * UV)
            
            improved = False
            for alpha in [lr, lr*0.7, lr*0.5, lr*0.35, lr*0.25]:
                U_new = U - alpha * dU
                V_new = V - alpha * dV
                W_new = W - alpha * dW
                
                U_sel_new = U_new[idx_u, :]
                V_sel_new = V_new[idx_v, :]
                W_sel_new = W_new[idx_w, :]
                new_vals = (U_sel_new * V_sel_new * W_sel_new).sum(axis=1)
                new_err = float(np.max(np.abs(new_vals - 1.0)))
                
                if new_err < best_err:
                    U, V, W = U_new, V_new, W_new
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    improved = True
                    break
            
            if not improved:
                lr *= 0.5
                if lr < 1e-9:
                    break
            
            if best_err < 1e-10:
                break
            
            if iteration % 60 == 0 and iteration > 0:
                lr *= 0.88
        
        if best_err < 1e-3:
            U, V, W, final_err = _polish_decomposition(best_U, best_V, best_W, idx_u, idx_v, idx_w, deadline, max_iter=200)
            best_err = final_err
            best_U, best_V, best_W = U, V, W
        
        if best_err < 1e-7:
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            err_rounded = _compute_error_vectorized(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-9:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from a lower bound upward
    # For 4x5x6, known bounds suggest rank >= ~50
    for target_rank in range(50, 120):
        if time.time() > deadline - 20:
            break
        
        if target_rank < 70:
            num_attempts = 8
            max_iter = 350
        else:
            num_attempts = 5
            max_iter = 300
        
        result = _try_rank_decomposition(target_rank, num_attempts=num_attempts, max_iter=max_iter, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
