import time
import numpy as np
import random

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    """Build the rank-32 decomposition: 4 Strassen blocks (cols 0-7) + 1 trivial block (col 8)."""
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        # M4
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        # M6
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        # M7
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residuals(U, V, W, tensor_a, tensor_b, tensor_c):
    """Compute residuals for all tensor entries: pred - 1."""
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    # For each entry (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r]
    # Use efficient computation
    rank = U.shape[1]
    # U[a,:] * V[b,:] has shape (N, rank)
    UV = U[tensor_a, :] * V[tensor_b, :]  # (N, rank)
    # Then dot with W[c,:]
    pred = np.einsum('nr,cr->nc', UV, W[tensor_c, :])  # (N, 1)
    pred = pred.squeeze()  # (N,)
    return pred - 1.0  # residuals

def _max_abs_residual(U, V, W, tensor_a, tensor_b, tensor_c):
    """Compute max absolute residual."""
    res = _compute_residuals(U, V, W, tensor_a, tensor_b, tensor_c)
    return np.max(np.abs(res))

def _sum_sq_residual(U, V, W, tensor_a, tensor_b, tensor_c):
    """Compute sum of squared residuals."""
    res = _compute_residuals(U, V, W, tensor_a, tensor_b, tensor_c)
    return np.sum(res ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build rank-32 solution
    U32, V32, W32 = _build_rank32()
    
    # Build tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    tensor_a = np.array([e[0] for e in tensor_entries])
    tensor_b = np.array([e[1] for e in tensor_entries])
    tensor_c = np.array([e[2] for e in tensor_entries])
    
    # Verify rank-32 is correct
    max_res_32 = _max_abs_residual(U32, V32, W32, tensor_a, tensor_b, tensor_c)
    
    # Try to find rank-31 decomposition
    # Strategy: Try dropping each column, then do local search to minimize residual
    # while trying to drive one column toward zero
    
    best_U = U32.tolist()
    best_V = V32.tolist()
    best_W = W32.tolist()
    best_rank = 32
    
    target_rank = 31
    search_budget = 100.0  # seconds for search
    
    # Pre-compute for efficiency
    nm, mp, pn = n * m, m * p, p * n
    
    # Try each column to drop
    for drop_col in range(32):
        if time.time() - start_time > deadline:
            break
        
        # Initialize with rank-32 minus the dropped column
        U_cur = np.delete(U32, drop_col, axis=1).copy()
        V_cur = np.delete(V32, drop_col, axis=1).copy()
        W_cur = np.delete(W32, drop_col, axis=1).copy()
        
        # Check if this is already exact
        max_res = _max_abs_residual(U_cur, V_cur, W_cur, tensor_a, tensor_b, tensor_c)
        if max_res < 1e-10:
            best_U = U_cur.tolist()
            best_V = V_cur.tolist()
            best_W = W_cur.tolist()
            best_rank = target_rank
            break
        
        # Do local search to reduce residual
        # Use a simple gradient-based approach with random restarts
        ss_res = _sum_sq_residual(U_cur, V_cur, W_cur, tensor_a, tensor_b, tensor_c)
        
        # Try a few random perturbations + gradient steps
        for restart in range(3):
            if time.time() - start_time > deadline:
                break
            
            # Add small random perturbation
            if restart > 0:
                U_cur = U_cur + 0.01 * np.random.randn(nm, target_rank)
                V_cur = V_cur + 0.01 * np.random.randn(mp, target_rank)
                W_cur = W_cur + 0.01 * np.random.randn(pn, target_rank)
            
            # Gradient descent with momentum
            lr = 0.005
            momentum = 0.9
            vU = np.zeros_like(U_cur)
            vV = np.zeros_like(V_cur)
            vW = np.zeros_like(W_cur)
            
            for step in range(200):
                if time.time() - start_time > deadline:
                    break
                
                # Compute residuals
                res = _compute_residuals(U_cur, V_cur, W_cur, tensor_a, tensor_b, tensor_c)
                
                # Compute gradients
                # d/dU[a,r] = 2 * sum_{entries with a'=a} res[entry] * V[b,r] * W[c,r]
                # d/dV[b,r] = 2 * sum_{entries with b'=b} res[entry] * U[a,r] * W[c,r]
                # d/dW[c,r] = 2 * sum_{entries with c'=c} res[entry] * U[a,r] * V[b,r]
                
                # Efficient gradient computation
                # For U: for each a, sum over entries where tensor_a == a
                grad_U = np.zeros_like(U_cur)
                grad_V = np.zeros_like(V_cur)
                grad_W = np.zeros_like(W_cur)
                
                # res has shape (N,)
                # For each entry idx: a=tensor_a[idx], b=tensor_b[idx], c=tensor_c[idx]
                # grad_U[a, r] += 2 * res[idx] * V[b, r] * W[c, r]
                
                # Vectorized: 
                # UV_prod = U[a,:] * V[b,:] -> (N, rank)
                # grad_U[a, r] = 2 * sum_{idx: tensor_a[idx]==a} res[idx] * V[tensor_b[idx], r] * W[tensor_c[idx], r]
                
                # This is expensive to do fully vectorized for small N, so do it directly
                for idx in range(len(tensor_a)):
                    a = tensor_a[idx]
                    b = tensor_b[idx]
                    c = tensor_c[idx]
                    r = res[idx]
                    if abs(r) > 1e-12:
                        grad_U[a, :] += 2 * r * V_cur[b, :] * W_cur[c, :]
                        grad_V[b, :] += 2 * r * U_cur[a, :] * W_cur[c, :]
                        grad_W[c, :] += 2 * r * U_cur[a, :] * V_cur[b, :]
                
                # Update with momentum
                vU = momentum * vU + lr * grad_U
                vV = momentum * vV + lr * grad_V
                vW = momentum * vW + lr * grad_W
                
                U_cur -= vU
                V_cur -= vV
                W_cur -= vW
                
                # Check convergence
                if step % 20 == 0:
                    max_res = _max_abs_residual(U_cur, V_cur, W_cur, tensor_a, tensor_b, tensor_c)
                    if max_res < 1e-10:
                        best_U = U_cur.tolist()
                        best_V = V_cur.tolist()
                        best_W = W_cur.tolist()
                        best_rank = target_rank
                        break
                    ss_res = _sum_sq_residual(U_cur, V_cur, W_cur, tensor_a, tensor_b, tensor_c)
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a rank-31 solution, verify it
    if best_rank == target_rank:
        U_final = np.array(best_U)
        V_final = np.array(best_V)
        W_final = np.array(best_W)
        max_res_final = _max_abs_residual(U_final, V_final, W_final, tensor_a, tensor_b, tensor_c)
        if max_res_final > 1e-8:
            # Fall back to rank-32
            best_U = U32.tolist()
            best_V = V32.tolist()
            best_W = W32.tolist()
            best_rank = 32
    
    return {"U": best_U, "V": best_V, "W": best_W}
