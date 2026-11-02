"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _schoolbook(n, m, p):
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
    return U, V, W

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _tensor_error(U_np, V_np, W_np, T):
    M1 = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]
    R = np.einsum('abr,cr->abc', M1, W_np)
    err = np.sum((R - T) ** 2)
    return err

def _snap_to_half_int(U, V, W):
    U = [[round(x * 2) / 2 for x in row] for row in U]
    V = [[round(x * 2) / 2 for x in row] for row in V]
    W = [[round(x * 2) / 2 for x in row] for row in W]
    return U, V, W

def _try_polish_gradient(U, V, W, T, rank, deadline, initial_err):
    """
    Gradient-based local search refinement.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    best_err = initial_err
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    lr = 0.1
    momentum = 0.9
    vU = np.zeros_like(U_np)
    vV = np.zeros_like(V_np)
    vW = np.zeros_like(W_np)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute gradients
        M1 = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]  # (nm, mp, rank)
        R = np.einsum('abr,cr->abc', M1, W_np)                # (nm, mp, pn)
        residual = R - T  # (nm, mp, pn)
        
        # Gradient of error w.r.t. U[a, r]: 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
        # grad_U[a, r] = 2 * sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
        # = 2 * (residual * V[:, :, np.newaxis] * W[np.newaxis, :, np.newaxis]).sum over b,c... 
        # Actually: grad_U[a, r] = 2 * sum_b sum_c residual[a,b,c] V[b,r] W[c,r]
        # Let's compute: temp[b, c, r] = V[b,r] * W[c,r]
        # Then grad_U[a, r] = 2 * sum_{b,c} residual[a,b,c] * temp[b,c,r]
        # = 2 * einsum('abc,bcr->ar', residual, V[:, np.newaxis, :] * W[np.newaxis, :, :])
        
        temp_VW = V_np[:, np.newaxis, :] * W_np[np.newaxis, :, :]  # (mp, pn, rank)
        grad_U = 2.0 * np.einsum('abc,bcr->ar', residual, temp_VW)
        
        # grad_V[b, r] = 2 * sum_{a,c} residual[a,b,c] * U[a,r] * W[c,r]
        temp_UW = U_np[:, np.newaxis, :] * W_np[np.newaxis, :, :]  # (nm, pn, rank)
        grad_V = 2.0 * np.einsum('abc,acr->br', residual, temp_UW)
        
        # grad_W[c, r] = 2 * sum_{a,b} residual[a,b,c] * U[a,r] * V[b,r]
        temp_UV = U_np[:, np.newaxis, :] * V_np[np.newaxis, :, :]  # (nm, mp, rank)
        grad_W = 2.0 * np.einsum('abc,abr->cr', residual, temp_UV)
        
        # Update with momentum
        vU = momentum * vU - lr * grad_U
        vV = momentum * vV - lr * grad_V
        vW = momentum * vW - lr * grad_W
        
        U_np = U_np + vU
        V_np = V_np + vV
        W_np = W_np + vW
        
        new_err = _tensor_error(U_np, V_np, W_np, T)
        
        if new_err < best_err:
            best_err = new_err
            best_U = U_np.copy()
            best_V = V_np.copy()
            best_W = W_np.copy()
            
            if best_err < 1e-10:
                # Try snapping
                U_snapped, V_snapped, W_snapped = _snap_to_half_int(
                    best_U.tolist(), best_V.tolist(), best_W.tolist()
                )
                U_s_np = np.array(U_snapped)
                V_s_np = np.array(V_snapped)
                W_s_np = np.array(W_snapped)
                if _tensor_error(U_s_np, V_s_np, W_s_np, T) < 1e-10:
                    return U_snapped, V_snapped, W_snapped, True
        else:
            # Revert if error increased too much
            if new_err > best_err * 1.1:
                U_np = best_U.copy()
                V_np = best_V.copy()
                W_np = best_W.copy()
                vU *= 0.5
                vV *= 0.5
                vW *= 0.5
                lr *= 0.9
        
        # Decay learning rate slowly
        lr *= 0.9995
    
    # Final snap attempt
    U_snapped, V_snapped, W_snapped = _snap_to_half_int(
        best_U.tolist(), best_V.tolist(), best_W.tolist()
    )
    U_s_np = np.array(U_snapped)
    V_s_np = np.array(V_snapped)
    W_s_np = np.array(W_snapped)
    if _tensor_error(U_s_np, V_s_np, W_s_np, T) < 1e-10:
        return U_snapped, V_snapped, W_snapped, True
    
    return U_snapped, V_snapped, W_snapped, False

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    T = _build_tensor(n, m, p)
    
    # Try ranks from 24 down to 20
    for target_rank in [24, 23, 22, 21, 20]:
        if time.time() > deadline:
            break
        
        # Initialize with schoolbook truncated to target_rank
        U, V, W = _schoolbook(n, m, p)
        U = [row[:target_rank] for row in U]
        V = [row[:target_rank] for row in V]
        W = [row[:target_rank] for row in W]
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        initial_err = _tensor_error(U_np, V_np, W_np, T)
        
        # If rank is 24, it's already exact
        if initial_err < 1e-10:
            return {"U": U, "V": V, "W": W}
        
        # Try gradient-based refinement
        U_out, V_out, W_out, success = _try_polish_gradient(
            U, V, W, T, target_rank, deadline, initial_err
        )
        
        if success:
            return {"U": U_out, "V": V_out, "W": W_out}
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
