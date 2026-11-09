import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _solve_linear_system(A, b):
    if A.shape[1] == 0:
        return np.zeros(0)
    x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return x

def _als_iteration(U, V, W, T, num_a, num_b, num_c, rank):
    new_U = np.zeros_like(U)
    for a in range(num_a):
        M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_b * num_c, rank)
        b_vec = T[a, :, :].reshape(-1)
        new_U[a, :] = _solve_linear_system(M_a, b_vec)
    U = new_U
    
    new_V = np.zeros_like(V)
    for b in range(num_b):
        M_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(num_a * num_c, rank)
        b_vec = T[:, b, :].reshape(-1)
        new_V[b, :] = _solve_linear_system(M_b, b_vec)
    V = new_V
    
    new_W = np.zeros_like(W)
    for c in range(num_c):
        M_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(num_a * num_b, rank)
        b_vec = T[:, :, c].reshape(-1)
        new_W[c, :] = _solve_linear_system(M_c, b_vec)
    W = new_W
    
    return U, V, W

def _gradient_polish(U, V, W, T, num_a, num_b, num_c, rank, n_iter=50, lr=0.01):
    def loss_and_grad(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_recon - T
        loss = np.sum(diff ** 2)
        
        grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        return loss, grad_U, grad_V, grad_W
    
    for it in range(n_iter):
        loss, gU, gV, gW = loss_and_grad(U, V, W)
        if loss < 1e-15:
            break
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
    
    return U, V, W

def _snap_to_rationals(U, V, W, max_denom=4):
    """
    Try to snap the entries of U, V, W to rational numbers with small denominators.
    This is a heuristic to recover exact solutions from floating point approximations.
    """
    def snap_matrix(M):
        M_snapped = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                # Check integers
                if abs(val - round(val)) < 1e-6:
                    M_snapped[i, j] = round(val)
                    continue
                
                # Check halves
                if abs(val * 2 - round(val * 2)) < 1e-5:
                    M_snapped[i, j] = round(val * 2) / 2.0
                    continue
                
                # Check quarters
                if abs(val * 4 - round(val * 4)) < 1e-4:
                    M_snapped[i, j] = round(val * 4) / 4.0
                    continue
                
                # Keep original if no good snap found
                M_snapped[i, j] = val
        return M_snapped

    U_s = _snap_matrix(U)
    V_s = _snap_matrix(V)
    W_s = _snap_matrix(W)
    
    # Verify if the snapped version is still a valid solution
    T_recon = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
    if np.allclose(T_recon, T, atol=1e-9):
        return U_s, V_s, W_s
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    num_a = n * m  # 6
    num_b = m * p  # 21
    num_c = p * n  # 14
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank as fallback
    
    # Theoretical lower bound is max(num_a, num_b, num_c) = 21
    # We search from a slightly lower bound to be safe, but realistically 21-25 is the sweet spot
    # Known results for 2x3x7 are around 24-26. Let's search 21 to 30.
    
    min_rank = 21
    max_rank_to_try = 30
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 15:
            break
        
        found = False
        num_restarts = 20 # Increase restarts for lower ranks
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = 42 + restart * 1000 + rank
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(num_a, rank) * 0.1
            V = np.random.randn(num_b, rank) * 0.1
            W = np.random.randn(num_c, rank) * 0.1
            
            prev_loss = np.inf
            stalled = False
            
            for it in range(100):
                if time.time() > deadline - 8:
                    break
                
                U, V, W = _als_iteration(U, V, W, T, num_a, num_b, num_c, rank)
                
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                loss = np.sum((T_recon - T) ** 2)
                
                if loss < 1e-12:
                    # Found a solution! Try to snap to rationals
                    snapped = _snap_to_rationals(U, V, W)
                    if snapped is not None:
                        U, V, W = snapped
                    found = True
                    break
                
                if abs(prev_loss - loss) < 1e-14:
                    # Stuck, try gradient polish
                    U, V, W = _gradient_polish(U, V, W, T, num_a, num_b, num_c, rank, n_iter=50, lr=0.005)
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    loss = np.sum((T_recon - T) ** 2)
                    if loss < 1e-12:
                        snapped = _snap_to_rationals(U, V, W)
                        if snapped is not None:
                            U, V, W = snapped
                        found = True
                        break
                    # If still stuck after polish, mark as stalled
                    stalled = True
                    break
                
                prev_loss = loss
            
            if found:
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if found:
            break
    
    # If we didn't find a better rank, use schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((num_a, rank))
        V = np.zeros((num_b, rank))
        W = np.zeros((num_c, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    U, V, W = best_result
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
