import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_sb = n * m * p  # 60
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1
                    val = np.einsum('c,c,c->', U_r[i,j], V_r[j,k], W_r[k,i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(X):
        return np.round(X * 2) / 2.0
    
    # Try to find lower rank decompositions
    # Focus on ranks that are plausible improvements
    # For 3x4x5, try ranks from 50 to 59
    
    np.random.seed(42)
    
    for target_rank in range(50, min(60, rank_sb)):
        if time.time() > deadline:
            break
        
        # Multiple restarts for each rank
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random values scaled appropriately
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Simple gradient descent to minimize ||T_approx - T_target||^2
            # T_target[i,j,k] = 1 for all valid indices
            lr = 0.01
            max_iter = 100
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                # Compute T_approx
                T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                residual = T_approx - 1.0
                
                # Check if we're close enough
                if np.max(np.abs(residual)) < 1e-8:
                    break
                
                # Compute gradients
                # dT/dU_r[i,j,c] = V_r[j,k] * W_r[k,i] summed over k
                grad_U = np.einsum('jkc,kic->ijc', V_r, W_r) * residual[:, :, :]
                # Actually need to be more careful
                # T_approx[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
                # dT[i,j,k]/dU_r[i',j',c'] = delta(i,i')delta(j,j') V_r[j,k,c'] W_r[k,i,c']
                
                # Gradient for U
                grad_U = np.zeros_like(U_r)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            grad_U[i,j] += residual[i,j,k] * np.outer(V_r[j,k], W_r[k,i]) / target_rank
                
                # Gradient for V  
                grad_V = np.zeros_like(V_r)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            grad_V[j,k] += residual[i,j,k] * np.outer(U_r[i,j], W_r[k,i]) / target_rank
                
                # Gradient for W
                grad_W = np.zeros_like(W_r)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            grad_W[k,i] += residual[i,j,k] * np.outer(U_r[i,j], V_r[j,k]) / target_rank
                
                # Update
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
                
                U = U_r.reshape(nm, target_rank)
                V = V_r.reshape(mp, target_rank)
                W = W_r.reshape(pn, target_rank)
            
            # After optimization, try rounding to half-integers
            U_rounded = round_to_half(U)
            V_rounded = round_to_half(V)
            W_rounded = round_to_half(W)
            
            if verify(U_rounded, V_rounded, W_rounded):
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                best_rank = target_rank
                break  # Found a valid decomposition at this rank
        
        if best_rank < rank_sb:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_sb:
        U_out, V_out, W_out = best_U, best_V, best_W
    else:
        U_out, V_out, W_out = U_sb, V_sb, W_sb
    
    # Convert to lists
    U_list = [[float(U_out[i, j]) for j in range(U_out.shape[1])] for i in range(U_out.shape[0])]
    V_list = [[float(V_out[i, j]) for j in range(V_out.shape[1])] for i in range(V_out.shape[0])]
    W_list = [[float(W_out[i, j]) for j in range(W_out.shape[1])] for i in range(W_out.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
