import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Build the target tensor indices
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    # Try to find a low-rank decomposition using complex-valued optimization
    # Start with rank 70 and work down
    for target_rank in [70, 72, 74, 76, 78, 80]:
        success = _try_rank_decomposition(nm, mp, pn, target_rank, T_target)
        if success is not None:
            return success
    
    # Fallback to schoolbook rank 80
    return _schoolbook_decomposition(nm, mp, pn, n, m, p)

def _try_rank_decomposition(nm, mp, pn, rank, T_target):
    """
    Try to find a rank decomposition using complex-valued gradient descent
    followed by rational reconstruction.
    """
    np.random.seed(12345)
    
    # Multiple random restarts
    for restart in range(5):
        # Initialize with complex random values
        U = (np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)) * 0.1
        V = (np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)) * 0.1
        W = (np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)) * 0.1
        
        # Flatten parameters for optimization
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def loss_and_grad(params):
            U_flat = params[:nm*rank].reshape(nm, rank)
            V_flat = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W_flat = params[nm*rank+mp*rank:].reshape(pn, rank)
            
            # Compute full tensor
            T_full = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            residual = T_full - T_target
            
            # Loss: squared Frobenius norm
            loss = np.sum(np.abs(residual)**2)
            
            # Gradients (complex)
            dU = 2 * np.einsum('abc,br,cr->ar', residual, np.conj(V_flat), np.conj(W_flat))
            dV = 2 * np.einsum('abc,ar,cr->br', residual, np.conj(U_flat), np.conj(W_flat))
            dW = 2 * np.einsum('abc,ar,br->cr', residual, np.conj(U_flat), np.conj(V_flat))
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            return loss, grad
        
        # Use L-BFGS-B with complex values treated as real/imag pairs
        # Actually, let's use manual gradient descent with adaptive step
        
        lr = 0.01
        best_loss = np.inf
        best_params = params.copy()
        
        for step in range(2000):
            loss, grad = loss_and_grad(params)
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if best_loss < 1e-12:
                break
            
            # Adaptive learning rate
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                params -= lr * grad
            else:
                break
            
            # Decay learning rate
            if step % 200 == 0 and step > 0:
                lr *= 0.95
        
        # Extract best solution
        U_best = best_params[:nm*rank].reshape(nm, rank)
        V_best = best_params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W_best = best_params[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Snap to half-integers
        U_half = np.round(U_best.real * 2) / 2
        V_half = np.round(V_best.real * 2) / 2
        W_half = np.round(W_best.real * 2) / 2
        
        # Verify
        T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        error = np.sum((T_half - T_target)**2)
        
        if error < 1e-10:
            return {
                "U": U_half.tolist(),
                "V": V_half.tolist(),
                "W": W_half.tolist()
            }
        
        # Try integer snapping too
        U_int = np.round(U_best.real)
        V_int = np.round(V_best.real)
        W_int = np.round(W_best.real)
        
        T_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
        error_int = np.sum((T_int - T_target)**2)
        
        if error_int < 1e-10:
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
    
    return None

def _schoolbook_decomposition(nm, mp, pn, n, m, p):
    """Fallback to trivial rank n*m*p decomposition"""
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
