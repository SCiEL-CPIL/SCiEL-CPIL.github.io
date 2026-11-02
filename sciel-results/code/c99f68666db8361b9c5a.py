import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Create the sparse tensor indices
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
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    # Try multiple ranks starting from a lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = 20
    # But border rank heuristics suggest trying around 30-50
    ranks_to_try = [25, 30, 35, 40, 45, 50]
    
    best_result = None
    best_rank = float('inf')
    
    for rank in ranks_to_try:
        np.random.seed(rank)  # Different seed for each rank
        
        # Initialize with complex values for border rank heuristic
        # Use small random complex values to avoid singularities
        U = np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)
        V = np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)
        W = np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)
        
        # Scale to appropriate magnitude
        U = U / np.sqrt(rank)
        V = V / np.sqrt(rank)
        W = W / np.sqrt(rank)
        
        # Perform gradient descent on the real and imaginary parts
        # Flatten the complex matrices
        U_real = U.real.copy()
        U_imag = U.imag.copy()
        V_real = V.real.copy()
        V_imag = V.imag.copy()
        W_real = W.real.copy()
        W_imag = W.imag.copy()
        
        # Define objective function
        def objective(params):
            U_r = params[:nm*rank].reshape(nm, rank)
            U_i = params[nm*rank:2*nm*rank].reshape(nm, rank)
            V_r = params[2*nm*rank:2*nm*rank+mp*rank].reshape(mp, rank)
            V_i = params[2*nm*rank+mp*rank:2*nm*rank+2*mp*rank].reshape(mp, rank)
            W_r = params[2*nm*rank+2*mp*rank:2*nm*rank+2*mp*rank+pn*rank].reshape(pn, rank)
            W_i = params[2*nm*rank+2*mp*rank+pn*rank:].reshape(pn, rank)
            
            U_c = U_r + 1j * U_i
            V_c = V_r + 1j * V_i
            W_c = W_r + 1j * W_i
            
            # Compute tensor
            T_full = np.einsum('ar,br,cr->abc', U_c, V_c, W_c)
            
            # Residual (take real part since target is real)
            residual = T_full.real - T_target
            
            # Also penalize imaginary part to keep it small
            imag_penalty = np.sum(T_full.imag**2)
            
            return np.sum(residual**2) + 0.1 * imag_penalty
        
        def gradient(params):
            U_r = params[:nm*rank].reshape(nm, rank)
            U_i = params[nm*rank:2*nm*rank].reshape(nm, rank)
            V_r = params[2*nm*rank:2*nm*rank+mp*rank].reshape(mp, rank)
            V_i = params[2*nm*rank+mp*rank:2*nm*rank+2*mp*rank].reshape(mp, rank)
            W_r = params[2*nm*rank+2*mp*rank:2*nm*rank+2*mp*rank+pn*rank].reshape(pn, rank)
            W_i = params[2*nm*rank+2*mp*rank+pn*rank:].reshape(pn, rank)
            
            U_c = U_r + 1j * U_i
            V_c = V_r + 1j * V_i
            W_c = W_r + 1j * W_i
            
            T_full = np.einsum('ar,br,cr->abc', U_c, V_c, W_c)
            residual = T_full.real - T_target
            imag_part = T_full.imag
            
            # Gradient with respect to real parts
            dU_r = 2 * np.einsum('abc,br,cr->ar', residual, V_c.real, W_c.real) - 2 * np.einsum('abc,br,cr->ar', residual, V_c.imag, W_c.imag)
            dU_i = 2 * np.einsum('abc,br,cr->ar', residual, V_c.real, W_c.imag) + 2 * np.einsum('abc,br,cr->ar', residual, V_c.imag, W_c.real)
            
            dV_r = 2 * np.einsum('abc,ar,cr->br', residual, U_c.real, W_c.real) - 2 * np.einsum('abc,ar,cr->br', residual, U_c.imag, W_c.imag)
            dV_i = 2 * np.einsum('abc,ar,cr->br', residual, U_c.real, W_c.imag) + 2 * np.einsum('abc,ar,cr->br', residual, U_c.imag, W_c.real)
            
            dW_r = 2 * np.einsum('abc,ar,br->cr', residual, U_c.real, V_c.real) - 2 * np.einsum('abc,ar,br->cr', residual, U_c.imag, V_c.imag)
            dW_i = 2 * np.einsum('abc,ar,br->cr', residual, U_c.real, V_c.imag) + 2 * np.einsum('abc,ar,br->cr', residual, U_c.imag, V_c.real)
            
            # Add imaginary penalty gradients
            dU_r += 0.1 * 2 * np.einsum('abc,br,cr->ar', imag_part, V_c.real, W_c.real) - 0.1 * 2 * np.einsum('abc,br,cr->ar', imag_part, V_c.imag, W_c.imag)
            dU_i += 0.1 * 2 * np.einsum('abc,br,cr->ar', imag_part, V_c.real, W_c.imag) + 0.1 * 2 * np.einsum('abc,br,cr->ar', imag_part, V_c.imag, W_c.real)
            
            dV_r += 0.1 * 2 * np.einsum('abc,ar,cr->br', imag_part, U_c.real, W_c.real) - 0.1 * 2 * np.einsum('abc,ar,cr->br', imag_part, U_c.imag, W_c.imag)
            dV_i += 0.1 * 2 * np.einsum('abc,ar,cr->br', imag_part, U_c.real, W_c.imag) + 0.1 * 2 * np.einsum('abc,ar,cr->br', imag_part, U_c.imag, W_c.real)
            
            dW_r += 0.1 * 2 * np.einsum('abc,ar,br->cr', imag_part, U_c.real, V_c.real) - 0.1 * 2 * np.einsum('abc,ar,br->cr', imag_part, U_c.imag, V_c.imag)
            dW_i += 0.1 * 2 * np.einsum('abc,ar,br->cr', imag_part, U_c.real, V_c.imag) + 0.1 * 2 * np.einsum('abc,ar,br->cr', imag_part, U_c.imag, V_c.real)
            
            return np.concatenate([dU_r.flatten(), dU_i.flatten(), dV_r.flatten(), dV_i.flatten(), dW_r.flatten(), dW_i.flatten()])
        
        # Initial parameters
        initial_params = np.concatenate([U_real.flatten(), U_imag.flatten(), V_real.flatten(), V_imag.flatten(), W_real.flatten(), W_imag.flatten()])
        
        # Use L-BFGS-B optimizer
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B', jac=gradient, 
                            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < 1e-6:
                # Extract optimized complex matrices
                U_r = result.x[:nm*rank].reshape(nm, rank)
                U_i = result.x[nm*rank:2*nm*rank].reshape(nm, rank)
                V_r = result.x[2*nm*rank:2*nm*rank+mp*rank].reshape(mp, rank)
                V_i = result.x[2*nm*rank+mp*rank:2*nm*rank+2*mp*rank].reshape(mp, rank)
                W_r = result.x[2*nm*rank+2*mp*rank:2*nm*rank+2*mp*rank+pn*rank].reshape(pn, rank)
                W_i = result.x[2*nm*rank+2*mp*rank+pn*rank:].reshape(pn, rank)
                
                U_opt = U_r + 1j * U_i
                V_opt = V_r + 1j * V_i
                W_opt = W_r + 1j * W_i
                
                # Take real parts for final answer
                U_final = U_opt.real
                V_final = V_opt.real
                W_final = W_opt.real
                
                # Rational reconstruction: snap to nearest half-integers
                U_half = np.round(U_final * 2) / 2
                V_half = np.round(V_final * 2) / 2
                W_half = np.round(W_final * 2) / 2
                
                # Verify
                T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                error = np.sum((T_half - T_target)**2)
                
                if error < 1e-8:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                        
        except Exception as e:
            continue
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
