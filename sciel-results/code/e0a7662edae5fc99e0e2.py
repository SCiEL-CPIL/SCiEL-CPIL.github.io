import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Indices for full problem
    # A: 3x3 -> 9
    # B: 3x4 -> 12
    # C: 3x4 -> 12 (gamma = 3k+i)
    
    # We will try to construct a rank 33 solution.
    # Part 1: M_3,3,3 (k=0,1,2) with rank 24
    # Part 2: M_3,3,1 (k=3) with rank 9
    
    # First, find a rank 24 decomposition for M_3,3,3
    # Indices for M_3,3,3:
    # alpha = 3i + j
    # beta = 4j + k (k in 0,1,2)
    # gamma = 3k + i
    
    def get_m333_indices():
        valid = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    alpha = 3 * i + j
                    beta = 4 * j + k
                    gamma = 3 * k + i
                    valid.append((alpha, beta, gamma))
        return valid

    def get_m331_indices():
        valid = []
        for i in range(3):
            for j in range(3):
                k = 3
                alpha = 3 * i + j
                beta = 4 * j + k
                gamma = 3 * k + i
                valid.append((alpha, beta, gamma))
        return valid

    def optimize_m333(rank):
        valid_indices = get_m333_indices()
        total_alpha = 9
        total_beta = 12
        total_gamma = 9 # Only k=0,1,2 so gamma 0..8
        
        # We only care about beta indices 0,1,2,4,5,6,8,9,10
        # And gamma indices 0..8
        
        def loss_and_grad(params, U_shape, V_shape, W_shape):
            num_u = U_shape[0] * rank
            num_v = V_shape[0] * rank
            num_w = W_shape[0] * rank
            
            U_flat = params[:num_u]
            V_flat = params[num_u:num_u+num_v]
            W_flat = params[num_u+num_v:num_u+num_v+num_w]
            
            U = U_flat.reshape(U_shape)
            V = V_flat.reshape(V_shape)
            W = W_flat.reshape(W_shape)
            
            T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
            
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, g) in valid_indices:
                val = T_approx[a, b, g]
                err = val - 1.0
                loss += err * err
                grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
                grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
                grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
            
            # Zero out invalid indices
            # We need to define which indices are invalid for the M_3,3,3 sub-problem
            # Valid alphas: 0..8
            # Valid betas: 0,1,2,4,5,6,8,9,10
            # Valid gammas: 0..8
            # Invalid are all others in the 9x12x9 space? 
            # Actually, we just need to ensure T_approx is 0 where it shouldn't be.
            # For efficiency, we can just check the valid ones and assume the rest are 0 
            # if the rank is low enough? No, we must constrain them.
            
            # Iterate over all possible indices in the sub-space
            for a in range(9):
                for b in [0,1,2,4,5,6,8,9,10]:
                    for g in range(9):
                        if (a,b,g) not in valid_indices:
                            val = T_approx[a, b, g]
                            err = val
                            loss += err * err
                            grad_U[a, :] += 2.0 * err * V[b, :] * W[g, :]
                            grad_V[b, :] += 2.0 * err * U[a, :] * W[g, :]
                            grad_W[g, :] += 2.0 * err * U[a, :] * V[b, :]
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        U_shape = (9, rank)
        V_shape = (12, rank)
        W_shape = (9, rank)
        
        best_U, best_V, best_W = None, None, None
        best_loss = float('inf')
        
        start_time = time.time()
        deadline = start_time + 100
        
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randn(*U_shape) * 0.5
            V = np.random.randn(*V_shape) * 0.5
            W = np.random.randn(*W_shape) * 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(loss_and_grad, params, args=(U_shape, V_shape, W_shape), 
                               jac=True, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': 2000})
            except Exception:
                continue
                
            if res.fun < best_loss:
                best_loss = res.fun
                num_u = U_shape[0] * rank
                num_v = V_shape[0] * rank
                best_U = res.x[:num_u].reshape(U_shape)
                best_V = res.x[num_u:num_u+num_v].reshape(V_shape)
                best_W = res.x[num_u+num_v:num_u+num_v+num_w].reshape(W_shape)
                
                # Try to snap
                U_s = np.round(best_U * 2) / 2
                V_s = np.round(best_V * 2) / 2
                W_s = np.round(best_W * 2) / 2
                
                # Verify snap
                T_approx = np.einsum('ac,bc,dc->abd', U_s, V_s, W_s)
                is_valid = True
                for (a, b, g) in valid_indices:
                    if not np.isclose(T_approx[a, b, g], 1.0, atol=1e-6):
                        is_valid = False
                        break
                if is_valid:
                    for a in range(9):
                        for b in [0,1,2,4,5,6,8,9,10]:
                            for g in range(9):
                                if (a,b,g) not in valid_indices:
                                    if not np.isclose(T_approx[a, b, g], 0.0, atol=1e-6):
                                        is_valid = False
                                        break
                            if not is_valid: break
                        if not is_valid: break
                    if is_valid:
                        return U_s, V_s, W_s
                        
        return None

    # Try to find rank 24 for M_3,3,3
    res_24 = optimize_m333(24)
    
    if res_24 is not None:
        U_part1, V_part1, W_part1 = res_24
        # Construct Part 2 (M_3,3,1)
        U_part2 = np.zeros((9, 9))
        V_part2 = np.zeros((12, 9))
        W_part2 = np.zeros((12, 9)) # Note: W shape for full problem is 12xR, but here only gamma 9,10,11 are used
        
        col = 0
        for i in range(3):
            for j in range(3):
                alpha = 3 * i + j
                beta = 4 * j + 3
                gamma = 3 * 3 + i
                U_part2[alpha, col] = 1.0
                V_part2[beta, col] = 1.0
                W_part2[gamma, col] = 1.0
                col += 1
        
        # Combine
        U_full = np.hstack([U_part1, U_part2])
        V_full = np.hstack([V_part1, V_part2])
        # W_part1 has shape 9x24. We need to embed it into 12x33.
        W_full = np.zeros((12, 33))
        W_full[:9, :24] = W_part1
        W_full[:, 24:] = W_part2
        
        # Final Verification
        T_approx = np.einsum('ac,bc,dc->abd', U_full, V_full, W_full)
        
        valid_full = []
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    alpha = 3 * i + j
                    beta = 4 * j + k
                    gamma = 3 * k + i
                    valid_full.append((alpha, beta, gamma))
                    
        is_valid = True
        for (a, b, g) in valid_full:
            if not np.isclose(T_approx[a, b, g], 1.0, atol=1e-6):
                is_valid = False
                break
                
        if is_valid:
            # Check invalids
            valid_set = set(valid_full)
            for a in range(9):
                for b in range(12):
                    for g in range(12):
                        if (a,b,g) not in valid_set:
                            if not np.isclose(T_approx[a, b, g], 0.0, atol=1e-6):
                                is_valid = False
                                break
                    if not is_valid: break
                if not is_valid: break
                
            if is_valid:
                return {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }

    # Fallback to schoolbook
    rank = 36
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    column = 0
    for i in range(3):
        for j in range(3):
            for k in range(4):
                alpha = 3 * i + j
                beta = 4 * j + k
                gamma = 3 * k + i
                U[alpha][column] = 1
                V[beta][column] = 1
                W[gamma][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
