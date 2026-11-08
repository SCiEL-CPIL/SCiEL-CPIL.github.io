"""Matrix multiplication 2,4,6 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the target tensor T of shape (n*m, m*p, p*n) = (8, 24, 12)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i with i=i', j=j', k=k'
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T.flatten()
    
    def tensor_product(U, V, W, R):
        """Compute sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]."""
        result = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for r in range(R):
            result += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r], optimize=True)
        return result
    
    def objective_and_grads(U, V, W, R):
        """Compute objective and gradients."""
        prod = tensor_product(U, V, W, R)
        diff = prod - T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            grad_U[:, r] = np.einsum('ijk,j,k->i', diff, V[:, r], W[:, r], optimize=True)
            grad_V[:, r] = np.einsum('ijk,i,k->j', diff, U[:, r], W[:, r], optimize=True)
            grad_W[:, r] = np.einsum('ijk,i,j->k', diff, U[:, r], V[:, r], optimize=True)
        
        obj = np.dot(diff.flatten(), diff.flatten())
        return obj, grad_U, grad_V, grad_W
    
    def verify_decomp(U, V, W):
        """Verify that the decomposition exactly reconstructs T."""
        prod = tensor_product(U, V, W, U.shape[1])
        return np.max(np.abs(prod - T))
    
    def snap_to_half_int(M):
        """Snap values to nearest 0.5."""
        return np.round(M * 2.0) / 2.0
    
    # Build the rank-42 baseline using block Strassen
    rank_42 = 42
    U_base = np.zeros((n * m, rank_42), dtype=np.float64)
    V_base = np.zeros((m * p, rank_42), dtype=np.float64)
    W_base = np.zeros((p * n, rank_42), dtype=np.float64)
    
    # Strassen-like coefficients for 2x2 blocks
    # We decompose 2x4 * 4x6 into blocks of 2x2
    # There are 2 block rows (i: 0,1) and 3 block columns (k: 0,1,2)
    # Each block multiplication is 2x2 * 2x2
    
    # For each (a,b) block position where a in {0,1} (row block), b in {0,1,2} (col block)
    # We use 7 Strassen multiplications
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U_base[i * m + j, col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V_base[j * p + k, col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W_base[k * n + i, col] += coeff
                
                col += 1
    
    # Verify baseline
    base_err = verify_decomp(U_base, V_base, W_base)
    
    best_rank = rank_42
    best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    # Try to find lower rank via gradient-based optimization
    np.random.seed(42)
    
    for R in range(41, 30, -1):
        if time.time() > deadline - 25:
            break
        
        found = False
        max_attempts = 4 if R >= 38 else 3
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Initialize by sampling R columns from the Rank 42 baseline
            idx = np.random.choice(42, R, replace=False)
            U_try = U_base[:, idx].copy()
            V_try = V_base[:, idx].copy()
            W_try = W_base[:, idx].copy()
            
            # Add small noise to escape local minima
            U_try += np.random.randn(n * m, R) * 0.05
            V_try += np.random.randn(m * p, R) * 0.05
            W_try += np.random.randn(p * n, R) * 0.05
            
            lr = 0.03
            obj_prev = float('inf')
            stall_count = 0
            best_obj = float('inf')
            
            max_steps = 150
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    break
                
                obj, dU, dV, dW = objective_and_grads(U_try, V_try, W_try, R)
                
                if obj < 1e-10:
                    found = True
                    break
                
                if obj < best_obj:
                    best_obj = obj
                    stall_count = 0
                else:
                    stall_count += 1
                
                if stall_count > 20:
                    lr *= 0.5
                    if lr < 1e-7:
                        break
                
                # Adaptive learning rate
                U_try -= lr * dU
                V_try -= lr * dV
                W_try -= lr * dW
                
                obj_prev = obj
            
            # After optimization, try snapping to half-integers
            if obj < 1e-3:
                U_snap = snap_to_half_int(U_try)
                V_snap = snap_to_half_int(V_try)
                W_snap = snap_to_half_int(W_try)
                
                prod_snap = tensor_product(U_snap, V_snap, W_snap, R)
                max_err_snap = np.max(np.abs(prod_snap - T))
                
                if max_err_snap < 1e-9:
                    found = True
                    U_try, V_try, W_try = U_snap, V_snap, W_snap
                    break
            elif obj < 1e-10:
                found = True
                break
        
        if found:
            max_err = verify_decomp(U_try, V_try, W_try)
            if max_err < 1e-8:
                best_rank = R
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                break
    
    # Final verification
    if time.time() < deadline - 3:
        final_err = verify_decomp(best_U, best_V, best_W)
        if final_err > 1e-6:
            best_rank = rank_42
            best_U, best_V, best_W = U_base.copy(), V_base.copy(), W_base.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
