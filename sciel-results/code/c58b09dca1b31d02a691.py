"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.allclose(S, 1.0, atol=1e-6)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook
    U, V, W, rank = schoolbook()
    
    # Strategy: Try to find a decomposition with rank < 90.
    # The problem is equivalent to finding a rank-r decomposition of the MM tensor.
    # We can use a randomized approach with gradient descent in the continuous domain,
    # then snap to half-integers.
    # However, the landscape is non-convex and full of local minima.
    
    # A more robust approach for small ranks might be to use the structure of the tensor.
    # For 3x5x6, the schoolbook rank is 90.
    # Known best ranks for small matrices:
    # 2x2x2: 7 (Strassen)
    # 3x3x3: 23 (Winograd?)
    # 4x4x4: 49 (Winograd?)
    # 3x5x6: ?
    
    # Let's try a few random restarts with gradient descent for ranks 89 down to 80.
    # If we find a solution in float space, we try to snap it to half-integers.
    
    def try_rank(target_rank, max_iters=1000):
        if time.time() > deadline:
            return None
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                return None
                
            # Initialize with small random values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Use Adam optimizer manually? Or just SGD with momentum?
            # Let's use a simple gradient descent with adaptive learning rate.
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(max_iters):
                if time.time() > deadline:
                    return None
                    
                U_t = U.reshape(n, m, target_rank)
                V_t = V.reshape(m, p, target_rank)
                W_t = W.reshape(p, n, target_rank)
                
                # Compute S and Error
                S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                E = S - 1.0
                loss = np.sum(E*E)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if loss < 1e-10:
                    break
                    
                # Compute Gradients
                dU_t = np.einsum('ijk,jkc,kic->ijc', E, V_t, W_t)
                dV_t = np.einsum('ijk,ijc,kic->jkc', E, U_t, W_t)
                dW_t = np.einsum('ijk,ijc,jkc->kic', E, U_t, V_t)
                
                dU = dU_t.reshape(n*m, target_rank)
                dV = dV_t.reshape(m*p, target_rank)
                dW = dW_t.reshape(p*n, target_rank)
                
                # Update with momentum
                vU = momentum * vU + (1 - momentum) * dU
                vV = momentum * vV + (1 - momentum) * dV
                vW = momentum * vW + (1 - momentum) * dW
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
                
            # Check if the best found is valid
            if verify(best_U, best_V, best_W, target_rank):
                # Try to snap to half-integers
                U_snapped = np.round(best_U * 2) / 2
                V_snapped = np.round(best_V * 2) / 2
                W_snapped = np.round(best_W * 2) / 2
                
                if verify(U_snapped, V_snapped, W_snapped, target_rank):
                    return U_snapped, V_snapped, W_snapped, target_rank
                else:
                    # If snapping fails, maybe we can keep it if the evaluator accepts floats?
                    # The task says "integer/half-integer entries".
                    # So we must return snapped values if they verify.
                    # If they don't verify, this attempt failed.
                    pass
                    
        return None

    # Try to improve the rank
    # Start from 89 down to 85
    for target_rank in range(rank - 1, 84, -1):
        if time.time() > deadline:
            break
            
        result = try_rank(target_rank)
        if result is not None:
            U, V, W, rank = result
            break
            
    # If no improvement found, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
