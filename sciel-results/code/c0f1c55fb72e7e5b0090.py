# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n, m, p = 3, 4, 4
    schoolbook_rank = n * m * p
    
    # 1. Construct Schoolbook Solution (Rank 48)
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, idx] = 1.0
                V_sb[j * p + k, idx] = 1.0
                W_sb[k * n + i, idx] = 1.0
                idx += 1
                
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def verify(U, V, W, tol=1e-9):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape for einsum
        # U -> (n, m, R)
        # V -> (m, p, R)
        # W -> (p, n, R)
        # S_ijk = sum_c U_ijk V_jkc W_kic
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # Compute S
        # einsum: 'ijc,jkc,kic->ijk'
        S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Check if S == 1
        if not np.allclose(S, 1.0, atol=tol):
            return False
        return True

    def try_rank(R, num_trials=10, max_steps=500):
        if R <= 0:
            return None
        if time_left() < 10:
            return None
            
        for trial in range(num_trials):
            if time_left() < 5:
                break
                
            # Random initialization
            # Scale to keep values reasonable
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            
            lr = 0.01
            prev_loss = float('inf')
            
            for step in range(max_steps):
                if time_left() < 5:
                    break
                    
                # Compute S
                S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                # Error
                E = 1.0 - S
                
                # Loss
                loss = np.sum(E**2)
                
                # Check convergence
                if loss < 1e-10:
                    # Try to round
                    U_flat = U_r.reshape(n*m, R)
                    V_flat = V_r.reshape(m*p, R)
                    W_flat = W_r.reshape(p*n, R)
                    
                    U_round = np.round(U_flat * 2) / 2
                    V_round = np.round(V_flat * 2) / 2
                    W_round = np.round(W_flat * 2) / 2
                    
                    if verify(U_round, V_round, W_round):
                        return U_round, V_round, W_round
                    # If rounding fails, maybe try exact values if they are close to integers
                    # But the problem requires integer/half-integer.
                    # If loss is very low, the values are likely close to the solution.
                    # We can try a local search around the rounded values?
                    # For now, return None if rounding fails, as we need exact half-integers.
                    break

                # Compute Gradients
                # dL/dU_ijk = -2 * sum_k E_ijk V_jkc W_kic
                # dL/dV_jkc = -2 * sum_i E_ijk U_ijk W_kic
                # dL/dW_kic = -2 * sum_j E_ijk U_ijk V_jkc
                
                dU = -2.0 * np.einsum('ijk,jkc,kic->ijc', E, V_r, W_r)
                dV = -2.0 * np.einsum('ijk,ijc,kic->jkc', E, U_r, W_r)
                dW = -2.0 * np.einsum('ijk,ijc,jkc->kic', E, U_r, V_r)
                
                # Update
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
                
                # Optional: Learning rate decay
                if step % 50 == 0 and step > 0:
                    lr *= 0.95

                if loss > prev_loss:
                    # Revert step? Simple backtracking
                    U_r += lr * dU
                    V_r += lr * dV
                    W_r += lr * dW
                    lr *= 0.5
                    continue
                prev_loss = loss

            # After GD, check if we got close enough to round
            S = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            loss = np.sum((1.0 - S)**2)
            if loss < 1e-6:
                U_flat = U_r.reshape(n*m, R)
                V_flat = V_r.reshape(m*p, R)
                W_flat = W_r.reshape(p*n, R)
                
                # Try rounding to nearest half-integer
                U_round = np.round(U_flat * 2) / 2
                V_round = np.round(V_flat * 2) / 2
                W_round = np.round(W_flat * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    return U_round, V_round, W_round

        return None

    # Try rank 47
    if time_left() > 15:
        result_47 = try_rank(schoolbook_rank - 1, num_trials=5, max_steps=200)
        if result_47:
            best_U, best_V, best_W = result_47
            best_rank = schoolbook_rank - 1

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
