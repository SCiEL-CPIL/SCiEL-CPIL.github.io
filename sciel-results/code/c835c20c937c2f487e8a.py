import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 100  # Safety margin
    
    best_result = None
    best_error = float('inf')
    
    # Try ranks from lower bound upwards
    min_rank = max(n*m, m*p, p*n)
    
    for rank in range(min_rank, min_rank + 10): # Try up to min_rank + 10
        if time.time() - start_time > max_time:
            break
            
        # Initialize
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Optimization Loop
        lr = 0.01
        steps = 500
        
        for step in range(steps):
            if time.time() - start_time > max_time:
                break
            
            # Compute S and Error
            # S[i, j, j_prime, k] = sum_r U[i*m+j, r] * V[j_prime*p+k, r] * W[k*n+i, r]
            # This is a 4D tensor.
            
            # Efficient computation:
            # Reshape U to (n, m, rank)
            U_reshaped = U.reshape(n, m, rank)
            # Reshape V to (m, p, rank)
            V_reshaped = V.reshape(m, p, rank)
            # Reshape W to (p, n, rank) -> Note: W index is k*n+i. So W[k, i, r]
            W_reshaped = W.reshape(p, n, rank)
            
            # We need S[i, j, j_prime, k]
            # S[i, j, j_prime, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j_prime, k, r] * W_reshaped[k, i, r]
            
            # This can be computed via einsum
            # 'ijr, jkr, kir -> ijik'? No, indices are i, j, j_prime, k.
            # U: i, j, r
            # V: j_prime, k, r  (Note: V row is j_prime*p+k. So V_reshaped[j_prime, k, r])
            # W: k, i, r        (Note: W row is k*n+i. So W_reshaped[k, i, r])
            
            # S[i, j, jp, k] = sum_r U[i, j, r] V[jp, k, r] W[k, i, r]
            # einsum: 'ijr,jkr,kir->ijpk' ? No, the 'i' in W is the same 'i' as in U.
            # The 'k' in V is the same 'k' as in W.
            # So: U[i,j,r] * V[jp,k,r] * W[k,i,r]
            # Sum over r.
            # Result indices: i, j, jp, k.
            
            S = np.einsum('ijr,jkr,kir->ijpk', U_reshaped, V_reshaped, W_reshaped)
            
            # Target T[i, j, jp, k] = delta(j, jp)
            # T is 1 if j == jp, else 0.
            # We can create the target tensor.
            # Target shape (n, m, m, p).
            # T[i, j, jp, k] = 1 if j == jp else 0.
            # This is independent of i and k.
            # T = np.zeros((n, m, m, p))
            # for j in range(m): T[:, j, j, :] = 1
            
            # Error
            # E = sum(S - T)^2
            
            # Gradients
            # dE/dU[i, j, r] = 2 * sum_{jp, k} (S[i, j, jp, k] - T[i, j, jp, k]) * V[jp, k, r] * W[k, i, r]
            
            # Let's compute gradients using einsum as well.
            # diff = S - T
            # dU = 2 * einsum('ijpk, jkr, kir -> ijr', diff, V_reshaped, W_reshaped)
            # dV = 2 * einsum('ijpk, ijr, kir -> jkr', diff, U_reshaped, W_reshaped)
            # dW = 2 * einsum('ijpk, ijr, jkr -> kir', diff, U_reshaped, V_reshaped)
            
            # Create T
            T = np.zeros((n, m, m, p))
            for j in range(m):
                T[:, j, j, :] = 1.0
                
            diff = S - T
            
            dU = 2.0 * np.einsum('ijpk,jkr,kir->ijr', diff, V_reshaped, W_reshaped)
            dV = 2.0 * np.einsum('ijpk,ijr,kir->jkr', diff, U_reshaped, W_reshaped)
            dW = 2.0 * np.einsum('ijpk,ijr,jkr->kir', diff, U_reshaped, V_reshaped)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        # Check convergence
        error = np.sum(diff**2)
        
        if error < best_error:
            best_error = error
            best_result = (U.copy(), V.copy(), W.copy(), rank)
            
        if error < 1e-9:
            # Found exact solution (likely)
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            U_r = U_rounded.reshape(n, m, rank)
            V_r = V_rounded.reshape(m, p, rank)
            W_r = W_rounded.reshape(p, n, rank)
            S_check = np.einsum('ijr,jkr,kir->ijpk', U_r, V_r, W_r)
            if np.allclose(S_check, T, atol=1e-9):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                
    # Return best found
    if best_result:
        U, V, W, r = best_result
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        return {
            "U": U_rounded.tolist(),
            "V": V_rounded.tolist(),
            "W": W_rounded.tolist()
        }
        
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
