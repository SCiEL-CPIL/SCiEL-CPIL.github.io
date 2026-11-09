import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Target: Find U, V, W such that
    # sum_a U[i,j,a] * V[j',k,a] * W[k,i,a] == delta(j, j')
    # U shape: (n*m, r)
    # V shape: (m*p, r)
    # W shape: (p*n, r)
    
    # We will try to find a decomposition for various ranks.
    # Start with a reasonable guess. Schoolbook is 60.
    # Known bounds for 3x4x5 might be around 30-40.
    
    # Define the loss function
    def compute_loss(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reshape for easy indexing
        U_reshaped = U.T.reshape((r, n, m)) # r, i, j
        V_reshaped = V.T.reshape((r, m, p)) # r, j, k
        W_reshaped = W.T.reshape((r, p, n)) # r, k, i
        
        # We need sum_a U[i,j,a] V[j',k,a] W[k,i,a]
        # Let's compute the tensor T[i, j, j', k] = sum_a ...
        # This is a 4D tensor.
        
        # Efficient computation:
        # For each i, k:
        # M[i,k,j,j'] = sum_a U[i,j,a] V[j',k,a] W[k,i,a]
        
        # We can use einsum
        # U[i,j,a] V[j',k,a] -> sum over a
        # Then multiply by W[k,i,a] and sum over a? No, W is also summed over a.
        # So it's sum_a (U[i,j,a] V[j',k,a] W[k,i,a])
        
        # Let's compute S[i, j, j', k]
        # S = np.einsum('ija,jka,ki->ijjk', U_reshaped, V_reshaped, W_reshaped)
        # This might be slow for large r, but r is small (< 60).
        
        # Target is delta(j, j') for all i, k.
        # So for each i, k, the matrix S[i, :, :, k] should be Identity(m).
        
        # Loss = sum_{i,k} || S[i, :, :, k] - I_m ||_F^2
        
        S = np.einsum('ija,jka,ki->ijjk', U_reshaped, V_reshaped, W_reshaped)
        
        # Create target identity tensor
        # Target[i, j, j', k] = 1 if j==j' else 0
        # This is independent of i, k.
        I_m = np.eye(m)
        # Expand to (n, m, m, p)
        # Target[i, j, j', k] = I_m[j, j']
        # We can just subtract I_m from each slice.
        
        # S shape: (n, m, m, p)
        # We want S[i, :, :, k] == I_m
        # So loss = sum_{i,k} sum_{j,j'} (S[i,j,j',k] - I_m[j,j'])^2
        
        # Reshape S to (n*p, m*m)
        S_flat = S.transpose(0, 3, 1, 2).reshape(n*p, m*m)
        I_flat = np.tile(I_m, (n*p, 1)).reshape(n*p, m*m)
        
        diff = S_flat - I_flat
        return np.sum(diff**2)

    def random_init(r, scale=1.0):
        U = np.random.randn(n*m, r) * scale
        V = np.random.randn(m*p, r) * scale
        W = np.random.randn(p*n, r) * scale
        return U, V, W

    def refine(U, V, W, r, max_iters=200, lr=0.01):
        # Simple gradient descent
        # Loss L = sum (S - I)^2
        # Gradient w.r.t U, V, W
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            U_reshaped = U.T.reshape((r, n, m))
            V_reshaped = V.T.reshape((r, m, p))
            W_reshaped = W.T.reshape((r, p, n))
            
            # S[i, j, j', k] = sum_a U[i,j,a] V[j',k,a] W[k,i,a]
            S = np.einsum('ija,jka,ki->ijjk', U_reshaped, V_reshaped, W_reshaped)
            
            I_m = np.eye(m)
            # diff[i, j, j', k] = S[i, j, j', k] - I_m[j, j']
            # We need gradients.
            # dL/dU[i,j,a] = 2 * sum_{k, j'} (S[i,j,j',k] - I_m[j,j']) * V[j',k,a] * W[k,i,a]
            # dL/dV[j',k,a] = 2 * sum_{i, j} (S[i,j,j',k] - I_m[j,j']) * U[i,j,a] * W[k,i,a]
            # dL/dW[k,i,a] = 2 * sum_{i, j, j'} (S[i,j,j',k] - I_m[j,j']) * U[i,j,a] * V[j',k,a]
            
            # Compute residuals R[i, j, j', k] = S[i, j, j', k] - I_m[j, j']
            # Note: I_m[j,j'] is constant for each i,k slice.
            # Let's create a tensor R of shape (n, m, m, p)
            # R[i, j, j', k] = S[i, j, j', k] - I_m[j, j']
            
            # To compute gradients efficiently:
            # grad_U[i, j, a] = 2 * sum_{k, j'} R[i, j, j', k] * V[j', k, a] * W[k, i, a]
            # This is einsum('ijjk,jka,ki->ija', R, V_reshaped, W_reshaped)
            
            # We need R.
            # S is (n, m, m, p)
            # I_m is (m, m)
            # We can broadcast I_m to (n, m, m, p)
            # I_m_expanded = np.broadcast_to(I_m, (n, m, m, p))
            # R = S - I_m_expanded
            
            # However, creating I_m_expanded is small.
            
            # Let's compute R
            # S shape (n, m, m, p)
            # I_m shape (m, m)
            # R[i, j, j', k] = S[i, j, j', k] - I_m[j, j']
            
            # We can do:
            # R = S - np.tile(I_m, (n, 1, 1, p)).transpose(0, 1, 3, 2) ? No.
            # I_m[j, j'] needs to be aligned with S[i, j, j', k].
            # So I_m_expanded[i, j, j', k] = I_m[j, j']
            
            # Let's just use loops or careful einsum.
            # Actually, we can compute gradients without explicitly forming R if we use linearity.
            # But R is needed.
            
            # Let's form R
            # I_m is (m, m)
            # We want to subtract I_m[j, j'] from S[i, j, j', k]
            # This is equivalent to subtracting a tensor T where T[i, j, j', k] = I_m[j, j']
            
            # We can create T by broadcasting
            # I_m_4d = I_m[None, :, :, None] # (1, m, m, 1)
            # T = np.broadcast_to(I_m_4d, (n, m, m, p))
            # R = S - T
            
            # This creates a large array if n, m, p are large, but here they are small.
            
            # Let's optimize:
            # grad_U = 2 * einsum('ijjk,jka,ki->ija', R, V_reshaped, W_reshaped)
            # grad_V = 2 * einsum('ijjk,ija,ki->jka', R, U_reshaped, W_reshaped)
            # grad_W = 2 * einsum('ijjk,ija,jka->ki', R, U_reshaped, V_reshaped)
            
            # Update
            U -= lr * grad_U.T.reshape(n*m, r)
            V -= lr * grad_V.T.reshape(m*p, r)
            W -= lr * grad_W.T.reshape(p*n, r)
            
        # Return final loss
        return compute_loss(U, V, W, r)

    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    best_loss = float('inf')

    # Try ranks from low to high
    # Start with 20, 25, 30, 35, 40
    for r in [20, 25, 30, 35, 40, 45, 50]:
        if time.time() > deadline - 20:
            break
            
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            U, V, W = random_init(r)
            
            # Refine
            loss = refine(U, V, W, r, max_iters=100, lr=0.05)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = r
                
            # Check if loss is very small, then try to round
            if loss < 1e-4:
                # Try rounding to integers or half-integers
                # Scale? The problem says integer/half-integer.
                # We can try to round U, V, W to nearest 0.5
                
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                loss_r = compute_loss(U_r, V_r, W_r, r)
                if loss_r < 1e-6:
                    # Found an exact solution!
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }

    # If no exact solution found, return the best approximate one?
    # The task requires EXACT decomposition.
    # If we fail, we should return the schoolbook to be safe?
    # But the score is the rank. If we return schoolbook, score is 1.0.
    # If we return a non-exact one, it's invalid (score 0).
    # So we must return an exact one.
    
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

