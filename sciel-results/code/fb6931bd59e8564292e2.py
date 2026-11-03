import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Build the target tensor: T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))
    
    # Schoolbook: rank = n*m*p = 36
    best_rank = n * m * p
    U = np.zeros((num_A, best_rank))
    V = np.zeros((num_B, best_rank))
    W = np.zeros((num_C, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    
    # Try to reduce rank using structured approaches
    # Strategy: Kronecker product structure exploiting 6 = 2*3
    # The tensor for 2x3x6 can potentially be decomposed using the factorization
    # of dimensions. Let's try several target ranks with gradient descent.
    
    if time.time() < deadline:
        # Precompute all invalid triples for faster checking
        all_triples = set()
        for a in range(num_A):
            for b in range(num_B):
                for c in range(num_C):
                    all_triples.add((a, b, c))
        invalid_triples = all_triples - valid_triples
        invalid_list = list(invalid_triples)
        
        # Try several target ranks
        for target_rank in [30, 27, 24, 21, 18, 16, 15, 14, 13, 12]:
            if time.time() > deadline:
                break
            
            found = False
            # Multiple restarts with different random seeds
            for restart in range(3):
                if time.time() > deadline:
                    break
                
                np.random.seed(12345 + target_rank * 100 + restart)
                U_curr = np.random.randn(num_A, target_rank) * 0.1
                V_curr = np.random.randn(num_B, target_rank) * 0.1
                W_curr = np.random.randn(num_C, target_rank) * 0.1
                
                # Gradient descent with adaptive learning rate
                lr = 0.05
                best_res = float('inf')
                best_U_state = U_curr.copy()
                best_V_state = V_curr.copy()
                best_W_state = W_curr.copy()
                
                for step in range(300):
                    if time.time() > deadline:
                        break
                    
                    # Compute gradients efficiently
                    grad_U = np.zeros_like(U_curr)
                    grad_V = np.zeros_like(V_curr)
                    grad_W = np.zeros_like(W_curr)
                    residual_sum = 0
                    
                    # Process valid triples
                    for (a, b, c) in valid_triples:
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        err = val - 1.0
                        residual_sum += err * err
                        
                        grad_U[a, :] += err * V_curr[b, :] * W_curr[c, :]
                        grad_V[b, :] += err * U_curr[a, :] * W_curr[c, :]
                        grad_W[c, :] += err * U_curr[a, :] * V_curr[b, :]
                    
                    # Process a sample of invalid triples
                    if step % 10 == 0:  # Check all invalid every 10 steps
                        invalid_sample = invalid_list
                    else:
                        np.random.seed(step * 7 + restart)
                        sample_size = min(len(invalid_list), 100)
                        indices = np.random.choice(len(invalid_list), sample_size, replace=False)
                        invalid_sample = [invalid_list[i] for i in indices]
                    
                    for (a, b, c) in invalid_sample:
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        err = val
                        residual_sum += err * err * 0.5  # Weight invalid positions
                        grad_U[a, :] += err * 0.5 * V_curr[b, :] * W_curr[c, :]
                        grad_V[b, :] += err * 0.5 * U_curr[a, :] * W_curr[c, :]
                        grad_W[c, :] += err * 0.5 * U_curr[a, :] * V_curr[b, :]
                    
                    # Update with gradient descent
                    U_curr -= lr * grad_U
                    V_curr -= lr * grad_V
                    W_curr -= lr * grad_W
                    
                    # Decay learning rate
                    lr *= 0.995
                    
                    # Track best state
                    if residual_sum < best_res:
                        best_res = residual_sum
                        best_U_state = U_curr.copy()
                        best_V_state = V_curr.copy()
                        best_W_state = W_curr.copy()
                    
                    if residual_sum < 1e-10:
                        found = True
                        break
                
                # Verify the best state found in this restart
                U_check = best_U_state
                V_check = best_V_state
                W_check = best_W_state
                
                max_err = 0
                for (a, b, c) in valid_triples:
                    val = np.sum(U_check[a, :] * V_check[b, :] * W_check[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Check invalid positions
                    for (a, b, c) in invalid_list[:200]:  # Sample check
                        val = np.sum(U_check[a, :] * V_check[b, :] * W_check[c, :])
                        max_err = max(max_err, abs(val))
                
                if max_err < 1e-6:
                    best_rank = target_rank
                    best_U = U_check.tolist()
                    best_V = V_check.tolist()
                    best_W = W_check.tolist()
                    found = True
                    break
            
            if found:
                break
    
    # Final verification
    U_arr = np.array(best_U, dtype=float)
    V_arr = np.array(best_V, dtype=float)
    W_arr = np.array(best_W, dtype=float)
    
    # Verify valid triples
    for (a, b, c) in valid_triples:
        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        if abs(val - 1.0) > 1e-6:
            # Fallback to schoolbook if verification fails
            U = np.zeros((num_A, best_rank))
            V = np.zeros((num_B, best_rank))
            W = np.zeros((num_C, best_rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1
                        V[b, col] = 1
                        W[c, col] = 1
                        col += 1
            best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
