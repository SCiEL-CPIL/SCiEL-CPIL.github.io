import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # The trivial decomposition has rank n*m*p = 120
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1
    
    # Try to find a lower rank decomposition using optimization
    # We'll use a random initialization and gradient descent on the rank
    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()
    
    # Try various ranks from 100 down to 80 (reasonable range to search)
    for target_rank in range(100, 79, -1):
        if time.time() > deadline - 20:
            break
            
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # Use Adam-like optimization
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            t = 0
            max_iter = 2000
            loss_history = []
            
            for iteration in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                t += 1
                
                # Compute loss: sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            
                            # Compute the product sum
                            prod = np.sum(U[a, :] * V[b, :] * W[c, :])
                            error = prod - 1.0
                            loss += error * error
                            
                            # Gradients
                            # d/dU[a,r] = 2*error * V[b,r] * W[c,r]
                            grad_U[a, :] += 2 * error * V[b, :] * W[c, :]
                            grad_V[b, :] += 2 * error * U[a, :] * W[c, :]
                            grad_W[c, :] += 2 * error * U[a, :] * V[b, :]
                
                loss /= (n * m * p)
                loss_history.append(loss)
                
                # Check convergence
                if loss < 1e-6:
                    break
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Check if we achieved a valid decomposition
            if loss_history and loss_history[-1] < 1e-6:
                # Verify more carefully
                max_error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            prod = np.sum(U[a, :] * V[b, :] * W[c, :])
                            max_error = max(max_error, abs(prod - 1.0))
                
                if max_error < 1e-5:
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break  # Found a valid decomposition for this rank, try lower
    
    # Return the best result found
    if best_rank < trivial_rank:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
