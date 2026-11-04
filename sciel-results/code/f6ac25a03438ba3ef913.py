"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if S is all ones
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
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

def _gradient_descent(U, V, W, n, m, p, max_iter=5000, lr=0.01, tol=1e-10, deadline=None):
    """Gradient descent to minimize ||S - 1||^2 where S = U ⊙ V ⊙ W."""
    U_arr = np.array(U, dtype=float).reshape(n, m, -1)
    V_arr = np.array(V, dtype=float).reshape(m, p, -1)
    W_arr = np.array(W, dtype=float).reshape(p, n, -1)
    rank = U_arr.shape[2]
    
    best_loss = np.inf
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute S = sum_c U[:,:,c] * V[:,:,c] * W[:,:,c] (transposed indices)
        # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        
        # Loss = sum (S - 1)^2
        diff = S - 1.0
        loss = np.sum(diff ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_arr.copy()
            best_V = V_arr.copy()
            best_W = W_arr.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        # dL/dU[i,j,c] = 2 * (S[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_arr, W_arr)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_arr, W_arr)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_arr, V_arr)
        
        # Update
        U_arr -= lr * dU
        V_arr -= lr * dV
        W_arr -= lr * dW
    
    return best_U.reshape(n*m, rank).tolist(), best_V.reshape(m*p, rank).tolist(), best_W.reshape(p*n, rank).tolist()

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to round to half-integers and verify."""
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol):
        return U_r, V_r, W_r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to best_rank - 1
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations with gradient descent
        max_attempts = 200
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            seed = target_rank * 10000 + attempt
            U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=seed)
            
            # Run gradient descent
            U_opt, V_opt, W_opt = _gradient_descent(
                U_init, V_init, W_init, n, m, p,
                max_iter=2000, lr=0.01, tol=1e-10,
                deadline=deadline - 5
            )
            
            # Check if we found a valid solution
            if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                # Try to round to half-integers
                result = _try_round_and_verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-9)
                if result is not None:
                    U_r, V_r, W_r = result
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # If rounding fails, try with slightly different rounding
                    # or accept if very close
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
                        # Try rounding with different strategies
                        for round_mode in ['nearest_half', 'floor_half', 'ceil_half']:
                            if round_mode == 'nearest_half':
                                U_r = [[round(x * 2) / 2 for x in row] for row in U_opt]
                                V_r = [[round(x * 2) / 2 for x in row] for row in V_opt]
                                W_r = [[round(x * 2) / 2 for x in row] for row in W_opt]
                            elif round_mode == 'floor_half':
                                U_r = [[int(x * 2) / 2 for x in row] for row in U_opt]
                                V_r = [[int(x * 2) / 2 for x in row] for row in V_opt]
                                W_r = [[int(x * 2) / 2 for x in row] for row in W_opt]
                            else:  # ceil_half
                                U_r = [[int(x * 2 + 1) / 2 for x in row] for row in U_opt]
                                V_r = [[int(x * 2 + 1) / 2 for x in row] for row in V_opt]
                                W_r = [[int(x * 2 + 1) / 2 for x in row] for row in W_opt]
                            
                            if _verify(U_r, V_r, W_r, n, m, p, tol=1e-9):
                                best = {"U": U_r, "V": V_r, "W": W_r}
                                best_rank = target_rank
                                found = True
                                break
                        
                        if found:
                            break
        
        if best_rank == target_rank:
            break
    
    # Return the best found
    return best
