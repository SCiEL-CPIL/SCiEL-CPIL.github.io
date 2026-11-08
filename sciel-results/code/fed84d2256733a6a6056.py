import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W (lists) form a valid decomposition."""
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Compute sum over r of U[:,r] outer V[:,r] outer W[:,r]
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_approx = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_approx, T, atol=1e-9)

    def to_list(arr):
        return [[float(x) for x in row] for row in arr]

    # Try to find rank 27 decomposition using optimization
    def try_optimize(target_rank, max_iters=5000, lr=0.01, n_restarts=3):
        if time.time() > deadline:
            return None
        best_loss = float('inf')
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with a valid rank-30 solution, then project
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with schoolbook-like structure
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i * m * p + j * p + k < target_rank:
                            U[i * m + j, i * m * p + j * p + k] = 1.0
                            V[j * p + k, i * m * p + j * p + k] = 1.0
                            W[k * n + i, i * m * p + j * p + k] = 1.0
            
            # Add random perturbation
            U += np.random.randn(n * m, target_rank) * 0.01
            V += np.random.randn(m * p, target_rank) * 0.01
            W += np.random.randn(p * n, target_rank) * 0.01
            
            current_lr = lr
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute T_approx = sum_r U[:,r] outer V[:,r] outer W[:,r]
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                loss = np.sum((T_approx - T) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                # dL/dU[a,r] = 2 * sum_{b,c} (T_approx[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
                # dL/dV[b,r] = 2 * sum_{a,c} (T_approx[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
                # dL/dW[c,r] = 2 * sum_{a,b} (T_approx[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
                
                diff = T_approx - T  # shape (n*m, m*p, p*n)
                
                # Gradient for U: shape (n*m, target_rank)
                # grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                
                # Gradient for V: shape (m*p, target_rank)
                # grad_V[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                
                # Gradient for W: shape (p*n, target_rank)
                # grad_W[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Update
                U -= current_lr * grad_U
                V -= current_lr * grad_V
                W -= current_lr * grad_W
                
                # Adaptive learning rate
                if it % 100 == 0 and it > 0:
                    current_lr *= 0.95
            
            # Check if we found a valid solution
            if best_solution is not None:
                U_sol, V_sol, W_sol = best_solution
                # Round to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                U_list = to_list(U_rounded)
                V_list = to_list(V_rounded)
                W_list = to_list(W_rounded)
                
                if verify(U_list, V_list, W_list):
                    return U_list, V_list, W_list
        
        return None

    # First, try the known rank-28 decomposition
    def construct_rank_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
        # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
                
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
        # M1: (a+b)(e+f)
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M2: (c+d)e
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 0][r] = 1
        W[0 * n + 1][r] = 1
        W[1 * n + 1][r] = -1
        r += 1
        
        # M3: a(f-j)
        U[0 * m + 0][r] = 1
        V[0 * p + 1][r] = 1
        V[1 * p + 1][r] = -1
        W[1 * n + 0][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M4: d(i-e)
        U[1 * m + 1][r] = 1
        V[1 * p + 0][r] = 1
        V[0 * p + 0][r] = -1
        W[0 * n + 0][r] = 1
        W[0 * n + 1][r] = 1
        r += 1
        
        # M5: (a+b)k
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = -1
        W[1 * n + 0][r] = 1
        r += 1
        
        # M6: (c+d)(e+f)
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 0][r] = 1
        V[0 * p + 1][r] = 1
        W[1 * n + 1][r] = 1
        r += 1
        
        # M7: (b-d)(f+j)
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 0][r] = 1
        V[1 * p + 1][r] = 1
        W[0 * n + 0][r] = 1
        r += 1
        
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
        # M1: (a+b)(g+h)
        U[0 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M2: (c+d)g
        U[1 * m + 0][r] = 1
        U[1 * m + 1][r] = 1
        V[0 * p + 2][r] = 1
        W[2 * n + 1][r] = 1
        W[3 * n + 1][r] = -1
        r += 1
        
        # M3: a(h-k)
        U[0 * m + 0][r] = 1
        V[0 * p + 3][r] = 1
        V[1 * p + 3][r] = -1
        W[3 * n + 0][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M4: d(k-g)
        U[1 * m + 1][r] = 1
        V[1 * p + 2][r] = 1
        V[0 * p + 2][r] = -1
        W[2 * n + 0][r] = 1
        W[2 * n + 1][r] = 1
        r += 1
        
        # M5: (a+b)k
        U[0 * m + 0][r] = 1
        U[0 * m + 1][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = -1
        W[3 * n + 0][r] = 1
        r += 1
        
        # M6: (c+d)(g+h)
        U[1 * m + 0][r] = 1
        U[0 * m + 0][r] = -1
        V[0 * p + 2][r] = 1
        V[0 * p + 3][r] = 1
        W[3 * n + 1][r] = 1
        r += 1
        
        # M7: (b-d)(h+k)
        U[0 * m + 1][r] = 1
        U[1 * m + 1][r] = -1
        V[1 * p + 2][r] = 1
        V[1 * p + 3][r] = 1
        W[2 * n + 0][r] = 1
        r += 1
        
        return U, V, W

    # Try to find rank 27 decomposition
    if time.time() < deadline - 5:
        result = try_optimize(27, max_iters=3000, lr=0.005, n_restarts=2)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fall back to rank 28
    U, V, W = construct_rank_28()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
