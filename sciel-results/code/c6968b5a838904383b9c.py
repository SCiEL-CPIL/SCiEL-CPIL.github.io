import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Build the tensor T[a,b,c]
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a decomposition of rank r < 18
    # Use a simple approach: try random initializations and gradient descent
    # Or use algebraic methods
    
    # First, let's try known results. For 2x3 x 3x3, the minimum rank is known.
    # Let me try ranks from 17 down to see if we can find a solution.
    
    best_result = fallback
    
    # Try to find lower rank using numerical optimization
    for target_rank in range(17, 10, -1):
        if time.time() > deadline - 30:
            break
        # Try several random initializations
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            np.random.seed(42 + trial)
            U_init = np.random.randn(nm, target_rank)
            V_init = np.random.randn(mp, target_rank)
            W_init = np.random.randn(pn, target_rank)
            
            # Simple gradient descent
            lr = 0.01
            for iteration in range(500):
                if time.time() > deadline - 30:
                    break
                # Compute current tensor
                T_current = np.einsum('ar,br,cr->abc', U_init, V_init, W_init)
                residual = T - T_current
                loss = np.sum(residual ** 2)
                
                if loss < 1e-8:
                    break
                
                # Compute gradients
                # dL/dU = 2 * residual contracted with V, W
                grad_U = np.einsum('abc,br,cr->ar', residual, V_init, W_init)
                grad_V = np.einsum('abc,ar,cr->br', residual, U_init, W_init)
                grad_W = np.einsum('abc,ar,br->cr', residual, U_init, V_init)
                
                U_init -= lr * grad_U
                V_init -= lr * grad_V
                W_init -= lr * grad_W
            
            if np.sum((T - np.einsum('ar,br,cr->abc', U_init, V_init, W_init)) ** 2) < 1e-6:
                # Found a valid decomposition
                # Convert to list format with integer/half-integer check
                U_list = [[float(x) for x in row] for row in U_init]
                V_list = [[float(x) for x in row] for row in V_init]
                W_list = [[float(x) for x in row] for row in W_init]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        if best_result is not fallback:
            break
    
    return best_result
