"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120s safety margin
    
    # The constraint: for all i, j, j', k:
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    
    # Number of constraints: n * m * m * p
    num_constraints = n * m * m * p
    
    def check_solution(U, V, W, r):
        """Check if U, V, W with rank r satisfy the tensor identity exactly."""
        # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > 1e-9:
                            return False
        return True
    
    def try_rank(r, max_iters=5000):
        """Try to find a valid decomposition of rank r."""
        if time.time() > deadline - 10:
            return None
        
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
        
        # Try to optimize
        best_violation = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute violation
            violation = 0.0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c]
                            expected = 1.0 if j == jp else 0.0
                            violation += (s - expected) ** 2
            
            if violation < best_violation:
                best_violation = violation
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if violation < 1e-12:
                    # Verify with exact check
                    if check_solution(best_U, best_V, best_W, r):
                        return best_U, best_V, best_W
            
            # Random perturbation
            if np.random.random() < 0.5:
                idx = np.random.randint(n*m)
                col = np.random.randint(r)
                best_U[idx, col] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                U = best_U.copy()
            elif np.random.random() < 0.5:
                idx = np.random.randint(m*p)
                col = np.random.randint(r)
                V[idx, col] = np.random.choice([-1, -0.5, 0, 0.5, 1])
            else:
                idx = np.random.randint(p*n)
                col = np.random.randint(r)
                W[idx, col] = np.random.choice([-1, -0.5, 0, 0.5, 1])
        
        return None
    
    # Try ranks from 56 downward
    for r in range(n*m*p, 20, -1):
        if time.time() > deadline - 15:
            break
        result = try_rank(r, max_iters=3000)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
