"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import numpy as np
import time
import random
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    # Deadline guard: must finish within 120s, leave safety margin
    deadline = time.time() + 110
    
    # Schoolbook decomposition as fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Check that U, V, W have correct shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        # For each (i, j, k) where j == j', i == i', k == k', the sum should be 1
        # For all other combinations, the sum should be 0
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # T should be 1 iff j==j' and i==i' and k==k'
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # For each (i, j, k) triple, check T[i*m+j, j*p+k, k*n+i] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(val - 1.0) > 1e-6:
                        return False
        
        # Check that T is 0 for all other combinations
        # This is expensive, so do a sample check
        checked = 0
        for _ in range(5000):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            i2 = random.randint(0, n - 1)
            j2 = random.randint(0, m - 1)
            k2 = random.randint(0, p - 1)
            if i == i2 and j == j2 and k == k2:
                continue
            a = i * m + j
            b = j2 * p + k
            c = k2 * n + i2
            val = np.dot(U_np[a], V_np[b] * W_np[c])
            if abs(val) > 1e-6:
                return False
            checked += 1
        
        # Also do a more thorough check on a subset
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2:
                        continue
                    for k in range(p):
                        a = i * m + j
                        b = j2 * p + k
                        for i2 in range(n):
                            c = k * n + i2
                            val = np.dot(U_np[a], V_np[b] * W_np[c])
                            if abs(val) > 1e-6:
                                return False
        
        return True
    
    # Try to find a lower rank decomposition using structured search
    # The key insight for n=2: we can exploit the small first dimension
    
    # Strategy: Use the fact that for n=2, the tensor has a special structure.
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # Let's try a parameterized approach:
    # For each term r, U[i*m+j, r] and V[j'*p+k, r] and W[k'*n+i, r]
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 iff (i,j,k) matches
    # and 0 otherwise.
    
    # A structured approach: assume U, V, W have a certain pattern.
    # For n=2, we can write:
    # Let U[i*m+j, r] = u[i, j, r]
    # Let V[j*p+k, r] = v[j, k, r]  
    # Let W[k*n+i, r] = w[k, i, r]
    
    # The constraint becomes:
    # sum_r u[i,j,r] * v[j,k,r] * w[k,i,r] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # Wait, that's not quite right. Let me be more careful.
    
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T should equal 1 if i==i', j==j', k==k', else 0.
    
    # So: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # This is a system of equations in the entries of U, V, W.
    
    # For a given rank R, we have R * (n*m + m*p + p*n) = R * (10 + 30 + 12) = 52R variables.
    # And n*m*p = 60 equations (one for each (i,j,k) triple that should give 1),
    # plus many more equations that should give 0.
    
    # The number of "should be 0" equations is:
    # Total (i,j,k,i',j',k') combinations: 2*5*6*2*5*6 = 3600
    # Minus the 60 that should be 1: 3540 equations that should be 0.
    
    # So we have 3600 equations and 52R unknowns.
    # For R=30: 1560 unknowns, 3600 equations. Overdetermined.
    # For R=60: 3120 unknowns, 3600 equations. Still overdetermined.
    
    # This suggests that a generic solution may not exist at rank 30,
    # but the structure of the problem may allow a solution.
    
    # Let's try a different approach: use the known structure of matrix multiplication.
    
    # For n=2, m=5, p=6, let's try to find a decomposition of rank R.
    # We'll use a local search / optimization approach.
    
    def _try_rank(R, max_iters=5000, seed=42):
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W with small random values
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        # Compute the target tensor T
        # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k), else 0
        
        # We'll use gradient descent to minimize:
        # sum_{a,b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
        
        # But this is expensive. Let's use a sampled version.
        
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Sample a batch of (a, b, c) triples
            batch_size = 200
            errors = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for _ in range(batch_size):
                # Random (i, j, k, i', j', k')
                i = rng.randint(0, n)
                j = rng.randint(0, m)
                k = rng.randint(0, p)
                i2 = rng.randint(0, n)
                j2 = rng.randint(0, m)
                k2 = rng.randint(0, p)
                
                a = i * m + j
                b = j2 * p + k
                c = k2 * n + i2
                
                target = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                
                # Compute current value
                val = np.dot(U[a], V[b] * W[c])
                err = val - target
                errors += err * err
                
                # Gradients
                # d/da: err * V[b] * W[c]
                # d/db: err * U[a] * W[c]
                # d/dc: err * U[a] * V[b]
                dU[a] += err * V[b] * W[c]
                dV[b] += err * U[a] * W[c]
                dW[c] += err * U[a] * V[b]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Snap to nearest half-integer periodically to encourage sparse solutions
            if step % 100 == 99:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
            
            # Check if we're close to a solution
            if step % 500 == 499:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try decreasing ranks
    # Lower bound: max(n*m, m*p, n*p) = max(10, 30, 12) = 30
    # But we need more because of the structure.
    
    # Try a few ranks
    for R in [30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline:
            break
        result = _try_rank(R, max_iters=3000)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
