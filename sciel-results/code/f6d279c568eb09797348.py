import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Target Tensor T
    # T[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i
    # Non-zero when: u=3i+j, v=3j+k, w=3k+i
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    def check_solution(U, V, W, r):
        """
        U: (9, r)
        V: (9, r)
        W: (9, r)
        Returns True if decomposition is exact.
        """
        # Compute reconstructed tensor
        # T_rec[u, v, w] = sum_k U[u, k] * V[v, k] * W[w, k]
        for w in range(p*n):
            for v in range(m*p):
                for u in range(n*m):
                    s = 0
                    for k in range(r):
                        s += U[u, k] * V[v, k] * W[w, k]
                    if s != T[u, v, w]:
                        return False
        return True

    def solve_for_W(U, V, r):
        """
        Solves for W given U, V.
        Returns W if a clean solution exists, else None.
        """
        W = np.zeros((p*n, r))
        
        for w in range(p*n):
            A = np.zeros((n*m * m*p, r))
            b = np.zeros(n*m * m*p)
            
            idx = 0
            for u in range(n*m):
                for v in range(m*p):
                    for k in range(r):
                        A[idx, k] = U[u, k] * V[v, k]
                    b[idx] = T[u, v, w]
                    idx += 1
                    
            # Solve A x = b
            # We want rational/integer solutions.
            # Use lstsq to get a candidate
            try:
                x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
                
            # Check residuals
            if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-9:
                return None
            if residuals is None: 
                 if not np.allclose(A @ x, b):
                     return None
            
            # Check if x is close to integers or half-integers
            # The problem allows integer/half-integer entries.
            # So 2*x should be integer.
            x_scaled = 2.0 * x
            x_rounded = np.round(x_scaled)
            if np.max(np.abs(x_scaled - x_rounded)) > 1e-9:
                return None
            
            W[w, :] = x_rounded / 2.0
            
        return W

    # Search strategy
    # Try ranks from low to high
    # Start from 9 (theoretical min) up to 26 (below schoolbook 27)
    
    random.seed(42)
    
    # We prioritize finding a valid decomposition.
    # We will try a few iterations for each rank.
    # If we find one, we return it.
    
    max_tries_per_rank = 20
    ranks_to_try = list(range(9, 27))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        for _ in range(max_tries_per_rank):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            # Entries from {-1, 0, 1}
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Sparse random
            for i in range(n*m):
                for k in range(r):
                    if random.random() < 0.5:
                        U[i, k] = random.choice([-1, 1])
                        
            for i in range(m*p):
                for k in range(r):
                    if random.random() < 0.5:
                        V[i, k] = random.choice([-1, 1])
                        
            # Solve for W
            W = solve_for_W(U, V, r)
            
            if W is not None:
                # Verify
                if check_solution(U, V, W, r):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }

    # If no better rank found, return schoolbook
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
# EVOLVE-BLOCK-END
