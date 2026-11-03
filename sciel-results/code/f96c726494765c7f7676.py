"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    n, m, p = 3, 3, 5
    # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (15, rank)
    # Convention: W[k*n+i, r]

    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        rank = len(U[0])
        # Check all (i, j, k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W

    # Start with schoolbook and try to reduce rank
    U, V, W = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W

    # Try various ranks from low to high
    for target_rank in range(15, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        # Try random initialization and local search
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < deadline - 20:
            attempts += 1
            
            # Random initialization with small values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Simple gradient-free local search
            for it in range(200):
                if time.time() > deadline - 20:
                    break
                    
                # Compute current error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(target_rank):
                                total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                            error += (total - 1.0) ** 2
                
                if error < 1e-10:
                    # Convert to list format
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if verify(U_list, V_list, W_list):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U, best_V, best_W = U_list, V_list, W_list
                    break
                
                # Random perturbation
                idx = random.randint(0, n*m + m*p + p*n - 1)
                if idx < n*m:
                    row, col = idx, random.randint(0, target_rank - 1)
                    U[row][col] += random.gauss(0, 0.1)
                elif idx < n*m + m*p:
                    row, col = idx - n*m, random.randint(0, target_rank - 1)
                    V[row][col] += random.gauss(0, 0.1)
                else:
                    row, col = idx - n*m - m*p, random.randint(0, target_rank - 1)
                    W[row][col] += random.gauss(0, 0.1)
            
            if time.time() > deadline - 20:
                break
        
        if best_rank <= target_rank:
            break

    # Return the best found
    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
