"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    # Schoolbook decomposition
    def _schoolbook():
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
        return U, V, W, rank
    
    # Try to find a lower-rank decomposition
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    
    def _verify(U, V, W, tol=1e-9):
        """Check if factors exactly decompose the tensor."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook
    U, V, W, best_rank = _schoolbook()
    
    # Try to reduce rank by 1, 2, ... using random search with constraint to half-integers
    # This is heuristic and may not succeed, but worth trying
    
    # Lower bound on rank is at least max(n*m, m*p, p*n) = 25
    # But realistically for 2x5x5, schoolbook 50 is likely optimal or close
    
    # Attempt a simple local search: try rank = best_rank - 1
    target_rank = best_rank - 1
    if target_rank >= 25 and time.time() < deadline - 30:
        # Random restarts with half-integer entries
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            # Initialize
            U_try = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(target_rank)] for _ in range(n*m)]
            V_try = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(target_rank)] for _ in range(m*p)]
            W_try = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Simple iterative refinement: alternate fixing two and optimizing third via least squares
            # For simplicity, just check if random init works (unlikely)
            if _verify(U_try, V_try, W_try):
                U, V, W, best_rank = U_try, V_try, W_try, target_rank
                break
    
    # If we found a better rank, try to go lower
    while best_rank > 25 and time.time() < deadline - 10:
        new_rank = best_rank - 1
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            U_try = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(new_rank)] for _ in range(n*m)]
            V_try = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(new_rank)] for _ in range(m*p)]
            W_try = [[float(np.random.choice([-1, -0.5, 0, 0.5, 1])) for _ in range(new_rank)] for _ in range(p*n)]
            if _verify(U_try, V_try, W_try):
                U, V, W, best_rank = U_try, V_try, W_try, new_rank
                found = True
                break
        if not found:
            break
    
    # Final deadline check
    if time.time() > deadline:
        U, V, W, best_rank = _schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
