"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]
DEADLINE = 150  # seconds, evaluator kills at 150s wall-clock

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    start = time.time()
    
    # We will try to find a low-rank decomposition.
    # The schoolbook rank is n*m*p = 2*3*9 = 54.
    # We will use a search to find a lower rank, if possible.
    # Known results for 2x3x9 matrix multiplication:
    # The rank is at most 54 (schoolbook).
    # We can try to reduce it.
    # Let's try a few random initializations and local search.
    # The tensor T has indices (i,j,k) with i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    
    # We'll use a simple local search: start with a random rank-r tensor,
    # and try to minimize the error.
    
    # First, let's try to see if we can do better than 54.
    # We'll try ranks from 54 down to some lower bound.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # So we can try ranks from 27 to 54.
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # We'll use a time limit for the search.
    time_limit = 120  # seconds, leave 30s safety margin
    
    # For each candidate rank, try a few random initializations.
    for rank in range(n*m*p, 26, -1):
        if time.time() - start > time_limit:
            break
            
        # Try a few random initializations
        for trial in range(5):
            if time.time() - start > time_limit:
                break
                
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Local search: gradient descent on the error
            # Error = || T - sum_c U[:,c] V[:,c] W[:,c] ||^2
            # We'll use a simple gradient step.
            
            # Build the target tensor in vector form
            # T is a tensor of shape (n*m, m*p, p*n)
            # We'll flatten it to a vector for easier handling.
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] = 1
            
            T_flat = T.flatten()
            
            # Local search
            lr = 0.01
            for step in range(100):
                if time.time() - start > time_limit:
                    break
                    
                # Current error
                # sum_c U[:,c] outer V[:,c] outer W[:,c]
                # This is expensive for large rank, so we'll compute it efficiently.
                # For now, let's just compute the error directly.
                
                # Compute the current tensor
                C = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    C += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                
                C_flat = C.flatten()
                err = np.dot(C_flat - T_flat, C_flat - T_flat)
                
                if err < 1e-6:
                    # Found a valid decomposition
                    # Convert to integer/half-integer if possible
                    U_int = np.rint(U * 2) / 2
                    V_int = np.rint(V * 2) / 2
                    W_int = np.rint(W * 2) / 2
                    
                    # Verify the integer/half-integer version
                    C_int = np.zeros((n*m, m*p, p*n))
                    for c in range(rank):
                        C_int += np.einsum('a,b,c->abc', U_int[:,c], V_int[:,c], W_int[:,c])
                    
                    if np.allclose(C_int, T, atol=1e-6):
                        best_rank = rank
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        break
                
                # Gradient computation
                # dErr/dU[a,c] = 2 * sum_{b,d} (C[a,b,d] - T[a,b,d]) * V[b,c] * W[d,c]
                # This is expensive. Let's use a simpler approach: random coordinate descent.
                
                # Pick a random coordinate
                mat = random.choice(['U', 'V', 'W'])
                idx1 = random.randint(0, (n*m, m*p, p*n)[mat == 'U' or mat == 'V' or mat == 'W'] - 1)
                idx1 = random.randint(0, (n*m if mat=='U' else m*p if mat=='V' else p*n) - 1)
                idx2 = random.randint(0, rank - 1)
                
                # Try a small perturbation
                delta = random.choice([-1, 1]) * random.uniform(0.01, 0.1)
                
                if mat == 'U':
                    U[idx1, idx2] += delta
                elif mat == 'V':
                    V[idx1, idx2] += delta
                else:
                    W[idx1, idx2] += delta
                
                # Recompute error
                C = np.zeros((n*m, m*p, p*n))
                for c in range(rank):
                    C += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                C_flat = C.flatten()
                new_err = np.dot(C_flat - T_flat, C_flat - T_flat)
                
                if new_err >= err:
                    # Revert
                    if mat == 'U':
                        U[idx1, idx2] -= delta
                    elif mat == 'V':
                        V[idx1, idx2] -= delta
                    else:
                        W[idx1, idx2] -= delta
        
        if best_rank < n*m*p:
            break
    
    # If no better decomposition found, use the schoolbook one
    if best_U is None:
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
    
    # Convert numpy arrays to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to nearest 0.5 if not already
    U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
    V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
    W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
