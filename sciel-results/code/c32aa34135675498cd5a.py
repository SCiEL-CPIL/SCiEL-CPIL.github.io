"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # For matrix mult: T[i,j,k] = delta_{j,k} (i.e., C[i,k] = sum_j A[i,j]*B[j,k])
        # So T[i*m+j, j*p+k, k*n+i] should be 1, all else 0
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute tensor T from decomposition
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Shape: (n*m, m*p, p*n)
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Expected: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # All other entries = 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected_idx = (i * m + j, j * p + k, k * n + i)
                    err = abs(T[expected_idx] - 1.0)
                    max_err = max(max_err, err)
        
        # Check that all non-expected entries are ~0
        # Create expected tensor
        T_expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0
        
        err_matrix = T - T_expected
        max_err = max(max_err, np.max(np.abs(err_matrix)))
        
        return max_err < 1e-10
    
    def try_rank(rank, max_iterations=5000, seed=42):
        """Try to find a rank-rank decomposition using iterative refinement."""
        rng = np.random.RandomState(seed)
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random sparse patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for c in range(rank):
                # Each factor picks one index from each dimension
                i = rng.randint(n)
                j = rng.randint(m) 
                k = rng.randint(p)
                
                U[i*m + j, c] = rng.choice([-1, 1])
                V[j*p + k, c] = rng.choice([-1, 1])
                W[k*n + i, c] = rng.choice([-1, 1])
            
            # Check if this random initialization works
            if verify(U, V, W, rank):
                return U.tolist(), V.tolist(), W.tolist()
            
            # Gradient descent style refinement
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor
                T = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Expected tensor
                T_expected = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_expected[i*m+j, j*p+k, k*n+i] = 1.0
                
                err = T - T_expected
                curr_err = np.sum(err**2)
                
                if curr_err < best_err:
                    best_err = curr_err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if curr_err < 1e-10:
                    return U.tolist(), V.tolist(), W.tolist()
                
                # Compute gradients
                # dT/dr_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
                # grad_U[a,c] = sum_{b,d} err[a,b,d] * V[b,c] * W[d,c]
                grad_U = np.einsum('abd,bc,dc->ac', err, V, W)
                grad_V = np.einsum('abd,ac,dc->bc', err, U, W)
                grad_W = np.einsum('abd,ac,bc->dc', err, U, V)
                
                # Adaptive learning rate
                lr = 0.1 / (1 + curr_err)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Occasionally try different initialization strategy
            if iteration % 100 == 0:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                
                # Try structured initialization based on known patterns
                for c in range(rank):
                    # Try to cover (i,j,k) tuples more systematically
                    i = (c // (m*p)) % n
                    j = (c // p) % m
                    k = c % p
                    
                    U[i*m + j, c] = 1.0
                    V[j*p + k, c] = 1.0
                    W[k*n + i, c] = 1.0
                
                if verify(U, V, W, rank):
                    return U.tolist(), V.tolist(), W.tolist()
        
        # Return best found if it's close
        if best_err < 1e-8 and best_U is not None:
            # Try to clean up
            U = best_U
            V = best_V
            W = best_W
            
            # Round to nearest half-integer or integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded, rank):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        return None
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank decompositions
    # Theoretical bounds: rank >= max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Known: 3x3x3 has rank 23 (Lader), so 3x3x6 >= 23
    # Try ranks from lower bound upward
    
    # First try some promising ranks
    candidate_ranks = [23, 24, 25, 26, 27, 28, 29, 30, 32, 36]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_iterations=2000, seed=rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If we didn't find anything better, return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
