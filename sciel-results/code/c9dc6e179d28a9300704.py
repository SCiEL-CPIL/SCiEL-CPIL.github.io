"""Matrix multiplication 2x3x6 - improved decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 36
    # Try to find lower rank decompositions
    
    # First, let's establish the schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # The tensor T[i,j,k] = 1 for all i,j,k (this is the multiplication tensor)
    # We need U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    def compute_tensor(U, V, W):
        """Vectorized computation of the tensor contraction."""
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                # Ur[i, j, :] is (r,)
                # Vr[j, :, :] is (p, r)
                # Wr[:, i, :] is (p, r)
                # T[i, j, k] = sum_r Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                BC = Vr[j, :, :] * Wr[:, i, :] # (p, r)
                T[i, j, :] = np.sum(Ur[i, j, :] * BC, axis=1)
        return T

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        T = compute_tensor(U, V, W)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def random_search(rank, iterations=10000):
        """Try to find a rank-r decomposition via random search with half-integers."""
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # Use half-integer values (multiples of 0.5) to satisfy the problem constraint
            # Scale: values in {-2, -1, 0, 1, 2} * 0.5
            # Or just random integers scaled?
            # Let's try random integers in a small range, then divide by 2
            U = np.random.randint(-4, 5, size=(n*m, rank)) * 0.5
            V = np.random.randint(-4, 5, size=(m*p, rank)) * 0.5
            W = np.random.randint(-4, 5, size=(p*n, rank)) * 0.5
            
            # Compute error
            T = compute_tensor(U, V, W)
            err = np.sum((T - 1.0) ** 2)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            # Early stop if we found exact solution
            if err < 1e-12:
                return best_U, best_V, best_W, err
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from 35 down to 30 (closest to schoolbook, more likely to exist if they do)
    # We prioritize higher ranks first as they are more likely to have solutions
    for rank in range(35, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Random search
        U, V, W, err = random_search(rank, iterations=5000)
        
        if err < 1e-9:
            # Verify
            if verify(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    # If we found something better, return it
    if best_rank < n * m * p:
        return best_result
    
    # Otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
