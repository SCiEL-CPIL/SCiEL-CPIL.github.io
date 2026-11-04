import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def check_decomposition_fast(U, V, W, tol=1e-9):
        """
        Vectorized check for exact decomposition.
        U: (nm, R) -> reshape to (n, m, R)
        V: (mp, R) -> reshape to (m, p, R)
        W: (pn, R) -> reshape to (p, n, R)
        Target: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1
        """
        R = U.shape[1]
        U_reshaped = U.reshape(n, m, R)
        V_reshaped = V.reshape(m, p, R)
        W_reshaped = W.reshape(p, n, R)
        
        # Compute the tensor T
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # Using einsum for clarity and speed
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, 1.0, atol=tol)

    def try_rank_random(R, num_attempts=50000):
        """Try random search for a valid decomposition of rank R using vectorized check."""
        # We use small integers to increase the chance of exact hits
        # Values in {-1, 0, 1} or {-2, -1, 0, 1, 2}
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Random integers
            U = np.random.randint(-1, 2, size=(nm, R))
            V = np.random.randint(-1, 2, size=(mp, R))
            W = np.random.randint(-1, 2, size=(pn, R))
            
            if check_decomposition_fast(U, V, W):
                return U, V, W
        return None

    # Strategy:
    # 1. Try to find a rank < 60 using fast random search.
    # 2. If found, return it.
    # 3. Otherwise, return schoolbook (rank 60).
    
    # We know rank 60 is possible.
    # The theoretical lower bound is max(n*m, m*p, p*n) = 30.
    # Let's try to find a rank < 60.
    
    # Try ranks from 59 down to 50 with random search.
    # Since random search is now much faster, we can try more ranks.
    
    for R in range(59, 49, -1):
        if time.time() > deadline - 5:
            break
        
        result = try_rank_random(R, num_attempts=20000)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
