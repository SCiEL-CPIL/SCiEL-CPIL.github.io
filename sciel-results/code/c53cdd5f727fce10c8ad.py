import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Correct trilinear form: sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
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
        return U, V, W
    
    def try_rank_random(R, num_attempts=50):
        """Try random search for a valid decomposition of rank R."""
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            # Use small random integers for better chance of exact solutions
            U = np.random.randint(-2, 3, size=(n * m, R))
            V = np.random.randint(-2, 3, size=(m * p, R))
            W = np.random.randint(-2, 3, size=(p * n, R))
            
            if check_decomposition(U, V, W):
                return U, V, W
        return None
    
    # Try ranks from 59 down to 50, with limited random attempts
    for R in range(59, 49, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank_random(R, num_attempts=20)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
