import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    T = np.zeros((size_u, size_v, size_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def calculate_error(U, V, W):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((R - T) ** 2)

    def local_search(U, V, W, max_steps=10000):
        current_error = calculate_error(U, V, W)
        if current_error == 0:
            return U, V, W, True
            
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        rank = U.shape[1]
        
        for step in range(max_steps):
            if time.time() > deadline:
                return None, None, None, False
                
            # Pick a random matrix
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
                
            row = np.random.randint(mat.shape[0])
            col = np.random.randint(mat.shape[1])
            
            # Try +/- 1
            original_val = mat[row, col]
            
            # Try +1
            mat[row, col] = original_val + 1
            new_error = calculate_error(U, V, W)
            if new_error < current_error:
                current_error = new_error
                if current_error == 0:
                    return U, V, W, True
            else:
                mat[row, col] = original_val
                
            # Try -1
            mat[row, col] = original_val - 1
            new_error = calculate_error(U, V, W)
            if new_error < current_error:
                current_error = new_error
                if current_error == 0:
                    return U, V, W, True
            else:
                mat[row, col] = original_val
                
        return U, V, W, False

    # Try rank 23
    rank = 23
    num_attempts = 50 # Adjust based on time
    
    for attempt in range(num_attempts):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(42 + attempt)
        # Initialize with small integers
        U = np.random.randint(-2, 3, (size_u, rank))
        V = np.random.randint(-2, 3, (size_v, rank))
        W = np.random.randint(-2, 3, (size_w, rank))
        
        U, V, W, found = local_search(U, V, W, max_steps=5000)
        
        if found:
            # Verify one last time
            if calculate_error(U, V, W) == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
