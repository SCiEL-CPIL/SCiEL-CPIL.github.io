"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        # Verify shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check the tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # for all i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Improved random search with better heuristics
    
    def check_decomposition_fast(U, V, W):
        """Fast check using NumPy."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.int8)
        V_arr = np.array(V, dtype=np.int8)
        W_arr = np.array(W, dtype=np.int8)
        
        # Reshape to 3D tensors
        U_tensor = U_arr.reshape(n, m, rank)
        V_tensor = V_arr.reshape(m, p, rank)
        W_tensor = W_arr.reshape(p, n, rank)
        
        # Compute the tensor product and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U_tensor[i, j] * V_tensor[j, k] * W_tensor[k, i])
                    if total != 1:
                        return False
        return True
    
    # Try to find a better decomposition using randomized search
    np.random.seed(42)
    
    # Try smaller ranks first with more attempts
    for target_rank in [22, 23, 24, 25, 26, 27, 28, 30, 32, 36, 40, 48]:
        if time.time() > deadline - 10:
            break
        
        # More attempts for smaller ranks
        attempts = 50 if target_rank <= 25 else 20
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if check_decomposition_fast(U_list, V_list, W_list):
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
