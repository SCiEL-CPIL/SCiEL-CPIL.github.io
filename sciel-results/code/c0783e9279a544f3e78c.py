import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 72
    
    def _verify(U, V, W):
        """Verify if U, V, W decompose the matrix multiplication tensor.
        
        Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check the tensor condition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Start with schoolbook decomposition (rank 72)
    rank = rank_school
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Time budget
    start_time = time.time()
    time_limit = 140  # Leave 10s margin
    
    # Try to find lower rank decompositions using structured search
    for target_rank in range(71, 45, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try several random initializations with integer/half-integer values
        found = False
        for attempt in range(15):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(attempt * 10000 + target_rank * 100)
            
            # Generate sparse random matrices with values in {±1, ±0.5}
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Fill with sparse random values
            density = 0.25
            for c in range(target_rank):
                for idx in range(n * m):
                    if np.random.random() < density:
                        U_try[idx, c] = np.random.choice([1, -1, 0.5, -0.5])
                for idx in range(m * p):
                    if np.random.random() < density:
                        V_try[idx, c] = np.random.choice([1, -1, 0.5, -0.5])
                for idx in range(p * n):
                    if np.random.random() < density:
                        W_try[idx, c] = np.random.choice([1, -1, 0.5, -0.5])
            
            # Verify
            if _verify(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                found = True
                break
        
        if found:
            break
    
    # Try a greedy removal approach if still at schoolbook rank
    if best_rank == 72 and time.time() - start_time < time_limit - 5:
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()
        current_rank = rank
        removed = [False] * rank
        
        # Try to remove terms one by one
        for c_remove in range(rank - 1, -1, -1):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Temporarily remove term c_remove
            removed[c_remove] = True
            
            # Check if the remaining terms still satisfy the condition
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            if not removed[c]:
                                val += U_work[i*m+j, c] * V_work[j*p+k, c] * W_work[k*n+i, c]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                current_rank -= 1
            else:
                removed[c_remove] = False
        
        if current_rank < best_rank:
            keep_indices = [c for c in range(rank) if not removed[c]]
            best_rank = current_rank
            best_U = U_work[:, keep_indices].copy()
            best_V = V_work[:, keep_indices].copy()
            best_W = W_work[:, keep_indices].copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
