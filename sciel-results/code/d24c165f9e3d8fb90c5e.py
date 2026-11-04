import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s wall-clock limit

    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72

    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    def check_decomposition(U, V, W, rank):
        """Check if U, V, W of given rank satisfy the decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def check_decomposition_fast(U, V, W, rank):
        """Vectorized check using numpy."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    # Strategy: Try to find structured lower-rank decompositions
    # Key idea: Use the algebraic structure to combine schoolbook terms
    
    # Approach 1: Try rank 71, 70, etc. by finding if any schoolbook term can be eliminated
    # by adjusting others
    
    # Approach 2: Use known results for smaller cases and extend
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8
    # For 3x3x3, best known is 23 (Lader 1976) vs schoolbook 27
    
    # For 3x3x8, we can think of it as 3 independent 3x3x1 multiplications? No, they share A.
    # Actually, C[:,k] = A @ B[:,k] for each k. So it's like computing 8 matrix-vector products.
    
    # Let me try a different structured approach:
    # The tensor T_{i,j,k} = 1 for all i,j,k
    # We want rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r)_{i,j} = U[i*m+j, r], (v_r)_{j,k} = V[j*p+k, r], (w_r)_{k,i} = W[k*n+i, r]
    
    # Try to find a decomposition with some structure.
    
    # Let me try: for each (i,j), the vector across k is constant (all 1s in V part for fixed j,k)
    # Actually no, V[j*p+k, r] varies with k.
    
    # Let me try a random search with better initialization:
    # Start near schoolbook and perturb
    
    random.seed(12345)
    
    def try_perturbed_schoolbook(target_rank, n_attempts=50):
        """Try to find a decomposition of target_rank by perturbing schoolbook."""
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Start with first target_rank schoolbook terms
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with first target_rank schoolbook columns
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            
            if col < target_rank:
                # Not enough schoolbook terms, fill rest randomly
                for r in range(col, target_rank):
                    for idx in range(n * m):
                        if random.random() < 0.3:
                            U[idx][r] = random.choice([-1, 1])
                    for idx in range(m * p):
                        if random.random() < 0.3:
                            V[idx][r] = random.choice([-1, 1])
                    for idx in range(p * n):
                        if random.random() < 0.3:
                            W[idx][r] = random.choice([-1, 1])
            
            # Try to fix the remaining terms by adjusting values
            # This is complex, so just check
            if check_decomposition_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
            
            # Try small perturbations
            for _ in range(10):
                # Pick a random entry to perturb
                mat = random.choice([U, V, W])
                rows = len(mat)
                row = random.randint(0, rows - 1)
                col_idx = random.randint(0, target_rank - 1)
                old_val = mat[row][col_idx]
                new_val = old_val + random.choice([-1, 1])
                mat[row][col_idx] = new_val
                
                if check_decomposition_fast(U, V, W, target_rank):
                    return {"U": U, "V": V, "W": W}
                
                # Revert if not helpful
                mat[row][col_idx] = old_val
        
        return None

    # Try various ranks
    for target_rank in range(71, 55, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_perturbed_schoolbook(target_rank, n_attempts=30)
        if result:
            best_result = result
            best_rank = target_rank
            break
    
    # Also try pure random search for lower ranks
    def try_random_rank(target_rank, n_attempts=20):
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                return None
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for r in range(target_rank):
                # Sparse random
                for idx in range(n * m):
                    if random.random() < 0.4:
                        U[idx][r] = random.choice([-1, 1])
                for idx in range(m * p):
                    if random.random() < 0.4:
                        V[idx][r] = random.choice([-1, 1])
                for idx in range(p * n):
                    if random.random() < 0.4:
                        W[idx][r] = random.choice([-1, 1])
            
            if check_decomposition_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        
        return None

    for target_rank in range(70, 50, -1):
        if time.time() > deadline - 5:
            break
        
        result = try_random_rank(target_rank, n_attempts=10)
        if result:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
