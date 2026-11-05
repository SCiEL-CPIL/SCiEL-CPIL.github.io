"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Schoolbook as fallback (rank 72)
    rank_school = n * m * p  # 72
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Precompute index arrays for fast vectorized error computation
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def check_decomposition_exact(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication using integer arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        # Also verify that non-target entries are zero
        # For efficiency, check a sample of non-target entries
        for _ in range(50):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            # This is a target entry, so skip
            # Instead, check random (a, b, c) that are not target
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            # Check if (a, b, c) is a target entry
            is_target = False
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        if ii * m + jj == a and jj * p + kk == b and kk * n + ii == c:
                            is_target = True
                            break
                    if is_target:
                        break
                if is_target:
                    break
            if not is_target:
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != 0:
                    return False
        return True

    def compute_error_vectorized(U_arr, V_arr, W_arr, rank):
        """Compute sum of squared errors using vectorized ops."""
        total = np.zeros(n * m * p, dtype=np.float64)
        for r in range(rank):
            total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
        return np.sum((total - 1.0) ** 2)

    # Strategy: Use block structure and try to find linear dependencies
    # The key insight is that for each (i,j) pair, the equation is:
    # sum_k U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
    # This means for fixed (i,j), the vector [V[j*p+0,r]*W[0*n+i,r], ..., V[j*p+7,r]*W[7*n+i,r]]
    # must satisfy: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    
    # Let's try a different approach: use the fact that the problem has a product structure
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
    
    # This can be written as: for each r, we have vectors u_r, v_r, w_r
    # and we need sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # Let's try to use a structured initialization based on the Kronecker product structure
    # The 3x3x8 matmul tensor = (3x3 matmul tensor) ⊗ (8x1 selection tensor)
    # Actually, it's more like: T[i,j,k] = 1 for all i,j,k
    
    # Let's try rank reduction by checking if we can merge schoolbook terms
    # For terms (i,j,k) and (i',j',k'), we can try to express one as a combination
    
    # Better approach: use the block structure to create independent sub-problems
    # For each k, the terms involving k form a 3x3 sub-problem
    # But the W matrix couples k and i, so we need to be careful
    
    # Let's try a greedy approach: start with schoolbook and try to remove terms
    # by expressing them as linear combinations of others
    
    U_arr = np.array(U_school, dtype=np.float64)
    V_arr = np.array(V_school, dtype=np.float64)
    W_arr = np.array(W_school, dtype=np.float64)
    
    # Try to reduce rank by finding linear dependencies
    # For each column r, check if it can be expressed as a combination of others
    # This is complex, so let's try a simpler approach:
    
    # Use the fact that the tensor has a product structure
    # T[i,j,k] = 1 = (sum over i' of something) * (sum over j' of something) * ...
    
    # Actually, let's try a direct optimization approach with better initialization
    # Initialize with block structure and use local search
    
    # Try different target ranks
    for target_rank in range(70, 50, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        max_attempts = 5 if target_rank <= 65 else 3
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
            
            # Initialize with block structure
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Allocate terms to blocks based on k-value
            # For k=0,1,2: use 9 terms (schoolbook for 3x3x3)
            # For k=3,4,5: use 9 terms
            # For k=6,7: use 6 terms
            # Total: 24 terms minimum
            
            if target_rank >= 24:
                # Use a structured initialization
                # For each k, assign terms to cover the 3x3 input space
                terms_per_k = [9, 9, 6]  # for k=0-2, k=3-5, k=6-7
                total_terms = sum(terms_per_k)
                
                if target_rank >= total_terms:
                    # Initialize with schoolbook for each block
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if col < target_rank:
                                    U[i * m + j][col] = 1
                                    V[j * p + k][col] = 1
                                    W[k * n + i][col] = 1
                                    col += 1
                    # If we have extra terms, add random perturbations
                    for r in range(col, target_rank):
                        for i in range(n):
                            for j in range(m):
                                if random.random() < 0.1:
                                    U[i * m + j][r] = random.choice([-1, 1])
                        for j in range(m):
                            for k in range(p):
                                if random.random() < 0.1:
                                    V[j * p + k][r] = random.choice([-1, 1])
                        for k in range(p):
                            for i in range(n):
                                if random.random() < 0.1:
                                    W[k * n + i][r] = random.choice([-1, 1])
            
            # Local search with gradient-like updates
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            
            best_err = compute_error_vectorized(U_arr, V_arr, W_arr, target_rank)
            
            for step in range(300):
                if time.time() > deadline - 10:
                    break
                
                err = compute_error_vectorized(U_arr, V_arr, W_arr, target_rank)
                
                if err < 1e-6:
                    # Try to round to integers and verify exactly
                    U_int = [[int(round(x)) for x in row] for row in U_arr.tolist()]
                    V_int = [[int(round(x)) for x in row] for row in V_arr.tolist()]
                    W_int = [[int(round(x)) for x in row] for row in W_arr.tolist()]
                    if check_decomposition_exact(U_int, V_int, W_int, target_rank):
                        best_rank = target_rank
                        best_result = {"U": U_int, "V": V_int, "W": W_int}
                        found = True
                        break
                
                if err < best_err * 0.999:
                    best_err = err
                
                # Perturb a random element
                mat_idx = random.choice([0, 1, 2])
                if mat_idx == 0:
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, target_rank - 1)
                    U_arr[row, col] += random.choice([-1, 1])
                elif mat_idx == 1:
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, target_rank - 1)
                    V_arr[row, col] += random.choice([-1, 1])
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, target_rank - 1)
                    W_arr[row, col] += random.choice([-1, 1])
            
            if found:
                break
        
        if found:
            break
    
    # Final check: if we found something better, return it; else return schoolbook
    if best_rank < rank_school:
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
