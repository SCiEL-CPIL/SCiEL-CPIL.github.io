import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_target = 28
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if A[a] and B[b] contribute to C[c]
    # a = i*m + j
    # b = j*p + k
    # c = i*p + k (Wait, W index is k*n + i? No, task says W[k*n+i, r])
    # Let's check the task description: "output factor follows the official trace convention W[k*n+i,r]"
    # This means the index for W is $c = k \cdot n + i$.
    # Let's re-verify the tensor indices.
    # A is n x m. Index a = i*m + j.
    # B is m x p. Index b = j*p + k.
    # C is n x p. Index c = i*p + k.
    # But the task says W index is k*n + i. This is a transposed/flattened order for C?
    # C is usually n x p. Flattened row-major is i*p + k.
    # Flattened column-major is k*n + i.
    # The task says "W[k*n+i, r]". So the index c in W is $k \cdot n + i$.
    # So for the tensor equation:
    # Sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
    # T[a, b, c] is 1 if the indices match.
    
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    T = np.zeros((a_dim, b_dim, c_dim), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # As per task convention
                T[a, b, c] = 1

    # We want to find U, V, W of rank R such that sum_r U[:,r] V[:,r] W[:,r] = T
    # U shape (a_dim, R), V shape (b_dim, R), W shape (c_dim, R)
    
    # Start with Schoolbook R=32
    R_school = n * m * p
    U_school = np.zeros((a_dim, R_school), dtype=int)
    V_school = np.zeros((b_dim, R_school), dtype=int)
    W_school = np.zeros((c_dim, R_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    # Verify Schoolbook
    # prod = np.einsum('ar,br,cr->abc', U_school, V_school, W_school)
    # assert np.array_equal(prod, T)

    # Try to find rank 28
    # Use a random search with local improvements
    start_time = time.time()
    timeout = 110 # seconds
    
    best_rank = R_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # If we have time, try to find rank 28
    # A known rank 28 solution exists.
    # I will attempt a randomized search.
    
    def error(U, V, W):
        # Compute tensor product
        # This is expensive, so we only do it for candidates
        prod = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((prod - T)**2)
    
    # Strategy: Random restarts with simulated annealing on integer entries
    # Since the solution is known to have small integers, we can search in {-2, -1, 0, 1, 2}
    
    found = False
    attempts = 0
    while time.time() - start_time < timeout and not found:
        attempts += 1
        R = 28
        U = np.zeros((a_dim, R), dtype=int)
        V = np.zeros((b_dim, R), dtype=int)
        W = np.zeros((c_dim, R), dtype=int)
        
        # Initialize with random small integers
        for r in range(R):
            # Sparse initialization
            nz_u = random.randint(2, 5)
            nz_v = random.randint(2, 5)
            nz_w = random.randint(2, 5)
            
            idx_u = random.sample(range(a_dim), nz_u)
            idx_v = random.sample(range(b_dim), nz_v)
            idx_w = random.sample(range(c_dim), nz_w)
            
            for idx in idx_u:
                U[idx, r] = random.choice([-1, 1])
            for idx in idx_v:
                V[idx, r] = random.choice([-1, 1])
            for idx in idx_w:
                W[idx, r] = random.choice([-1, 1])
                
        # Local search: flip bits or change values
        for step in range(1000):
            if time.time() - start_time > timeout:
                break
            
            # Pick a random matrix, row, col
            mat_name = random.choice(['U', 'V', 'W'])
            mat = {'U': U, 'V': V, 'W': W}[mat_name]
            dim = {'U': a_dim, 'V': b_dim, 'W': c_dim}[mat_name]
            
            r_idx = random.randint(0, R-1)
            c_idx = random.randint(0, dim-1)
            
            old_val = mat[c_idx, r_idx]
            new_val = random.choice([-2, -1, 0, 1, 2])
            if new_val == old_val:
                continue
                
            mat[c_idx, r_idx] = new_val
            
            err = error(U, V, W)
            if err == 0:
                found = True
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
            else:
                # Accept if better, or with some probability
                # We need a baseline error to compare
                # This simple hill climbing might get stuck.
                # Let's just keep the best found so far in this restart
                # Actually, let's just try to drive error to 0.
                # If error increases, revert.
                mat[c_idx, r_idx] = old_val
        
        if found:
            break
            
    if found:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
