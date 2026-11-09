import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    # Schoolbook fallback
    rank_school = n * m * p
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
        
    # Try to find a lower rank decomposition
    # Target rank 28
    target_rank = 28
    if time.time() < deadline - 20:
        result = find_decomposition(n, m, p, target_rank, deadline)
        if result:
            return result
            
    return {"U": U_school, "V": V_school, "W": W_school}

def find_decomposition(n, m, p, rank, deadline):
    # Randomized search for a rank R decomposition
    # We represent U, V, W as lists of lists
    # We use a simple local search
    
    # Initialize with random small integers
    U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
    
    # Compute the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i, j, k] = 1 if i,j,k are valid indices for the multiplication tensor
    # The multiplication tensor T is defined by T[i, j, k] = 1 if the index corresponds to a product term
    # Actually, the tensor for matrix multiplication C = AB is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So the tensor T has indices i (row of A), j (col of A/row of B), k (col of B)
    # T[i, j, k] = 1 for all valid i, j, k.
    # Wait, the tensor is defined such that the bilinear form is sum T[i, j, k] A[i, j] B[j, k] = C[i, k]
    # So T[i, j, k] is 1 if the index (i, j, k) corresponds to a term in the sum.
    # For standard multiplication, every (i, j, k) is a term.
    # So T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    
    # The decomposition is:
    # sum_r U[i, r] V[j, r] W[k, r] = T[i, j, k]
    # where U is indexed by (i, j)? No.
    # U is indexed by (i, j) where i is row of A, j is col of A.
    # V is indexed by (j, k) where j is row of B, k is col of B.
    # W is indexed by (k, i) where k is col of B, i is row of A.
    # Wait, the standard decomposition is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This is a sum over j.
    # The tensor decomposition is:
    # T = sum_r u_r \otimes v_r \otimes w_r
    # where u_r is a vector of size n*m, v_r is a vector of size m*p, w_r is a vector of size p*n.
    # And T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T[i, j, k] = 1 for all i, j, k.
    
    # So we want to find U, V, W such that for all i, j, k:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations.
    # We can try to satisfy them.
    
    # Let's check the error
    def compute_error(U, V, W):
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if val != 1:
                        error += 1
        return error
    
    # Local search
    best_error = compute_error(U, V, W)
    if best_error == 0:
        return {"U": U, "V": V, "W": W}
    
    # Simple hill climbing
    while time.time() < deadline - 5:
        # Pick a random element to change
        mat = random.choice(["U", "V", "W"])
        if mat == "U":
            idx = random.randint(0, n*m-1)
            r = random.randint(0, rank-1)
            old_val = U[idx][r]
            new_val = random.choice([-1, 0, 1])
            U[idx][r] = new_val
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
                if best_error == 0:
                    return {"U": U, "V": V, "W": W}
            else:
                U[idx][r] = old_val
        elif mat == "V":
            idx = random.randint(0, m*p-1)
            r = random.randint(0, rank-1)
            old_val = V[idx][r]
            new_val = random.choice([-1, 0, 1])
            V[idx][r] = new_val
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
                if best_error == 0:
                    return {"U": U, "V": V, "W": W}
            else:
                V[idx][r] = old_val
        else:
            idx = random.randint(0, p*n-1)
            r = random.randint(0, rank-1)
            old_val = W[idx][r]
            new_val = random.choice([-1, 0, 1])
            W[idx][r] = new_val
            new_error = compute_error(U, V, W)
            if new_error < best_error:
                best_error = new_error
                if best_error == 0:
                    return {"U": U, "V": V, "W": W}
            else:
                W[idx][r] = old_val
                
    return None
