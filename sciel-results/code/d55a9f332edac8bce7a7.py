import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    # Schoolbook rank is 64
    schoolbook_rank = n * m * p
    
    # Initialize with schoolbook
    def get_schoolbook():
        U = np.zeros((U_dim, schoolbook_rank))
        V = np.zeros((V_dim, schoolbook_rank))
        W = np.zeros((W_dim, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_init, V_init, W_init = get_schoolbook()
    
    # We want to find a lower rank.
    # Strategy: Start with a rank slightly below schoolbook and try to find a valid decomposition.
    # We will use a randomized search with a specific structure:
    # 1. Randomly initialize U, V, W with small integers.
    # 2. Compute the error tensor E = T - sum_t u_t v_t w_t.
    # 3. If error is zero, we found a solution.
    # 4. To guide the search, we will use a "repair" step: 
    #    For a fixed rank r, we can fix U and V, and solve for W.
    #    But we want to ensure W is integer.
    #    
    # New Strategy: Use the fact that the tensor is all 1s.
    # We can try ranks from 20 up to 64.
    # For each rank, we perform a random search.
    # To make it faster, we will use a "structured" random search:
    # We initialize U and V with random integers from {-1, 0, 1}.
    # Then we solve for W using least squares.
    # If W is close to integer, we round it and verify.
    
    deadline = time.time() + 140.0
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_init, V_init, W_init
    
    # Try to find a lower rank
    for rank in range(20, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Number of attempts per rank
        # We allocate time based on rank
        attempts = 5000
        for attempt in range(attempts):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            # Use small integers to increase chance of integer W
            U = np.random.randint(-2, 3, size=(U_dim, rank))
            V = np.random.randint(-2, 3, size=(V_dim, rank))
            
            # Solve for W
            W = np.zeros((W_dim, rank))
            valid = True
            for i in range(n):
                # For each i, we have a system of m*p equations in r*p unknowns (W[k,i,t])
                # Actually, W is indexed by (k*n+i, t).
                # The equation is: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
                # This is linear in W.
                # Unknowns: W[k*n+i, t] for k=0..p-1, t=0..rank-1
                # Let's flatten the unknowns for fixed i: 
                # vec(W_i) of length p*rank
                # Equation for (j,k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
                # This is a bit tricky because W index depends on k and i, but t is shared.
                
                A_mat = np.zeros((m*p, p*rank))
                rhs = np.ones(m*p)
                eq_idx = 0
                for j in range(m):
                    for k in range(p):
                        for t in range(rank):
                            # Coefficient for W[k*n+i, t] is U[i*m+j, t] * V[j*p+k, t]
                            # In the flattened vector, the index for W[k*n+i, t] is t*p + k
                            A_mat[eq_idx, t*p + k] = U[i*m+j, t] * V[j*p+k, t]
                        eq_idx += 1
                
                try:
                    W_i, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                except:
                    valid = False
                    break
                
                # Check if W_i is close to integer
                # W_i is of shape (p*rank,). Reshape to (rank, p) -> W[k*n+i, t] is W_i[t*p+k]
                # Actually, W_i[t*p+k] corresponds to W[k*n+i, t].
                # Let's check if all entries are close to integer.
                if np.all(np.abs(W_i - np.round(W_i)) < 1e-4):
                    W_i = np.round(W_i)
                    # Fill W
                    for t in range(rank):
                        for k in range(p):
                            W[k*n+i, t] = W_i[t*p + k]
                else:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify
            # Compute sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
            # Check if it equals 1 for all i,j,k
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                        if abs(val - 1.0) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                found = True
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if found:
            break
    
    # If no lower rank found, return schoolbook
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
