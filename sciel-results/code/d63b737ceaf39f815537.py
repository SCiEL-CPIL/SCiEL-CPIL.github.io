"""Combined program: structured search for rank-minimal decomposition of M_{3,4,7}."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 138.0  # Safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
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

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    fallback = schoolbook()

    # Core idea: For each rank r, we want to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    #
    # Strategy: Fix U and V randomly with entries in {-1, 0, 1}. Then for each (i,k),
    # the equation becomes: sum_r W[k*n+i, r] * M[i,k,r] = 1, where M[i,k,r] = sum_j U[i*m+j,r]*V[j*p+k,r].
    # If we can find W entries in {0, 0.5, 1} (or negatives) that satisfy this, we're done.
    #
    # Better strategy: Use the fact that the tensor is the all-ones tensor. 
    # For the all-ones tensor of shape (n, m, p), a rank-r decomposition exists iff
    # there are vectors u_i, v_j, w_k in R^r such that sum_t u_i[t] v_j[t] w_k[t] = 1 for all i,j,k.
    #
    # Key insight: If we can find r vectors a_1, ..., a_r in R^n, b_1, ..., b_r in R^m, c_1, ..., c_r in R^p
    # such that for all i, j, k: sum_t a_i[t] b_j[t] c_k[t] = 1, then we have a rank-r decomposition.
    #
    # This is equivalent to: the all-ones tensor = sum_t a_t ⊗ b_t ⊗ c_t.
    #
    # For the all-ones tensor, one simple construction: if we can find r such that there exist
    # vectors with the property that their outer products sum to all-ones.
    #
    # A known result: The rank of the all-ones tensor of shape (a,b,c) is at most min(a,b,c) * max(a,b,c)
    # but we can do better.
    #
    # Let's try a different approach: use random search with structured initialization.
    # Initialize U, V, W with random ±1 entries, then iteratively refine.
    #
    # Actually, let's use a smarter approach: for small ranks, try to find exact solutions
    # by solving a system of equations. For rank r, we have n*m*p equations and (n*m + m*p + p*n)*r unknowns.
    #
    # For r = 21 (as suggested by the strategy), let's try random search with verification.

    def try_rank_structured(rank, max_attempts):
        """Try to find a rank-r decomposition using random search with structured hints."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random U, V with entries in {-1, 0, 1}
            U = np.array([[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(n * m)])
            V = np.array([[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(m * p)])
            
            # Compute M[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((n, p, rank))
            for i in range(n):
                for k in range(p):
                    for r_idx in range(rank):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j, r_idx] * V[j * p + k, r_idx]
                        M[i, k, r_idx] = s
            
            # Now for each (i,k), we need: sum_r W[k*n+i, r] * M[i,k,r] = 1
            # Try to find W with entries in {0, 0.5, 1, -0.5, -1}
            W = np.zeros((p, n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    row = M[i, k, :]
                    # Try to find a single r where row[r] = 1, or row[r] = 2 (use W=0.5), etc.
                    found = False
                    for r_idx in range(rank):
                        mv = row[r_idx]
                        if mv == 1:
                            W[k, i, r_idx] = 1
                            found = True
                            break
                        elif mv == -1:
                            W[k, i, r_idx] = -1
                            found = True
                            break
                        elif mv == 2:
                            W[k, i, r_idx] = 0.5
                            found = True
                            break
                        elif mv == -2:
                            W[k, i, r_idx] = -0.5
                            found = True
                            break
                    if not found:
                        # Try linear combination: find two r's where we can combine
                        # e.g., M[i,k,r1] = 1, M[i,k,r2] = 1 -> use W=0.5 for both
                        found = False
                        for r1 in range(rank):
                            for r2 in range(r1 + 1, rank):
                                if row[r1] == 1 and row[r2] == 1:
                                    W[k, i, r1] = 0.5
                                    W[k, i, r2] = 0.5
                                    found = True
                                    break
                                elif row[r1] == -1 and row[r2] == -1:
                                    W[k, i, r1] = -0.5
                                    W[k, i, r2] = -0.5
                                    found = True
                                    break
                                elif row[r1] == 2 and row[r2] == 0:
                                    W[k, i, r1] = 0.5
                                    found = True
                                    break
                                elif row[r1] == -2 and row[r2] == 0:
                                    W[k, i, r1] = -0.5
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if verify(U_list, V_list, W_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Also try a denser random search with real-valued entries, then round
    def try_rank_dense(rank, max_attempts):
        """Try random real-valued U, V, solve for W, then check if W has nice entries."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            
            # For each (i,k), we need: sum_r W[k*n+i, r] * M[i,k,r] = 1
            # where M[i,k,r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((n, p, rank))
            for i in range(n):
                for k in range(p):
                    for r_idx in range(rank):
                        s = 0
                        for j in range(m):
                            s += U[i * m + j, r_idx] * V[j * p + k, r_idx]
                        M[i, k, r_idx] = s
            
            # For each (i,k), we have a linear equation in W[k*n+i, :]
            # If rank(M[i,k,:]) >= 1, we can find a solution.
            # But we want W to have simple entries.
            
            # Try to find W such that the decomposition is exact
            W = np.zeros((p, n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    row = M[i, k, :]
                    # Find minimum norm solution to row @ W[k,i,:] = 1
                    if np.linalg.norm(row) < 1e-10:
                        valid = False
                        break
                    # W[k,i,:] = row / (row @ row)
                    W[k, i, :] = row / np.dot(row, row)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if verify(U_list, V_list, rank):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Search for lower-rank decompositions
    lower = max(n, m, p)  # = 7
    upper = n * m * p  # = 84

    best_result = None
    for rank in range(lower, upper):
        if time.time() > deadline:
            break
        
        # More attempts for smaller ranks
        if rank < 25:
            attempts = 200
        elif rank < 40:
            attempts = 100
        else:
            attempts = 50
        
        result = try_rank_structured(rank, attempts)
        if result is not None:
            return result
        
        # Also try dense search for some ranks
        if rank % 5 == 0:
            result = try_rank_dense(rank, 20)
            if result is not None:
                return result

    return fallback
