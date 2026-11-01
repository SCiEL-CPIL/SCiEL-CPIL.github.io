import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Keep margin for safety
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Search for lower ranks
    # The tensor is T[u, v, w] = sum_t U[u,t] V[v,t] W[w,t]
    # Non-zero terms are at u=i*m+j, v=j*p+k, w=k*n+i
    # For a fixed w=k*n+i, the non-zero terms are for j=0..m-1
    # u_j = i*m+j, v_j = j*p+k
    # Equation: sum_t U[u_j, t] V[v_j, t] W[w, t] = 1 for all j
    
    # We iterate r from some lower bound to best_rank - 1
    # Lower bound: max(n*m, m*p, p*n) is not quite right.
    # Rank must be at least the number of non-zero entries in any slice?
    # For fixed w, we have m equations. So r >= 1 is needed locally, but globally coupled.
    # For 2x3x7, let's try down to 20.
    
    np.random.seed(42)
    
    for r in range(20, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        # Using integer or half-integer entries might help, but float is fine for search
        # We'll use random floats in [-1, 1] or small integers
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            # Generate random U and V
            # Sparse or dense? Dense random often fails.
            # Let's try random integers in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n * m, r))
            V = np.random.choice([-1, 0, 1], size=(m * p, r))
            
            # Check consistency for each w
            valid = True
            W = np.zeros((p * n, r))
            
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    # Build A matrix of size m x r
                    A = np.zeros((m, r))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        A[j, :] = U[u_idx, :] * V[v_idx, :]
                    
                    # Check if 1 is in the column space of A
                    # Solve A W_w = 1
                    try:
                        W_w, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        # Check residual
                        if np.linalg.norm(A @ W_w - np.ones(m)) > 1e-9:
                            valid = False
                            break
                        W[w_idx, :] = W_w
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full tensor just in case
                # T[u, v, w] should be 1 if valid, 0 otherwise
                # We constructed W to satisfy the non-zero terms.
                # We must ensure the zero terms are also satisfied.
                # T[u, v, w] = sum_t U[u,t] V[v,t] W[w,t]
                # For non-zero terms, it is 1 by construction.
                # For zero terms, it must be 0.
                # Zero terms are those where indices don't match.
                # e.g. u=i*m+j, v=j'*p+k, w=k'*n+i' with j!=j' or k!=k' or i!=i'
                
                # Check a few zero terms or all?
                # All zero terms: (n*m)*(m*p)*(p*n) - n*m*p = 1764 - 42 = 1722
                # We can check all of them.
                
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            # This is a non-zero term, should be 1
                            val = np.dot(U[u, :] * V[v, :], W[w, :])
                            if abs(val - 1.0) > 1e-9:
                                error += 1
                                valid = False
                            
                # Check zero terms
                if valid:
                    # Iterate over all u, v, w
                    for u in range(n * m):
                        # Extract i, j from u
                        i = u // m
                        j = u % m
                        for v in range(m * p):
                            # Extract j', k from v
                            jp = v // p
                            k = v % p
                            if jp != j:
                                # This is a zero term in the tensor
                                # For all w, T[u,v,w] should be 0
                                # But W[w] depends on w.
                                # We need sum_t U[u,t] V[v,t] W[w,t] = 0 for ALL w
                                # This is a strong condition.
                                # Let's check for all w
                                for w in range(p * n):
                                    val = np.dot(U[u, :] * V[v, :], W[w, :])
                                    if abs(val) > 1e-9:
                                        error += 1
                                        valid = False
                                        break
                                if not valid:
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = r
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break # Found minimal rank in this search? No, we search r upwards.
                    # Since we start from low r, the first found is the lowest.
                    
        if best_rank < rank_school:
            break

    return best_result
