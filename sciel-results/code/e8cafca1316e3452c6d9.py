import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions of the tensor factors
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Build the target tensor T of shape (dim_u, dim_v, dim_w)
    # T[a, b, c] = 1 if indices are consistent, 0 otherwise
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # We want to decompose T into sum of rank-1 tensors u_k v_k w_k
    # Strategy: Greedy peeling with alternating maximization for each rank-1 term
    
    deadline = time.time() + 140.0  # Safety margin
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Initial guess for rank: start high and try to reduce? 
    # Or just peel until zero.
    # Peeling naturally finds a rank. We stop when R is zero.
    
    R = T.copy().astype(np.float64)
    
    U_list = []
    V_list = []
    W_list = []
    
    max_iterations = 10000 # Hard cap on peeling steps to prevent infinite loop if stuck
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        if not np.any(R):
            break
        
        # Find the best rank-1 approximation of R
        # Initialize u, v, w randomly or with zeros
        # Use a heuristic: start with random {-1, 0, 1} vectors
        u = np.random.choice([-1, 0, 1], size=dim_u)
        v = np.random.choice([-1, 0, 1], size=dim_v)
        w = np.random.choice([-1, 0, 1], size=dim_w)
        
        # Alternating maximization to maximize <R, u v w>
        # We want to maximize S = sum R[a,b,c] u[a] v[b] w[c]
        
        prev_score = -1e18
        converged = False
        
        for _iter in range(50): # Max iterations for one rank-1 term
            if time.time() > deadline:
                break
            
            # Update u: fix v, w. Maximize sum_a u[a] * (sum_b sum_c R[a,b,c] v[b] w[c])
            # Let M_u[a] = sum_{b,c} R[a,b,c] v[b] w[c]
            # To maximize sum u[a] M_u[a] with u in {-1,0,1}, set u[a] = sign(M_u[a])
            
            # Compute M_u efficiently
            # R is (dim_u, dim_v, dim_w)
            # v is (dim_v), w is (dim_w)
            # M_u = np.einsum('abc,b,c->a', R, v, w)
            M_u = np.einsum('abc,b,c->a', R, v, w)
            new_u = np.sign(M_u).astype(np.float64)
            
            # Update v: fix u, w
            # M_v[b] = sum_{a,c} R[a,b,c] u[a] w[c]
            M_v = np.einsum('abc,a,c->b', R, u, w)
            new_v = np.sign(M_v).astype(np.float64)
            
            # Update w: fix u, v
            # M_w[c] = sum_{a,b} R[a,b,c] u[a] v[b]
            M_w = np.einsum('abc,a,b->c', R, u, v)
            new_w = np.sign(M_w).astype(np.float64)
            
            # Check convergence
            if np.array_equal(new_u, u) and np.array_equal(new_v, v) and np.array_equal(new_w, w):
                converged = True
                break
                
            u, v, w = new_u, new_v, new_w
            
            # Calculate score to check improvement
            score = np.einsum('abc,a,b,c', R, u, v, w)
            if score == prev_score:
                # If score doesn't change, we might be stuck in a cycle or plateau
                # But sign updates usually monotonically increase or stay same if we are careful?
                # Actually, alternating maximization on discrete sets doesn't guarantee monotonic increase globally, 
                # but usually helps.
                pass
            prev_score = score
            
        if time.time() > deadline:
            break
            
        # Calculate the final score of this rank-1 term
        score = np.einsum('abc,a,b,c', R, u, v, w)
        
        # If the score is 0, we couldn't find a useful term. 
        # This might mean we are stuck or the remaining R is "orthogonal" to simple sign vectors.
        # In that case, we might need to restart or accept a non-optimal term.
        # For now, if score is 0, we break and return what we have (which is invalid if R != 0).
        # To ensure validity, we should only subtract if we make progress.
        if score == 0:
            # Try a few more random restarts for this rank-1 term
            found = False
            for _restart in range(10):
                if time.time() > deadline:
                    break
                u = np.random.choice([-1, 0, 1], size=dim_u)
                v = np.random.choice([-1, 0, 1], size=dim_v)
                w = np.random.choice([-1, 0, 1], size=dim_w)
                
                for _iter in range(20):
                    M_u = np.einsum('abc,b,c->a', R, v, w)
                    new_u = np.sign(M_u).astype(np.float64)
                    M_v = np.einsum('abc,a,c->b', R, u, w)
                    new_v = np.sign(M_v).astype(np.float64)
                    M_w = np.einsum('abc,a,b->c', R, u, v)
                    new_w = np.sign(M_w).astype(np.float64)
                    u, v, w = new_u, new_v, new_w
                    
                score = np.einsum('abc,a,b,c', R, u, v, w)
                if score > 0:
                    found = True
                    break
            
            if not found:
                # Stuck. Return current best if valid? 
                # If R is not zero, the decomposition is incomplete.
                # We must return a valid decomposition.
                # If we can't peel further, we fail to find a low rank.
                # We should fall back to schoolbook if we don't complete.
                # But we might have a partial decomposition.
                # The task requires EXACT decomposition.
                # So if we can't reach R=0, we must use a method that guarantees it.
                # Greedy peeling does NOT guarantee it.
                # We should probably switch to a guaranteed method if we get stuck.
                # For now, let's just break and return the best valid full decomposition found?
                # We haven't stored a valid full one unless R=0.
                # Let's assume we can always peel if we allow larger integers?
                # The problem allows integer/half-integer.
                # My vectors are {-1, 0, 1}.
                # If we get stuck, we might need to relax the search.
                
                # Fallback: Use schoolbook for the remaining part?
                # No, mixing is hard.
                # Let's just try to run longer or more restarts.
                # If we truly get stuck, we return the schoolbook solution to be safe (score 1.0).
                # But we want to beat 1.0.
                
                # Let's try one more time with a different random seed logic
                # If still stuck, break.
                pass
                
        if score > 0:
            # Subtract this term from R
            R -= np.einsum('a,b,c->abc', u, v, w)
            
            # Store the vectors
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
            # Check if R is zero
            if np.allclose(R, 0):
                break
                
        else:
            # If we can't find a term with score > 0, we are stuck.
            # We cannot proceed with this greedy approach to get an exact solution.
            # We should abort and return a known valid solution (schoolbook) to ensure score >= 1.0.
            # However, if we found some terms, R is not zero, so the solution is invalid.
            # We must either complete the decomposition or fail.
            # Failing means returning an invalid result (score 0).
            # Returning schoolbook means score 1.0.
            # So, if stuck, we should return schoolbook.
            
            # Let's try to detect if we are close to completion.
            # If R has very few non-zeros, maybe we can finish with schoolbook on the remainder?
            # No, the tensor structure is specific.
            
            # Let's just break and return schoolbook.
            break

    # If we completed the decomposition (R is zero), we have a valid result
    if np.allclose(R, 0) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T  # Shape (dim_u, rank)
        V = np.array(V_list).T  # Shape (dim_v, rank)
        W = np.array(W_list).T  # Shape (dim_w, rank)
        
        # Convert to list of lists as required? 
        # The interface says "factor matrices of shapes...". 
        # The seed program returns lists of lists.
        # NumPy arrays are usually accepted if they are iterable, but let's stick to the seed format to be safe.
        # Actually, the seed returns lists.
        
        # Check if rank is better than schoolbook
        schoolbook_rank = n * m * p
        if rank < schoolbook_rank:
             # Return the improved decomposition
             return {
                 "U": U.tolist(),
                 "V": V.tolist(),
                 "W": W.tolist()
             }
        else:
             # No improvement, return schoolbook
             pass

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
