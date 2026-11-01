import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook solution (rank 56)
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Check sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] == delta[j, j']
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 130.0

    # Strategy: Block decomposition
    # A (2x4) can be split into A1 (2x2) and A2 (2x2)
    # B (4x7) can be split into B1 (2x7) and B2 (2x7)
    # C = A1 B1 + A2 B2
    # This reduces the problem to two (2x2)*(2x7) multiplications.
    # Rank of (2x2)*(2x7) is at most 28 (schoolbook), but can be lower.
    # If we find a rank-r decomposition for (2x2)*(2x7), total rank is 2r.
    # Schoolbook for (2x2)*(2x7) is 28, so total 56.
    # If we can find rank 25 for (2x2)*(2x7), total is 50.
    
    # Let's try to find a rank-r decomposition for (2x2)*(2x7).
    # The tensor is T_{i, j, j', k} = delta_{j, j'} for i in 0..1, j, j' in 0..1, k in 0..6.
    # We want sum_c U_c V_c W_c = T.
    # U: (2*2, r) = (4, r)
    # V: (2*7, r) = (14, r)
    # W: (7*2, r) = (14, r)
    
    # We can use the fact that the 2x2 part is small.
    # For each column c, U_c is a vector in R^4, V_c in R^14, W_c in R^14.
    # The condition is:
    # For all i, k: sum_j, j' T_{i, j, j', k} = sum_j A_{i, j} B_{j', k} ... wait.
    # The condition is sum_c U[i*2+j, c] V[j'*7+k, c] W[k*2+i, c] = delta_{j, j'}.
    
    # Let's try to construct a solution for (2x2)*(2x7) with rank r.
    # We can view this as multiplying a 2x2 matrix by a 2x7 matrix.
    # This is equivalent to multiplying a 2x2 matrix by 7 different 2x1 vectors? No.
    # It's just 7 separate 2x2 * 1x2 multiplications? No, B is 2x7.
    # C = A B. A is 2x2, B is 2x7.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    
    # We can use Strassen for the 2x2 part if we could decouple the k dimension, but we can't.
    # However, we can use the fact that the 2x2 multiplication has rank 7 (Strassen).
    # If we could use Strassen for each k, we'd need 7*7=49? No, that's not how it works.
    # The k dimension is in B and C.
    
    # Let's try a random search for (2x2)*(2x7) with rank r.
    # If we find rank r, we can combine two such decompositions to get rank 2r for the original problem.
    # Note: The two decompositions must be independent (disjoint columns).
    
    # Let's search for rank r from 28 down to 20 for the sub-problem.
    # If we find r=25, total rank is 50.
    
    def find_sub_decomposition(rank_sub, max_iters=1000):
        # U: (4, rank_sub), V: (14, rank_sub), W: (14, rank_sub)
        # Condition: sum_c U[i*2+j, c] V[j'*7+k, c] W[k*2+i, c] = delta_{j, j'}
        
        # We can try to solve this using linear algebra.
        # Fix U and W, solve for V? Or fix U and V, solve for W?
        # The equations are linear in V for fixed U, W.
        # Let's fix U and W randomly and solve for V.
        # Actually, it's a system of linear equations for the entries of U, V, W? No, it's bilinear/trilinear.
        
        # Let's try random search with small integers.
        for _ in range(max_iters):
            U_sub = np.random.randint(-1, 2, size=(4, rank_sub))
            V_sub = np.random.randint(-1, 2, size=(14, rank_sub))
            W_sub = np.random.randint(-1, 2, size=(14, rank_sub))
            
            # Verify
            valid = True
            for i in range(2):
                for k in range(7):
                    for j in range(2):
                        for jp in range(2):
                            s = 0.0
                            for c in range(rank_sub):
                                s += U_sub[i*2+j, c] * V_sub[jp*7+k, c] * W_sub[k*2+i, c]
                            if abs(s - (1.0 if j == jp else 0.0)) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if valid:
                return U_sub, V_sub, W_sub
        return None

    # Try to find a sub-decomposition
    for rank_sub in range(28, 19, -1):
        if time.time() > deadline:
            break
            
        result = find_sub_decomposition(rank_sub, max_iters=500)
        if result is not None:
            U_sub, V_sub, W_sub = result
            # Construct the full solution by concatenating two copies
            U_full = np.hstack([U_sub, U_sub])
            V_full = np.hstack([V_sub, V_sub])
            W_full = np.hstack([W_sub, W_sub])
            
            # Wait, this is not correct.
            # The full problem is A (2x4) * B (4x7).
            # We split A = [A1, A2] and B = [B1; B2].
            # C = A1 B1 + A2 B2.
            # The first term uses indices j=0,1 for A and j=0,1 for B.
            # The second term uses indices j=2,3 for A and j=2,3 for B.
            
            # So we need to map the sub-problem indices to the full indices.
            # Sub-problem: A_sub (2x2), B_sub (2x7).
            # U_sub[i*2+j, c] corresponds to U_full[i*4 + j, c] for the first block (j=0,1).
            # V_sub[jp*7+k, c] corresponds to V_full[jp*7+k, c] for the first block (jp=0,1).
            # W_sub[k*2+i, c] corresponds to W_full[k*2+i, c].
            
            # For the second block:
            # U_sub[i*2+j, c] corresponds to U_full[i*4 + (2+j), c] for the second block (j=0,1 -> 2,3).
            # V_sub[jp*7+k, c] corresponds to V_full[(2+jp)*7+k, c] for the second block (jp=0,1 -> 2,3).
            # W_sub[k*2+i, c] corresponds to W_full[k*2+i, c] (same i, k).
            
            # So we can construct U_full, V_full, W_full with rank 2 * rank_sub.
            U_full = np.zeros((n*m, 2 * rank_sub))
            V_full = np.zeros((m*p, 2 * rank_sub))
            W_full = np.zeros((p*n, 2 * rank_sub))
            
            # First block (j=0,1)
            for i in range(2):
                for j in range(2):
                    for c in range(rank_sub):
                        U_full[i*4 + j, c] = U_sub[i*2 + j, c]
                for jp in range(2):
                    for k in range(7):
                        for c in range(rank_sub):
                            V_full[jp*7 + k, c] = V_sub[jp*7 + k, c]
                for k in range(7):
                    for c in range(rank_sub):
                        W_full[k*2 + i, c] = W_sub[k*2 + i, c]
                        
            # Second block (j=2,3)
            for i in range(2):
                for j in range(2):
                    for c in range(rank_sub):
                        U_full[i*4 + (2+j), rank_sub + c] = U_sub[i*2 + j, c]
                for jp in range(2):
                    for k in range(7):
                        for c in range(rank_sub):
                            V_full[(2+jp)*7 + k, rank_sub + c] = V_sub[jp*7 + k, c]
                for k in range(7):
                    for c in range(rank_sub):
                        W_full[k*2 + i, rank_sub + c] = W_sub[k*2 + i, c]
            
            # Verify the full solution
            if verify(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2 * rank_sub
                break # Found a solution, try to improve if time permits? Or just break.
                
    # If no sub-decomposition was found, we stick with schoolbook.
    # But we can also try a direct random search on the full problem for ranks < 56.
    # This is less likely to succeed but might find a non-block solution.
    
    if best_rank == 56:
        # Try random search on full problem for rank 55 down to 50
        for rank in range(55, 49, -1):
            if time.time() > deadline:
                break
            for _ in range(50):
                if time.time() > deadline:
                    break
                U = np.random.randint(-1, 2, size=(n*m, rank))
                V = np.random.randint(-1, 2, size=(m*p, rank))
                W = np.random.randint(-1, 2, size=(p*n, rank))
                if verify(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
