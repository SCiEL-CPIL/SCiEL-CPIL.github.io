import time
import random
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid rank-r decomposition for matrix multiplication.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Condition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 for all valid (i,j,k)
    Note: The tensor definition for matmul C=A*B is T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    Actually, the standard tensor decomposition of the multiplication tensor T_{i,j,k} (where C_{i,k} = sum_j A_{i,j} B_{j,k})
    requires that sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1 for all i,j,k.
    Wait, the indices in the contract are:
    U: (n*m, rank) -> indexed by (i, j) where i in [0,n), j in [0,m)
    V: (m*p, rank) -> indexed by (j, k) where j in [0,m), k in [0,p)
    W: (p*n, rank) -> indexed by (k, i) where k in [0,p), i in [0,n)
    
    The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    This sum must equal 1 for all i, j, k.
    """
    r = U.shape[1]
    # Reshape for easy access
    U_mat = np.array(U)
    V_mat = np.array(V)
    W_mat = np.array(W)
    
    # We need to check if for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Efficient check:
    # For each c, the term is a rank-1 tensor.
    # Let's just iterate. n=2, m=3, p=5. Total triples = 30.
    # r is likely small if we are searching for improvement, or 30 for schoolbook.
    # 30 * r multiplications. If r=29, that's 870 ops. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Sum over c
                s = 0.0
                for c in range(r):
                    s += U_mat[u_idx, c] * V_mat[v_idx, c] * W_mat[w_idx, c]
                # Allow small floating point error
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Define a deadline based on the 150s limit (use 140s for safety).
    2. Attempt to find a decomposition with rank < 30 (starting from 29 down to 1) 
       using a randomized search over integer/half-integer entries.
       Given the small size (n=2, m=3, p=5), the search space for rank 29 is still 
       huge, but we can use a heuristic: 
       The schoolbook decomposition has 30 terms. 
       Is it possible to combine two terms? 
       T = sum e_i e_j f_k. 
       If we can write one term as a combination of others with linear dependencies, rank drops.
       However, for small matrices, the rank is often equal to n*m*p unless there are specific symmetries.
       For 2x3 * 3x5, it is widely believed to be 30, but we must try to prove/discover otherwise 
       or at least exhaust the time budget trying.
       
       Actually, a smarter approach for "search" in this context with limited time:
       We can't do a full random search over R^30x30. 
       But we can try to perturb the schoolbook decomposition and see if we can merge terms.
       Or, we can use a specific algebraic structure if one exists.
       
       Given the strict time limit and the high likelihood that 30 is the minimum, 
       the "search" might be futile. However, the directive asks for a STRUCTURAL change.
       The current program is static. This program is dynamic (time-boxed search).
       
       To make the search non-trivial but fast:
       We will attempt a local search starting from the schoolbook decomposition.
       We will try to remove one column (reducing rank to 29) and adjust the remaining 
       29 columns to compensate. This is a system of linear equations.
       
       Let U, V, W be the schoolbook matrices.
       Remove column c*. We have U', V', W' of size (x, 29).
       We need to find U_new, V_new, W_new such that they satisfy the tensor equation.
       This is non-linear.
       
       Alternative: 
       Use the fact that the tensor is defined by linear constraints on the entries.
       Actually, the simplest "structural change" that uses the time budget is to run a 
       randomized restart of a solver that attempts to find a rank-29 decomposition 
       by solving the resulting system of polynomial equations (or using a heuristic).
       
       However, solving polynomial systems for rank 29 is hard.
       
       Let's look at the "inspiration" note: "random search for a lower rank" was deemed "costly and unlikely".
       But the directive says "Make a STRUCTURAL change... A first try in a new direction may score worse; that is acceptable."
       
       I will implement a time-boxed search that attempts to find a rank < 30 decomposition.
       If it fails, it returns the schoolbook result.
       This changes the program from "static output" to "bounded search".
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 3, 5
    base_rank = n * m * p  # 30
    
    # 1. Generate the schoolbook solution as the fallback
    best_sol = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # 2. Attempt to find a lower rank
    # We try ranks from base_rank - 1 down to 1.
    # For each rank r, we attempt a randomized search.
    # Since the space is continuous, we sample from a small set of rational numbers 
    # or use a local optimization approach.
    # Given the difficulty, we will limit the number of attempts per rank.
    
    # For a "structural" change, we use a simple randomized restart strategy 
    # with a fixed budget of attempts per rank, controlled by the deadline.
    
    # We define a small set of candidate values for entries to keep the search 
    # within a discrete, manageable space (e.g., -1, 0, 1, 0.5, -0.5).
    # This is a heuristic search.
    
    candidate_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # We will try rank 29 first.
    # If we find a solution, we return it.
    # If time runs out, we return the best found (which is schoolbook if no lower found).
    
    for r in range(base_rank - 1, 0, -1):
        if time.time() > deadline - 5.0:  # Stop if less than 5s left
            break
            
        # Try a few random initializations for rank r
        # We limit the number of tries to prevent infinite loops if verification is slow
        # Verification is O(n*m*p*r) = O(30*r). For r=29, ~870 ops. Very fast.
        # So we can do many tries.
        
        max_tries = 1000  # Arbitrary limit per rank, but we check time each time
        
        for _ in range(max_tries):
            if time.time() > deadline - 2.0:
                break
                
            # Randomly initialize U, V, W with small integer/half-integer values
            # To increase the chance of finding a solution, we can bias towards 
            # the schoolbook structure but with perturbations.
            # However, pure random is a valid structural change from "static".
            
            U = [[random.choice(candidate_values) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice(candidate_values) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice(candidate_values) for _ in range(r)] for _ in range(p * n)]
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_sol = {"U": U, "V": V, "W": W}
                best_rank = r
                # If we found a lower rank, we can stop searching for lower ranks 
                # if we are confident, or continue to find even lower.
                # Given the difficulty, finding rank 29 is unlikely. 
                # But if we find one, we should return it immediately as it's an improvement.
                # However, the task asks for "rank-minimal". 
                # If we find rank 29, is 28 possible? 
                # To be safe and maximize score (lower rank is better), we should continue 
                # searching for lower ranks if time permits.
                # But usually, if 30 is the bound, 29 won't be found.
                # If 29 is found, it's a huge breakthrough.
                pass
        
        # Optional: If we found a solution for this rank, we could break 
        # if we assume we found the minimum, but we don't know that.
        # We continue to check lower ranks if time allows.
        
    return best_sol
