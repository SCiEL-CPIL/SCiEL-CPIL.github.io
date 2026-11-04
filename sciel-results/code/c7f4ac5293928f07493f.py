import numpy as np
import random
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Build the multiplication tensor in standard form
    # T[i,j,k] = 1 if we want A[i,j]*B[j,k] -> C[i,k]
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = n, m, p
        R = len(U[0]) if isinstance(U, list) else U.shape[1]
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over all terms r
                    total = 0.0
                    for r in range(R):
                        total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def solve_lp_support(U_support, V_support, W_support, n, m, p):
        """
        Given binary support matrices, solve for the actual values.
        The equation is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        This is bilinear in the unknowns. We can fix one set of variables and solve linearly for another.
        Alternatively, we can use the fact that for each (i,j,k), the constraint involves products of three variables.
        
        A simpler approach: assume the supports define which (i,j,k) triples are covered by which rank term.
        For each rank term r, it covers a specific (i,j,k) triple if U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1.
        
        Actually, let's think differently. The standard approach is:
        - Each rank term r corresponds to a choice of indices (i_r, j_r, k_r)
        - U[i*m+j, r] is non-zero only if i=i_r and j=j_r (or more generally, the support defines which entries are non-zero)
        - Similarly for V and W
        
        For the simplest case where each term has exactly one non-zero in each of U, V, W:
        Term r: U[i_r*m+j_r, r] = a_r, V[j_r*p+k_r, r] = b_r, W[k_r*n+i_r, r] = c_r
        Constraint: sum_r a_r*b_r*c_r * [i=i_r, j=j_r, k=k_r] = 1 for all (i,j,k)
        
        This means each (i,j,k) must be covered by exactly one term with a_r*b_r*c_r = 1, or multiple terms that sum to 1.
        
        Let's use a different approach: fix the supports, then the problem becomes:
        For each (i,j,k): sum_{r: U[i*m+j,r]!=0 and V[j*p+k,r]!=0 and W[k*n+i,r]!=0} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        This is still nonlinear. Let's try a simpler verification:
        If we set all non-zero entries to 1, check if the count of covering terms equals 1 for each (i,j,k).
        """
        n, m, p = n, m, p
        U_arr = np.array(U_support, dtype=float)
        V_arr = np.array(V_support, dtype=float)
        W_arr = np.array(W_support, dtype=float)
        R = U_arr.shape[1]
        
        # Check if each (i,j,k) is covered by exactly one term with product 1
        # If we set all non-zeros to 1, then the sum is the count of terms covering (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    count = 0
                    for r in range(R):
                        if U_arr[i*m+j, r] != 0 and V_arr[j*p+k, r] != 0 and W_arr[k*n+i, r] != 0:
                            count += 1
                    if count != 1:
                        return None
        
        # If exactly one term covers each (i,j,k), set all non-zeros to 1
        U_sol = U_support.copy()
        V_sol = V_support.copy()
        W_sol = W_support.copy()
        for r in range(R):
            for idx in range(U_sol.shape[0]):
                if U_sol[idx, r] != 0:
                    U_sol[idx, r] = 1
            for idx in range(V_sol.shape[0]):
                if V_sol[idx, r] != 0:
                    V_sol[idx, r] = 1
            for idx in range(W_sol.shape[0]):
                if W_sol[idx, r] != 0:
                    W_sol[idx, r] = 1
        
        # Verify
        if verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
            return (U_sol, V_sol, W_sol)
        return None

    start_time = time.time()
    max_time = 10  # seconds
    
    # Try to find a better rank using LP-like support search
    # Key insight: for matrix multiplication, the tensor has a special structure.
    # The known best bound for 3x3 * 3x7 is at most 63 (schoolbook).
    # Let's try to find supports where each (i,j,k) is covered by exactly one term.
    
    # For a valid decomposition with unit values, we need a "perfect matching" in the 3D sense:
    # Each rank term r picks one (i_r, j_r, k_r), and every (i,j,k) is picked exactly once.
    # This is equivalent to a 3D permutation matrix, which only works when n*m*p terms are used.
    
    # So we need to allow multiple terms to cover the same (i,j,k) with fractional values.
    # Let's try a different approach: use the known Strassen-like patterns.
    
    # For 3x3*3x7, let's try ranks from 45 to 62
    best_decomp = fallback
    best_rank = rank_schoolbook
    
    for R in range(45, min(63, rank_schoolbook + 1)):
        if time.time() - start_time > max_time:
            break
        
        # Try random support patterns with the constraint that each (i,j,k) is covered
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            U_support = np.zeros((n * m, R), dtype=int)
            V_support = np.zeros((m * p, R), dtype=int)
            W_support = np.zeros((p * n, R), dtype=int)
            
            # Assign each (i,j,k) to at least one term
            assigned = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        assigned[(i,j,k)] = []
            
            # Randomly assign each term to a random (i,j,k)
            for r in range(R):
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                U_support[i*m+j, r] = 1
                V_support[j*p+k, r] = 1
                W_support[k*n+i, r] = 1
                assigned[(i,j,k)].append(r)
            
            # Check coverage
            covered = all(len(terms) > 0 for terms in assigned.values())
            if not covered:
                continue
            
            # Try to solve: for each (i,j,k), the sum of products over covering terms must equal 1
            # If each (i,j,k) is covered by exactly one term, set that term's values to 1
            # If covered by multiple terms, we need to solve a system
            
            # Simple case: if each (i,j,k) is covered by exactly one term
            all_single = all(len(terms) == 1 for terms in assigned.values())
            if all_single:
                U_sol = U_support.astype(float)
                V_sol = V_support.astype(float)
                W_sol = W_support.astype(float)
                if verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                    best_rank = R
                    best_decomp = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                    break
            
            # For more complex cases, try a simple iterative method
            # Initialize all non-zeros to 1/sqrt(coverage)
            U_sol = np.zeros((n*m, R))
            V_sol = np.zeros((m*p, R))
            W_sol = np.zeros((p*n, R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms = assigned[(i,j,k)]
                        if len(terms) > 0:
                            val = 1.0 / len(terms)
                            for r in terms:
                                U_sol[i*m+j, r] = val ** (1/3)
                                V_sol[j*p+k, r] = val ** (1/3)
                                W_sol[k*n+i, r] = val ** (1/3)
            
            if verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                best_rank = R
                best_decomp = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                break
    
    if best_rank < rank_schoolbook:
        return best_decomp
    return fallback
# EVOLVE-BLOCK-END
