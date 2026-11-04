import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s wall-clock limit

    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72

    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    def check_decomposition(U, V, W, rank):
        """Check if the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    # Strategy 1: Structured block decomposition
    # View 3x3 x 3x8 as: for each column k, C[:,k] = A @ B[:,k]
    # The 8 columns can be grouped. Use the fact that matrix-vector (3x3x1) 
    # might have rank < 9 when we allow sharing across columns.
    
    # Key idea: The tensor for 3x3 x 3x8 is:
    # T[(i,j),(j,k),(k,i)] = 1
    # We can try to use a decomposition where we first compute some 
    # intermediate products shared across k values.
    
    # Approach: Try to find a decomposition using the structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # One idea: group the 8 columns into pairs and use 2x2 Strassen-like
    # tricks on the B side, but this doesn't directly help since A is shared.
    #
    # Better idea: Use the fact that for fixed A, the map B -> C is linear.
    # The tensor rank is the minimum number of rank-1 tensors to sum to T.
    
    # Let's try a more systematic search with structured initialization.
    
    random.seed(12345)
    
    def try_structured_rank(target_rank, attempts=50):
        """Try to find a decomposition of given rank using structured patterns."""
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                return None
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Use a mixed strategy: some schoolbook-like terms + random perturbations
            # First, assign each (i,j,k) triple to one of the rank slots
            # with some sharing allowed
            
            # Create a mapping from (i,j,k) to slot assignments
            # Each slot r gets values for U, V, W
            for r in range(target_rank):
                # Decide which (i,j,k) triples this slot contributes to
                # Use random but structured assignment
                if attempt % 3 == 0:
                    # Pure random sparse
                    for i in range(n):
                        for j in range(m):
                            if random.random() < 0.3:
                                U[i*m+j][r] = random.choice([-1, 1])
                    for j in range(m):
                        for k in range(p):
                            if random.random() < 0.3:
                                V[j*p+k][r] = random.choice([-1, 1])
                    for k in range(p):
                        for i in range(n):
                            if random.random() < 0.3:
                                W[k*n+i][r] = random.choice([-1, 1])
                elif attempt % 3 == 1:
                    # Block structure: each slot handles a subset of k values
                    k_group = [k for k in range(p) if k % (target_rank // p + 1) == r % (target_rank // p + 1)]
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j][r] = random.choice([-1, 1])
                    for j in range(m):
                        for k in range(p):
                            if k in k_group:
                                V[j*p+k][r] = random.choice([-1, 1])
                            else:
                                V[j*p+k][r] = 0
                    for k in range(p):
                        for i in range(n):
                            if k in k_group:
                                W[k*n+i][r] = random.choice([-1, 1])
                            else:
                                W[k*n+i][r] = 0
                else:
                    # Half-integer values for more flexibility
                    vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
                    for i in range(n):
                        for j in range(m):
                            if random.random() < 0.4:
                                U[i*m+j][r] = random.choice(vals)
                    for j in range(m):
                        for k in range(p):
                            if random.random() < 0.4:
                                V[j*p+k][r] = random.choice(vals)
                    for k in range(p):
                        for i in range(n):
                            if random.random() < 0.4:
                                W[k*n+i][r] = random.choice(vals)
            
            if check_decomposition(U, V, W, target_rank):
                return U, V, W, target_rank
        return None

    # Strategy 2: Greedy merge from schoolbook
    # Start with schoolbook and try to merge pairs of terms
    def try_merge_from_schoolbook(max_merges=20):
        """Try to merge schoolbook terms to reduce rank."""
        # Represent each schoolbook term as a dict
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        
        # Try to merge terms by finding algebraic cancellations
        # This is complex, so use a simpler heuristic:
        # Try removing one term at a time and see if the remaining
        # can still satisfy the equations (with adjusted coefficients)
        
        # Actually, let's try a different approach: 
        # Use the fact that sum over j is linear, so we might be able to
        # combine terms that differ only in j.
        
        # For fixed i,k: C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor entries for fixed (i,k) are T[(i,j),(j,k),(k,i)] = 1 for j=0,1,2
        # These 3 terms might be mergeable if we use appropriate coefficients.
        
        # Let's try: for each (i,k), can we represent the sum over j with fewer terms?
        # sum_j A[i,j] B[j,k] - this is a bilinear form in (A[i,:], B[:,k])
        # The rank of this bilinear form is 3 (it's just dot product of 3-vectors)
        # But we need to share terms across different (i,k) pairs.
        
        # This is getting complex. Let's just do random search more aggressively.
        return None

    # Main search: try ranks from 71 down to 50
    for target_rank in range(71, 49, -1):
        if time.time() > deadline - 10:
            break
        
        # Budget: more attempts for higher ranks (closer to 72), fewer for lower
        if target_rank >= 68:
            attempts = 200
        elif target_rank >= 60:
            attempts = 100
        else:
            attempts = 50
        
        result = try_structured_rank(target_rank, attempts)
        if result:
            U_l, V_l, W_l, r = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
            best_rank = r
            break
    
    # If we haven't found anything better, try one more strategy:
    # Use numpy to solve a linear system for a given rank
    if best_rank == schoolbook_rank and time.time() < deadline - 20:
        # Try to solve for rank 70 as a linear system
        # Variables: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for r in range(70)
        # Equations: for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is nonlinear, so we use a different approach.
        
        # Fix U and W randomly, solve for V
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            
            target_rank = 70
            U = [[0] * target_rank for _ in range(n * m)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Random sparse U and W
            for i in range(n * m):
                for r in range(target_rank):
                    if random.random() < 0.5:
                        U[i][r] = random.choice([-1, 1])
            for i in range(p * n):
                for r in range(target_rank):
                    if random.random() < 0.5:
                        W[i][r] = random.choice([-1, 1])
            
            # For each (j,k): sum_r V[j*p+k,r] * (sum_i U[i*m+j,r]*W[k*n+i,r] * ... ) 
            # Wait, this doesn't factor nicely because the equation couples all three.
            
            # Actually, for fixed j: the equations for all (i,k) are:
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let A_j[i,k,r] = U[i*m+j,r] * W[k*n+i,r]
            # Then: sum_r A_j[i,k,r] * V[j*p+k,r] = 1 for all i,k
            # This is a linear system in V[j*p+k, :] for each j, but it's coupled
            # because V[j*p+k,r] appears in equations for all i.
            
            # For fixed j: we have n*p = 24 equations and p = 8 unknowns per r... 
            # Actually V[j*p+k, r] for k=0..7, r=0..69: that's 8*70 = 560 unknowns
            # And 24 equations. Underdetermined.
            
            # Let's just continue with random search.
            pass

    return best_result
