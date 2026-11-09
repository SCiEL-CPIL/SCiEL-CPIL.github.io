import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    size_a = n * m
    size_b = m * p
    size_c = p * n
    
    # Build the tensor T over F_2 (and conceptually over Q)
    # T[a, b, c] = 1 if indices align
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T_mod2 = np.zeros((size_a, size_b, size_c), dtype=np.int8)
    T_exact = np.zeros((size_a, size_b, size_c), dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_mod2[a, b, c] = 1
                T_exact[a, b, c] = 1

    # Strategy: Randomized Greedy Peeling over F_2
    # We try to find a set of rank-1 tensors that sum to T in F_2.
    # If this set also sums to T in Q (which implies the coefficients are 0 or 1 and no "cancellation" happened in Q that didn't happen in F_2, or rather, the support is identical and coefficients match), we use it.
    # Note: A sum of rank-1 tensors with coeff 1 in F_2 might not equal T in Q if there are overlaps (e.g., 1+1=0 in F_2 but 2 in Q).
    # So we must verify the exact rational sum.
    
    U_list = []
    V_list = []
    W_list = []
    
    T_curr_mod2 = T_mod2.copy()
    T_curr_exact = T_exact.copy() # We track the exact residual to verify later? 
    # Actually, if we just collect the indices, we can verify at the end.
    
    indices_found = []
    
    start_time = time.time()
    max_iter = 104 # Must be < 105 to improve
    
    # Randomize the search order to avoid getting stuck in a local minimum of the mod-2 peeling?
    # Peeling is deterministic given the "first non-zero" strategy.
    # We can randomize the choice of the non-zero entry to explore different decompositions.
    
    # However, for this small size, we can just do a few random restarts.
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(5): # 5 random restarts
        if time.time() - start_time > 115:
            break
            
        T_work = T_mod2.copy()
        indices = []
        
        for r in range(max_iter):
            # Find a non-zero entry in T_work
            # To make it "randomized", we can pick a random non-zero entry
            non_zero_indices = np.argwhere(T_work != 0)
            if len(non_zero_indices) == 0:
                break
                
            # Pick a random one
            idx_choice = non_zero_indices[random.randint(0, len(non_zero_indices) - 1)]
            a, b, c = idx_choice
            
            indices.append((a, b, c))
            
            # Subtract the rank-1 term e_a \otimes e_b \otimes e_c from T_work
            # This flips the parity of T_work[a, :, :]? No.
            # T_work[a, b, c] changes by 1.
            # The rank-1 term is 1 only at (a,b,c).
            # So T_work[a, b, c] = (T_work[a, b, c] - 1) % 2
            T_work[a, b, c] = (T_work[a, b, c] - 1) % 2
            
            # If we broke early, we have a decomposition of rank len(indices)
        
        if len(indices) < best_rank:
            # Verify if this decomposition works over Q
            # Sum of e_a \otimes e_b \otimes e_c for (a,b,c) in indices
            # Check if it equals T_exact
            # Since T_exact is 0/1, and we are summing 0/1 vectors,
            # if any entry has sum > 1, it's invalid for "exact decomposition" with 0/1 coefficients?
            # The task allows "integer/half-integer entries".
            # If we use coefficients 1, the sum is integer.
            # If T_exact has a 1, and we sum two 1s at that spot, we get 2. Mismatch.
            # If T_exact has a 1, and we sum one 1, we get 1. Match.
            # If T_exact has a 0, and we sum 0, we get 0. Match.
            # So the decomposition is valid over Q iff the indices are *unique* (no duplicates) and cover exactly the support of T.
            # But T has 105 non-zero entries. If we peel 105 unique entries, rank is 105.
            # If we peel fewer, we must have "cancelled" some entries in F_2, meaning we hit them twice.
            # If we hit them twice, the coefficient in the sum is 2.
            # T_exact has 1. 2 != 1. So it's invalid.
            # UNLESS we allow coefficients to be -1 or other integers?
            # The peeling gives us a set of vectors. We can assign coefficients.
            # But the "greedy peeling" usually implies coefficient 1.
            # If we allow arbitrary integer coefficients, the problem is much harder.
            # Given the "modular arithmetic" hint, it implies the coefficients are derived from the mod-2 process, likely 0 or 1.
            
            # Check for uniqueness
            if len(indices) == len(set(indices)):
                # If unique, and count == 105, it's schoolbook.
                # If unique and count < 105, it's impossible because T has 105 non-zeros.
                pass
            
            # Actually, if we found a decomposition of rank R < 105 over F_2, 
            # it means we used R rank-1 terms to sum to T mod 2.
            # This does NOT guarantee a rational decomposition.
            
            # Since it's extremely unlikely to find a valid rational decomposition this way,
            # we will just fallback.
            pass
            
        # Fallback to schoolbook if no better valid one found
        if best_U is None:
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
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
