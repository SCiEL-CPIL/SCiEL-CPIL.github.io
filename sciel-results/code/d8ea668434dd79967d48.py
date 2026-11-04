"""Improved matrix multiplication tensor decomposition for M_{3,4,7}."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    def get_trivial_block(n, m, p):
        """Returns U, V, W for M_{n,m,p} with trivial rank n*m*p"""
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
        return U, V, W, rank
    
    def verify_full(U, V, W, rank, n, m, p):
        """Verify full decomposition for M_{n,m,p}"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m + j] * V_arr[j*p + k] * W_arr[k*n + i])
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid entries
        for a in range(n*m):
            i_a = a // m
            j_a = a % m
            for b in range(m*p):
                j_b = b // p
                k_b = b % p
                for c in range(p*n):
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(s) > 1e-9:
                            return False
        return True
    
    # Strategy: Decompose M_{3,4,7} into blocks
    # Option 1: Two M_{3,2,7} blocks (current approach)
    # Option 2: M_{3,3,7} + M_{3,1,7}
    # Option 3: Try to find better decompositions for smaller blocks
    
    # First, get the trivial decomposition as fallback
    U_triv, V_triv, W_triv, rank_triv = get_trivial_block(n, m, p)
    
    # Try to improve by finding better decompositions for M_{3,2,7}
    # The trivial rank for M_{3,2,7} is 3*2*7 = 42
    # If we can find rank 41 for M_{3,2,7}, total rank = 82
    
    n1, m1, p1 = 3, 2, 7
    U1_triv, V1_triv, W1_triv, rank1_triv = get_trivial_block(n1, m1, p1)
    
    best_U1 = U1_triv
    best_V1 = V1_triv
    best_W1 = W1_triv
    best_rank1 = rank1_triv
    
    # Try structured search for lower rank M_{3,2,7}
    # Strategy: Start from trivial decomposition and try to merge pairs of terms
    # by finding linear dependencies
    
    target_rank1 = rank1_triv - 1
    attempts = 0
    max_attempts = 5000
    
    # Precompute the trivial decomposition indices
    # For M_{3,2,7}: n1=3, m1=2, p1=7
    # Trivial rank = 42, each term (i,j,k) maps to index i*14 + j*7 + k
    
    # Try to find a rank-41 decomposition by merging two trivial terms
    # We need: sum over r of U[a,r]*V[b,r]*W[c,r] = 1 for valid (a,b,c), 0 otherwise
    
    # Approach: Try all pairs of trivial terms to see if we can merge them
    # Term t corresponds to (i_t, j_t, k_t) where t = i_t*14 + j_t*7 + k_t
    
    def get_term_index(i, j, k):
        return i * 14 + j * 7 + k
    
    # For each pair of terms, check if we can create a 2-term replacement
    # that satisfies all constraints
    
    found = False
    for t1 in range(rank1_triv):
        if time.time() > deadline - 10:
            break
        i1, j1, k1 = (t1 // 14), ((t1 // 7) % 2), (t1 % 7)
        
        for t2 in range(t1 + 1, rank1_triv):
            if time.time() > deadline - 10:
                break
            i2, j2, k2 = (t2 // 14), ((t2 // 7) % 2), (t2 % 7)
            
            # Try to merge terms t1 and t2 into a single term
            # We need to find U, V, W vectors for the new term such that:
            # For all valid (a,b,c): new_term_contribution + other_terms = 1
            # For all invalid (a,b,c): new_term_contribution + other_terms = 0
            
            # Simplified approach: try specific structured patterns
            # Pattern 1: Both terms share same i, different j,k
            # Pattern 2: Both terms share same j, different i,k
            # Pattern 3: Both terms share same k, different i,j
            
            # Try a few specific merge patterns
            merge_patterns = [
                # (U_scale1, U_scale2, V_scale1, V_scale2, W_scale1, W_scale2)
                (1, 0, 1, 1, 1, 1),  # Try to absorb t2 into t1
                (1, 1, 1, 0, 1, 1),  # Try to absorb t1 into t2
                (1, 1, 1, 1, 1, 0),
                (1, 1, 1, 1, 0, 1),
                (Fraction(1,2), Fraction(1,2), 1, 1, 1, 1),
                (1, 1, Fraction(1,2), Fraction(1,2), 1, 1),
                (1, 1, 1, 1, Fraction(1,2), Fraction(1,2)),
            ]
            
            for (us1, us2, vs1, vs2, ws1, ws2) in merge_patterns:
                if time.time() > deadline - 10:
                    break
                
                # Create a candidate decomposition with rank target_rank1
                # Remove terms t1 and t2, add one new term
                U1 = [[0] * target_rank1 for _ in range(n1*m1)]
                V1 = [[0] * target_rank1 for _ in range(m1*p1)]
                W1 = [[0] * target_rank1 for _ in range(p1*n1)]
                
                # Copy all terms except t1 and t2
                new_term_idx = target_rank1 - 1
                for t in range(rank1_triv):
                    if t == t1 or t == t2:
                        continue
                    r = t if t < new_term_idx else t - 1
                    i_t, j_t, k_t = (t // 14), ((t // 7) % 2), (t % 7)
                    U1[i_t*m1 + j_t][r] = 1
                    V1[j_t*p1 + k_t][r] = 1
                    W1[k_t*n1 + i_t][r] = 1
                
                # Add the new merged term
                # The new term should contribute to both (i1,j1,k1) and (i2,j2,k2)
                # This is tricky - let's try a different approach
                
                # Actually, let's try a more direct approach:
                # Start with trivial and try to zero out one term by adjusting others
                pass
    
    # Alternative: Use a more systematic random search with better initialization
    search_values = [0, 1, -1, Fraction(1,2), Fraction(-1,2), Fraction(2,3), Fraction(-2,3)]
    
    while time.time() < deadline - 15 and attempts < max_attempts:
        attempts += 1
        
        # Use a sparser initialization with more structure
        # Initialize with trivial pattern but with some random perturbations
        U1 = [[0] * target_rank1 for _ in range(n1*m1)]
        V1 = [[0] * target_rank1 for _ in range(m1*p1)]
        W1 = [[0] * target_rank1 for _ in range(p1*n1)]
        
        # Start with first target_rank1 terms from trivial decomposition
        for t in range(target_rank1):
            i_t, j_t, k_t = (t // 14), ((t // 7) % 2), (t % 7)
            U1[i_t*m1 + j_t][t] = 1
            V1[j_t*p1 + k_t][t] = 1
            W1[k_t*n1 + i_t][t] = 1
        
        # Randomly perturb some entries
        num_perturbations = random.randint(2, 8)
        for _ in range(num_perturbations):
            mat = random.choice(['U', 'V', 'W'])
            row = random.randint(0, {'U': n1*m1-1, 'V': m1*p1-1, 'W': p1*n1-1}[mat])
            col = random.randint(0, target_rank1-1)
            val = random.choice(search_values)
            
            if mat == 'U':
                U1[row][col] = val
            elif mat == 'V':
                V1[row][col] = val
            else:
                W1[row][col] = val
        
        # Quick check: valid entries sum to 1
        valid_ok = True
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    s = 0
                    for r in range(target_rank1):
                        s += U1[i*m1 + j][r] * V1[j*p1 + k][r] * W1[k*n1 + i][r]
                    if abs(s - 1) > 1e-9:
                        valid_ok = False
                        break
                if not valid_ok: break
            if not valid_ok: break
            
        if valid_ok:
            # Full verification
            U1_arr = np.array(U1, dtype=float)
            V1_arr = np.array(V1, dtype=float)
            W1_arr = np.array(W1, dtype=float)
            
            invalid_ok = True
            for a in range(n1*m1):
                i_a = a // m1
                j_a = a % m1
                for b in range(m1*p1):
                    j_b = b // p1
                    k_b = b % p1
                    for c in range(p1*n1):
                        k_c = c // n1
                        i_c = c % n1
                        
                        is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                        
                        if not is_valid:
                            s = np.sum(U1_arr[a] * V1_arr[b] * W1_arr[c])
                            if abs(s) > 1e-9:
                                invalid_ok = False
                                break
                    if not invalid_ok: break
                if not invalid_ok: break
            
            if invalid_ok:
                best_U1 = U1
                best_V1 = V1
                best_W1 = W1
                best_rank1 = target_rank1
                found = True
                break
    
    # If we found a better rank for M_{3,2,7}, also try M_{3,3,7} + M_{3,1,7}
    # M_{3,3,7} trivial rank = 63, M_{3,1,7} trivial rank = 21, total = 84
    # M_{3,2,7} + M_{3,2,7} trivial rank = 42 + 42 = 84
    # So both are equivalent in trivial case
    
    # Also try M_{2,4,7} + M_{1,4,7}: trivial ranks 56 + 28 = 84
    # Or M_{3,4,5} + M_{3,4,2}: trivial ranks 60 + 24 = 84
    
    # The current approach of 2x M_{3,2,7} seems reasonable
    rank_total = best_rank1 * 2
    
    U_total = [[0] * rank_total for _ in range(n * m)]  # 12 rows
    V_total = [[0] * rank_total for _ in range(m * p)]  # 28 rows
    W_total = [[0] * rank_total for _ in range(p * n)]  # 21 rows
    
    # Mapping for Block 1 (j=0,1)
    u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5, 4: 8, 5: 9}
    v_map_1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
               7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13}
    w_map_1 = {k*n1 + i: k*n1 + i for k in range(p1) for i in range(n1)}
    
    # Mapping for Block 2 (j=2,3)
    u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7, 4: 10, 5: 11}
    v_map_2 = {0: 14, 1: 15, 2: 16, 3: 17, 4: 18, 5: 19, 6: 20,
               7: 21, 8: 22, 9: 23, 10: 24, 11: 25, 12: 26, 13: 27}
    w_map_2 = w_map_1
    
    # Fill U_total
    for r in range(best_rank1):
        for u_idx_1, u_idx_orig in u_map_1.items():
            U_total[u_idx_orig][r] = best_U1[u_idx_1][r]
        for u_idx_2, u_idx_orig in u_map_2.items():
            U_total[u_idx_orig][r + best_rank1] = best_U1[u_idx_2][r]
    
    # Fill V_total
    for r in range(best_rank1):
        for v_idx_1, v_idx_orig in v_map_1.items():
            V_total[v_idx_orig][r] = best_V1[v_idx_1][r]
        for v_idx_2, v_idx_orig in v_map_2.items():
            V_total[v_idx_orig][r + best_rank1] = best_V1[v_idx_2][r]
    
    # Fill W_total
    for r in range(best_rank1):
        for w_idx, w_idx_orig in w_map_1.items():
            W_total[w_idx_orig][r] = best_W1[w_idx][r]
            W_total[w_idx_orig][r + best_rank1] = best_W1[w_idx][r]
    
    # Verify the final result
    if not verify_full(U_total, V_total, W_total, rank_total, n, m, p):
        # Fallback to trivial decomposition
        return {"U": U_triv, "V": V_triv, "W": W_triv}
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
