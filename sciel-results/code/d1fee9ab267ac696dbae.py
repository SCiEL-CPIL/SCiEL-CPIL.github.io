import math
import time

# EVOLVE-BLOCK-START
P = 27720  # lcm(1..12)
N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_m = 0
    best_L = 0
    
    # Precompute multiples conditions? 
    # For each m, we check L=1,2,...
    
    # To optimize, we can check larger L first? No, we need the MIN L for each m.
    # But we want to MAXIMIZE this min L.
    # So we compute min_L(m) for all m and pick the m with the largest min_L.
    
    for m in range(P):
        # Find min L for this m
        # We can stop searching L if current best_L is already beaten? 
        # No, min L is the value. We want the m where this value is highest.
        # So we must find the exact min L for each m.
        
        # Optimization: If we find an L such that a matching exists, that's the min L.
        # We iterate L from 1 upwards.
        
        min_L = None
        # Upper bound for L: worst case is around 100-150.
        # Let's iterate L from 1 to 200.
        for L in range(1, 201):
            if time.time() > deadline:
                return int(best_m)
                
            # Check if matching exists for interval (m, m+L]
            # Available numbers are m+1 ... m+L
            # For each i in 1..12, valid j (1..L) if (m+j) % i == 0
            
            # Backtracking search
            # Assign i from 1 to 12
            # State: used_j_mask (bitmask of L bits? L can be 144, so 144-bit int is fine in Python)
            # Or just a set/list of used j's.
            
            # Let's use a list of available j's for each i
            # avail[i] = list of j in 1..L such that (m+j) % i == 0
            
            # If any avail[i] is empty, no match.
            # Then backtrack.
            
            # For speed, precompute avail lists?
            # The condition (m+j) % i == 0 depends on m.
            # So we have to compute it for each m, L.
            
            # We can speed this up by noting that for a fixed i, the valid j's are periodic.
            # But L is small.
            
            # Let's just do the check.
            
            # Early exit: if any i has no valid j, return False
            # Also, if total available j's < 12, return False (though they must be distinct, so this is implicit in matching)
            
            # We'll use a simple recursive backtracking with memoization on (i, used_mask)?
            # used_mask can be large (L bits). But depth is only 12.
            # So branching factor is small.
            
            # Let's implement a helper function or inline it.
            
            # Optimization: Order i's by number of available choices (fewest first) to prune early.
            # But i is fixed 1..12. We can sort indices by len(avail[i]).
            
            indices = list(range(1, N+1))
            # We need to build avail lists first to sort them.
            avail = [[] for _ in range(N+1)]
            for i in range(1, N+1):
                # j in 1..L such that (m+j) % i == 0
                # m+j = k*i => j = k*i - m
                # 1 <= j <= L => 1 <= k*i - m <= L => m+1 <= k*i <= m+L
                # k_min = ceil((m+1)/i)
                # k_max = floor((m+L)/i)
                k_min = (m + i) // i  # ceil((m+1)/i) is (m+i)//i? 
                # Let's check: m=0, i=2. ceil(1/2)=1. (0+2)//2 = 1. Correct.
                # m=1, i=2. ceil(2/2)=1. (1+2)//2 = 1. Correct.
                # m=2, i=2. ceil(3/2)=2. (2+2)//2 = 2. Correct.
                # Formula: (m + i) // i is correct for ceil((m+1)/i)?
                # Wait. ceil(A/B) = (A+B-1)//B.
                # A = m+1. B = i.
                # ceil((m+1)/i) = (m+1+i-1)//i = (m+i)//i. Yes.
                
                k_max = (m + L) // i
                
                if k_min <= k_max:
                    for k in range(k_min, k_max + 1):
                        j = k * i - m
                        avail[i].append(j)
                else:
                    avail[i].append(None) # Marker for empty? Or just check len.
            
            # Check if any avail[i] is empty
            valid = True
            for i in range(1, N+1):
                if not avail[i]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Sort indices by number of options to improve backtracking
            # We need to map sorted indices back to original i for the constraint?
            # The constraints are on i (divisibility). The variables are j.
            # If we swap the order of processing i's, the logic remains the same.
            # We just need to ensure we pick distinct j's.
            
            sorted_indices = sorted(range(1, N+1), key=lambda x: len(avail[x]))
            
            # Backtracking
            used_j = set()
            
            def backtrack(idx_in_sorted):
                if idx_in_sorted == N:
                    return True
                i = sorted_indices[idx_in_sorted]
                for j in avail[i]:
                    if j not in used_j:
                        used_j.add(j)
                        if backtrack(idx_in_sorted + 1):
                            return True
                        used_j.remove(j)
                return False
            
            if backtrack(0):
                min_L = L
                break
        
        if min_L is not None:
            if min_L > best_L:
                best_L = min_L
                best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
