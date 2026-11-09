"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 80  # safety margin
    
    # Precompute: for each a in [1,n], and for each sum s in [2a+1, 2n],
    # check if a divides s. We'll use this to check feasibility.
    
    def is_feasible(S):
        """Check if set S satisfies the property."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                for m in range(j, k):
                    c = S_sorted[m]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def check_add(S_set, S_sorted, candidate):
        """Check if adding candidate to S maintains property. S_set is set, S_sorted is sorted list."""
        if candidate in S_set:
            return True
        # For each existing a in S: if a < candidate, then a is 'small', candidate is 'large'
        # For each existing b in S: if candidate < b, then candidate is 'small', b is 'large'
        # Cases:
        # 1. candidate is the smallest: for all b <= c in S, candidate must not divide b+c
        # 2. candidate is the largest: for all a < b <= c in S (where a,b,c are existing), already OK;
        #    also for all a < b in S where candidate=c: a must not divide b+candidate
        #    and for a < candidate (a in S, candidate is b or c): 
        #    Actually let's just do it generally.
        
        # General check: for all triples (x,y,z) in S ∪ {candidate} with x < y <= z, x ∤ (y+z)
        # New triples involving candidate:
        # Case A: candidate is smallest (candidate < all in S)
        #   For all y <= z in S: candidate ∤ (y+z)
        # Case B: candidate is in middle: some elements < candidate, some > candidate
        #   For x in S with x < candidate:
        #     For y in S with y > candidate, z in S with z >= y: x ∤ (y+z) -- already checked if x,y,z all in S
        #     Wait, we need to check triples where candidate is involved.
        #     Triples with candidate:
        #       (candidate, y, z) where y <= z in S and y > candidate: candidate ∤ (y+z)
        #       (x, candidate, z) where x < candidate < z in S: x ∤ (candidate+z)
        #       (x, y, candidate) where x < y < candidate in S: x ∤ (y+candidate)
        #       (x, candidate, candidate) where x < candidate in S: x ∤ (2*candidate)
        #       (candidate, y, y) where y > candidate in S: candidate ∤ (2*y)
        # Case C: candidate is largest:
        #   For x < y in S: x ∤ (y + candidate)
        #   For x in S: x ∤ (candidate + candidate) = x ∤ (2*candidate)
        
        # Let's just do a direct check for efficiency
        new_S = sorted(list(S_set) + [candidate])
        k = len(new_S)
        for i in range(k):
            a = new_S[i]
            for j in range(i+1, k):
                b = new_S[j]
                for m in range(j, k):
                    c = new_S[m]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Strategy 1: Start with upper tail [171, 256] and try to add smaller numbers
    best_set = set(range(171, 257))
    best_size = len(best_set)
    
    # Try adding numbers from 1 to 170
    for num in range(1, 171):
        if time.time() > deadline:
            break
        if check_add(best_set, None, num):
            best_set.add(num)
            best_size += 1
    
    # Local improvement: try swapping
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = sorted(best_set)
        for i in range(len(current_list)):
            for j in range(i+1, len(current_list)):
                if time.time() > deadline:
                    break
                a = current_list[i]
                b = current_list[j]
                # Try removing a and adding some other number
                # Try swapping: remove b, add a' where a' < 171 not in set
                # Actually let's try: remove one element, try to add two
                # Simpler: try single removal + addition
                for add_candidate in range(1, 171):
                    if add_candidate in best_set:
                        continue
                    new_set = best_set - {b}
                    if check_add(new_set, None, add_candidate):
                        new_set.add(add_candidate)
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_size = len(new_set)
                            improved = True
                            break
                if time.time() > deadline:
                    break
    
    # Another strategy: greedy from scratch with different orderings
    # Greedy from large to small
    greedy_set = set()
    for num in range(n, 0, -1):
        if time.time() > deadline:
            break
        if check_add(greedy_set, None, num):
            greedy_set.add(num)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Greedy from small to large
    greedy_set2 = set()
    for num in range(1, n+1):
        if time.time() > deadline:
            break
        if check_add(greedy_set2, None, num):
            greedy_set2.add(num)
    if len(greedy_set2) > best_size:
        best_set = greedy_set2
        best_size = len(greedy_set2)
    
    # Try to improve best_set further with local search
    # Try adding any number not in set
    for num in range(1, n+1):
        if time.time() > deadline:
            break
        if num not in best_set:
            if check_add(best_set, None, num):
                best_set.add(num)
                best_size += 1
    
    # Final verification
    final_list = sorted(best_set)
    # Verify feasibility
    k = len(final_list)
    for i in range(k):
        a = final_list[i]
        for j in range(i+1, k):
            b = final_list[j]
            for m in range(j, k):
                c = final_list[m]
                if (b + c) % a == 0:
                    # Should not happen, but return safe fallback
                    return list(range(171, 257))
    
    return final_list
# EVOLVE-BLOCK-END
