import time
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 80
    
    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True

    def score_of(A, B):
        s = {a + b for a in A for b in B}
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)

    def is_sumset(S):
        S_list = sorted(S)
        n = len(S_list)
        if n != 9:
             # Fallback for smaller sizes? 
             # For now, assume we only call with 9.
             # If we need smaller, we can generalize.
             return None
             
        # Try all subsets R1 of size 3
        for R1 in combinations(S_list, 3):
            R1_set = set(R1)
            # Calculate difference set of R1
            diffs_R1 = set()
            for i in range(3):
                for j in range(3):
                    if i != j:
                        diffs_R1.add(abs(R1[i] - R1[j]))
            
            S_rem = list(set(S_list) - R1_set)
            # We need to partition S_rem into R2 and R3
            # R2 must have same diff set as R1
            # R3 must have same diff set as R1
            
            # Iterate over subsets R2 of S_rem of size 3
            for R2 in combinations(S_rem, 3):
                R2_set = set(R2)
                R3_set = set(S_rem) - R2_set
                if len(R3_set) != 3: continue
                
                # Check if R2 has same diff set
                diffs_R2 = set()
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            diffs_R2.add(abs(R2[i] - R2[j]))
                if diffs_R2 != diffs_R1: continue
                
                # Check if R3 has same diff set
                R3_list = list(R3_set)
                diffs_R3 = set()
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            diffs_R3.add(abs(R3_list[i] - R3_list[j]))
                if diffs_R3 != diffs_R1: continue
                
                # If structure matches, try to reconstruct A and B
                # R1 = A + b1
                # R2 = A + b2
                # R3 = A + b3
                
                # Let A = {a1, a2, a3}
                # We can set b1 = 0 (relative) to find A?
                # No, A and B must be in [1, 128].
                # Let's express A in terms of R1.
                # a_i = r1_i - b1.
                # b2 = b1 + shift
                # b3 = b1 + shift3
                # We need to find b1 such that:
                # 1 <= a_i <= 128 for all i
                # 1 <= b_j <= 128 for all j
                # a_i are distinct (implied by R1 distinct)
                # b_j are distinct (implied by shifts distinct? shift, shift3, 0 must be distinct)
                
                R1_sorted = sorted(R1)
                R2_sorted = sorted(R2)
                shift = R2_sorted[0] - R1_sorted[0]
                if R2_sorted[1] - R1_sorted[1] != shift or R2_sorted[2] - R1_sorted[2] != shift:
                    continue
                
                # Similarly for R3
                R3_sorted = sorted(R3_set)
                shift3 = R3_sorted[0] - R1_sorted[0]
                if R3_sorted[1] - R1_sorted[1] != shift3 or R3_sorted[2] - R1_sorted[2] != shift3:
                    continue
                
                # Now we have R1, R2, R3 as translations of each other.
                # R1 = A + b1
                # R2 = A + b2
                # R3 = A + b3
                # Let A = {a1, a2, a3}.
                # We can set b1 = 0 (relative) to find A?
                # No, A and B must be in [1, 128].
                # Let's express A in terms of R1.
                # a_i = r1_i - b1.
                # b2 = b1 + shift
                # b3 = b1 + shift3
                # We need to find b1 such that:
                # 1 <= a_i <= 128 for all i
                # 1 <= b_j <= 128 for all j
                # a_i are distinct (implied by R1 distinct)
                # b_j are distinct (implied by shifts distinct? shift, shift3, 0 must be distinct)
                
                if shift == 0 or shift3 == 0 or shift == shift3:
                    continue # B elements not distinct
                
                # Range for b1:
                # a_i = r1_i - b1
                # 1 <= r1_i - b1 <= 128  =>  r1_i - 128 <= b1 <= r1_i - 1
                # b1 >= 1
                # b1 + shift >= 1 => b1 >= 1 - shift
                # b1 + shift3 >= 1 => b1 >= 1 - shift3
                # b1 <= 128
                # b1 + shift <= 128 => b1 <= 128 - shift
                # b1 + shift3 <= 128 => b1 <= 128 - shift3
                
                min_b1 = max(1, 1 - shift, 1 - shift3)
                max_b1 = min(128, 128 - shift, 128 - shift3)
                # Also from a_i constraints:
                for r in R1_sorted:
                    min_b1 = max(min_b1, r - 128)
                    max_b1 = min(max_b1, r - 1)
                
                if min_b1 > max_b1:
                    continue
                
                # Pick any valid b1
                b1_val = min_b1
                A = [r - b1_val for r in R1_sorted]
                B = [b1_val, b1_val + shift, b1_val + shift3]
                
                # Final validation
                if len(set(A)) != 3 or len(set(B)) != 3:
                    continue
                if not all(1 <= x <= 128 for x in A):
                    continue
                if not all(1 <= x <= 128 for x in B):
                    continue
                
                return A, B
        return None

    # Precompute primes up to 256
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    primes = [n for n in range(2, 257) if is_prime(n)]
    # Include 1 and 2 (2 is prime, 1 is coprime to all)
    candidates = [1] + primes
    
    # Generate coprime sets of size 9
    # Backtracking
    found_solution = None
    
    def backtrack(start_idx, current_set):
        nonlocal found_solution
        if found_solution is not None:
            return
        if len(current_set) == 9:
            # Check if this set is a sumset
            A, B = is_sumset(current_set)
            if A and B:
                found_solution = (A, B)
            return
        
        # Pruning: if we can't reach size 9
        remaining = len(candidates) - start_idx
        if len(current_set) + remaining < 9:
            return
            
        for i in range(start_idx, len(candidates)):
            num = candidates[i]
            # Check coprime with current set
            ok = True
            for x in current_set:
                if gcd(num, x) != 1:
                    ok = False
                    break
            if ok:
                backtrack(i + 1, current_set + [num])
                if found_solution is not None:
                    return

    # Start backtracking
    # To find a solution quickly, we might want to try smaller numbers first?
    # Candidates are sorted.
    backtrack(0, [])
    
    if found_solution:
        return found_solution

    # Fallback: Random search if backtracking fails (should not happen for size 9)
    # Or try size 8, 7...
    
    # Let's try size 8
    def backtrack_size(target_size, start_idx, current_set):
        nonlocal found_solution
        if found_solution is not None:
            return
        if len(current_set) == target_size:
            # Check if this set is a sumset
            # is_sumset currently assumes size 9.
            # We need a generalized is_sumset or just return the set and check manually?
            # For size 8, A+B can have 8 distinct sums.
            # This implies one collision.
            # This is more complex.
            # Let's stick to finding size 9.
            # If we really need fallback, we can do random search.
            return
        
        remaining = len(candidates) - start_idx
        if len(current_set) + remaining < target_size:
            return
            
        for i in range(start_idx, len(candidates)):
            num = candidates[i]
            ok = True
            for x in current_set:
                if gcd(num, x) != 1:
                    ok = False
                    break
            if ok:
                backtrack_size(target_size, i + 1, current_set + [num])
                if found_solution is not None:
                    return

    # If no size 9 found, try random search as before
    import random
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    while time.time() < deadline - 5:
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A, B
            if best_score >= 9:
                break
                
    return [best_A, best_B]

