import itertools
import math
import time

def score_set(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1/1)/log(1) is 0/0, limit is 1? Or just 1.
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    p = len(sum_set)
    d = len(diff_set)
    
    if p == 0: return 0
    ratio = d / p
    if ratio <= 1:
        # log(ratio) is negative or zero
        # We want to maximize, so ratio > 1 is good
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    val = 1 + math.log(ratio) / denom
    return val

def construct():
    start_time = time.time()
    deadline = 140.0 # 150s limit, keep margin
    
    # Initial best set
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Heuristic search
    # Try to expand the set or modify it
    
    # Candidate generation:
    # 1. Small exhaustive search for max_u <= 10
    # 2. Local search around best_set
    
    # Exhaustive for small max_u
    for max_u in range(1, 12):
        if time.time() > deadline:
            break
        # Subsets of {1, ..., max_u}
        # Too many for max_u=11 (2048 subsets). Manageable.
        for mask in range(1, 1 << max_u):
            U = [0]
            for i in range(max_u):
                if mask & (1 << i):
                    U.append(i + 1)
            if max(U) < max_u:
                continue # We want max to be exactly max_u for this loop iteration's logic?
                         # Actually score depends on max U.
                         # If we iterate max_u, we should ensure max(U)==max_u to avoid duplicates?
                         # No, just compute score.
            
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    # Local Search
    # Try adding, removing, or swapping elements
    # We need to bound the search space.
    # Let's try to increase max_u.
    
    current_set = best_set[:]
    current_score = best_score
    
    # Try extending the set by adding a new largest element
    # This is a greedy approach
    
    max_iterations = 10000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
            
        improved = False
        # Try adding a new element k
        # k should be larger than current max to potentially increase denominator slowly?
        # Or smaller to increase ratio?
        
        # Strategy: Try to find a set with high ratio.
        # Random restarts with local search?
        
        # Let's try a specific construction:
        # Sets of the form {0, 1, 2, ..., k} are bad (ratio ~1).
        # Sets like {0, 1, 3, 7, 15, ...} (2^k-1)?
        
        # Let's test geometric-like sets
        for base in [2, 3, 4]:
            U_test = [0]
            val = 1
            while val < 1000:
                U_test.append(val)
                val *= base
            s = score_set(U_test)
            if s > best_score:
                best_score = s
                best_set = U_test[:]
                
        # Try mixed bases or specific patterns
        # Pattern: 0, 1, 2, 4, 8, 16... (binary)
        U_bin = [0]
        val = 1
        while val < 1000:
            U_bin.append(val)
            val *= 2
        s = score_set(U_bin)
        if s > best_score:
            best_score = s
            best_set = U_bin[:]
            
        # Pattern: 0, 1, 2, 4, 7, 11, 16... (triangular numbers + 0?)
        # No, let's stick to search.
        
        # Local search on best_set
        # Pick a random element to remove or swap
        # Or add a random element
        
        if len(best_set) > 1:
            # Try removing a random element (not 0)
            idx = 1 + int(time.time_ns() % (len(best_set) - 1))
            U_test = best_set[:]
            U_test.pop(idx)
            s = score_set(U_test)
            if s > best_score:
                best_score = s
                best_set = U_test[:]
                improved = True
                
        # Try adding a random element
        max_u = max(best_set)
        # Add element in range [max_u, max_u*2]
        new_elem = max_u + int(time.time_ns() % (max_u + 1))
        if new_elem not in best_set:
            U_test = best_set + [new_elem]
            s = score_set(U_test)
            if s > best_score:
                best_score = s
                best_set = U_test[:]
                improved = True
        else:
            # Try swapping
            if len(best_set) > 1:
                idx = 1 + int(time.time_ns() % (len(best_set) - 1))
                U_test = best_set[:]
                old = U_test.pop(idx)
                new_elem = max_u + int(time.time_ns() % (max_u + 1))
                if new_elem not in U_test:
                    U_test.append(new_elem)
                    s = score_set(U_test)
                    if s > best_score:
                        best_score = s
                        best_set = U_test[:]
                        improved = True

        if not improved and time.time() - start_time > 10:
             # If stuck, try a different heuristic
             # Maybe the binary set is optimal for small sizes?
             # Let's check if we can do better with a specific known construction.
             # The prompt mentions size 2003.
             # This implies large sets are better.
             
             # Let's try to construct a large set.
             # If we take U = {0, 1, 2, ..., M}, score is low.
             # If we take U = {0, 1, 3, 9, 27, ...} (powers of 3).
             # Sumset of powers of 3 is unique representation (base 3 digits 0,1,2).
             # So |U+U| is large?
             # No, sum of two powers of 3 can collide?
             # 1+1=2 (not a power), 1+3=4, 3+3=6.
             # Actually, if U is {0, 3^0, 3^1, ..., 3^k}, then U+U has size roughly (2k+1)^2 / 2?
             # No, each sum is a sum of two base-3 numbers with digits 0 or 1.
             # The number of such sums is bounded by the number of pairs.
             # But many collisions?
             # 3^i + 3^j is unique for i!=j?
             # 3^0+3^1 = 4. 3^2=9.
             # 3^0+3^0=2.
             # It seems sums are distinct if we consider the "carry" structure?
             # In base 3, digits are 0 or 1. Sum of two such numbers has digits 0, 1, 2.
             # No carries if we just add?
             # 1*3^i + 1*3^i = 2*3^i.
             # 1*3^i + 1*3^j (i!=j) = 3^i + 3^j.
             # These are all distinct representations in base 3 (digits 0,1,2).
             # So |U+U| = number of pairs = (k+1)^2.
             # |U-U|: differences.
             # 3^i - 3^j.
             # These are also distinct?
             # 3^i - 3^j = 3^j (3^{i-j} - 1).
             # Since 3^{i-j}-1 is not a power of 3, and 3^j is, the factorization is unique?
             # Yes, likely.
             # So |U-U| = (k+1)^2 - 1 (excluding 0? No, 0 is included).
             # Pairs (i,j) with i!=j give non-zero diffs. i=j give 0.
             # So |U-U| = (k+1)^2.
             # Wait, 3^i - 3^j = -(3^j - 3^i).
             # So diffs come in pairs.
             # Total distinct diffs = 1 + (k+1)k.
             # Ratio = (k^2+k+1) / (k+1)^2 = (k^2+k+1)/(k^2+2k+1).
             # As k->inf, ratio -> 1.
             # Score -> 1.
             # So powers of 3 are bad for large k.
             
             # We need ratio > 1.
             # This requires |U-U| > |U+U|.
             # For powers of 3, they are equal (approx).
             
             # We need a set where differences are "more unique" or "more numerous" than sums.
             # Or sums collide more than differences.
             # Sums collide if a+b = c+d.
             # Differences collide if a-b = c-d => a+d = c+b.
             # This is the same condition!
             # a+b = c+d <=> a-c = d-b.
             # So the number of collisions in sums is related to collisions in differences.
             # However, the ranges are different.
             # Sums are in [0, 2 max U].
             # Diffs are in [-max U, max U].
             # The "density" of sums vs diffs might differ.
             
             # For U={0,1,3}, |U+U|=6, |U-U|=7.
             # Collisions in sum: 0+3=1+2? No 2 not in U.
             # 0+1=1, 0+3=3, 1+1=2, 1+3=4, 3+3=6.
             # All distinct?
             # 0+3=3. 1+2? No.
             # Wait, 0+3=3. Is there another pair? No.
             # 1+1=2.
             # So sums are distinct?
             # Pairs: (0,0)->0, (0,1)->1, (0,3)->3, (1,0)->1, (1,1)->2, (1,3)->4, (3,0)->3, (3,1)->4, (3,3)->6.
             # Values: 0, 1, 3, 2, 4, 6.
             # 1 appears twice. 3 appears twice. 4 appears twice.
             # So |U+U|=6. Pairs=9. Collisions=3.
             
             # Diffs:
             # (0,0)->0, (0,1)->-1, (0,3)->-3
             # (1,0)->1, (1,1)->0, (1,3)->-2
             # (3,0)->3, (3,1)->2, (3,3)->0
             # Values: 0, -1, -3, 1, -2, 3, 2.
             # All distinct?
             # 0 appears 3 times.
             # Others once.
             # So |U-U|=7. Pairs=9. Collisions=2 (for 0).
             
             # We want more collisions in sums than in differences.
             # This happens if the set is "additively rich" but "subtractively sparse"?
             # Or vice versa.
             
             # Let's try to maximize collisions in sums.
             # This means many a+b = c+d.
             # This is typical for arithmetic progressions.
             # But APs have ratio ~1.
             
             # Let's try a set with many small elements.
             # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
             # |U+U| = 21.
             # |U-U| = 21.
             # Ratio 1.
             
             # What if we take {0, 1, 2, 4, 5, 8, 9, 16, 17, 32, 33...}?
             # Blocks of 2.
             # Let's test this pattern.
             
             U_block = [0]
             for k in range(10):
                 base = 2**(2*k)
                 U_block.append(base)
                 U_block.append(base + 1)
             
             s = score_set(U_block)
             if s > best_score:
                 best_score = s
                 best_set = U_block[:]

    return best_set
