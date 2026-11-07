import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Precompute: for each pair (b, c), which values a would be invalidated by b*c % a == 0
    # i.e., a divides b*c
    # Also precompute for efficiency
    
    # For can_add(x, S): we need to check for all pairs (b,c) in S:
    #   (b*c) % x != 0
    #   (x*c) % b != 0
    #   (x*b) % c != 0
    
    # Precompute divisibility: divs[a] = set of multiples of a in [2,256]
    # Actually, let's precompute a matrix or use bitmasks for speed
    
    # Let's use a different approach: precompute for each value x, 
    # the set of values y such that x | y (x divides y)
    # And for each pair (b,c), the set of a such that a | (b*c)
    
    # For speed, precompute:
    # divides[x] = set of y in [2,256] where x divides y
    divides = [set() for _ in range(N + 1)]
    for x in range(2, N + 1):
        for y in range(x, N + 1, x):
            divides[x].add(y)
    
    # For each pair (b, c), precompute which a values would conflict: a | (b*c)
    # This is expensive: 255^2 pairs * checking divisors of b*c
    # Instead, let's compute on the fly but with optimization
    
    # Key insight: a | (b*c) iff a/gcd(a,b) | c
    # So for a given x being added, and existing b,c in S:
    #   x | (b*c) iff x/gcd(x,b) | c
    #   b | (x*c) iff b/gcd(b,x) | c
    #   c | (x*b) iff c/gcd(c,x) | b
    
    # Precompute gcd table for speed
    # gcd_table[i][j] for i,j in 2..256
    # That's 255x255 = 65025 entries, very manageable
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(2, N + 1):
            gcd_table[i][j] = gcd(i, j)
    
    # Precompute: for each x and each b, compute x/gcd(x,b) and b/gcd(b,x)
    # div_ratio[x][b] = x // gcd(x,b)
    div_ratio = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(2, N + 1):
        for b in range(2, N + 1):
            g = gcd_table[x][b]
            div_ratio[x][b] = x // g
    
    def can_add_fast(x, S_list):
        # Check if x can be added to the set S_list (list of ints)
        # For each pair (b, c) in S_list, check:
        #   x | (b*c)  <=>  x/gcd(x,b) | c
        #   b | (x*c)  <=>  b/gcd(b,x) | c  (same as x/gcd(x,b) | c since gcd is symmetric)
        #   c | (x*b)  <=>  c/gcd(c,x) | b
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            # x | (b*c) for some c in S_list[i+1:]
            # b | (x*c) for some c in S_list[i+1:]
            # These two are equivalent: x/gcd(x,b) | c
            r1 = div_ratio[x][b]
            for j in range(i + 1, L):
                c = S_list[j]
                # Check if r1 divides c
                if c % r1 == 0:
                    return False
                # Check if c/gcd(c,x) divides b
                r2 = div_ratio[c][x]
                if b % r2 == 0:
                    return False
        return True
    
    def can_add(x, S):
        S_list = list(S)
        return can_add_fast(x, S_list)
    
    # Greedy Construction - try multiple strategies and pick best
    def greedy_descending():
        current_set = set()
        current_list = []
        for x in range(N, 1, -1):
            if can_add_fast(x, current_list):
                current_set.add(x)
                current_list.append(x)
        return current_set, current_list
    
    def greedy_ascending():
        current_set = set()
        current_list = []
        for x in range(2, N + 1):
            if can_add_fast(x, current_list):
                current_set.add(x)
                current_list.append(x)
        return current_set, current_list
    
    def greedy_randomized(seed):
        rng = random.Random(seed)
        current_set = set()
        current_list = []
        candidates = list(range(2, N + 1))
        rng.shuffle(candidates)
        for x in candidates:
            if can_add_fast(x, current_list):
                current_set.add(x)
                current_list.append(x)
        return current_set, current_list
    
    # Try multiple greedy strategies
    best_set, best_list = greedy_descending()
    best_size = len(best_set)
    
    alt_set, alt_list = greedy_ascending()
    if len(alt_set) > best_size:
        best_set, best_list = alt_set, alt_list
        best_size = len(alt_set)
    
    # Try a few randomized greedy
    for seed in range(5):
        r_set, r_list = greedy_randomized(seed)
        if len(r_set) > best_size:
            best_set, best_list = r_set, r_list
            best_size = len(r_set)
    
    current_set = set(best_set)
    current_list = list(best_set)
    current_list.sort()
    
    # Local Search with 1-opt (swap) and 2-opt (remove 2, add 2)
    def try_add_any(current_set, current_list):
        """Try to add any remaining element. Returns (success, new_set, new_list)"""
        remaining = [x for x in range(2, N + 1) if x not in current_set]
        for x in remaining:
            if can_add_fast(x, current_list):
                new_set = current_set | {x}
                new_list = current_list + [x]
                new_list.sort()
                return True, new_set, new_list
        return False, current_set, current_list
    
    def try_swap(current_set, current_list):
        """Try to swap one element out and another in. Returns (success, new_set, new_list)"""
        a_candidates = list(current_set)
        random.shuffle(a_candidates)
        remaining = [x for x in range(2, N + 1) if x not in current_set]
        random.shuffle(remaining)
        
        for a in a_candidates:
            # Remove a
            S_new_list = [x for x in current_list if x != a]
            S_new_set = current_set - {a}
            for b in remaining:
                if can_add_fast(b, S_new_list):
                    new_set = S_new_set | {b}
                    new_list = S_new_list + [b]
                    new_list.sort()
                    return True, new_set, new_list
        return False, current_set, current_list
    
    def try_2opt(current_set, current_list):
        """Try to remove 2 elements and add 2 different ones. Returns (success, new_set, new_list)"""
        a_candidates = list(current_set)
        random.shuffle(a_candidates)
        remaining = [x for x in range(2, N + 1) if x not in current_set]
        random.shuffle(remaining)
        
        L = len(a_candidates)
        for i in range(L):
            a1 = a_candidates[i]
            for j in range(i + 1, L):
                a2 = a_candidates[j]
                S_new_list = [x for x in current_list if x != a1 and x != a2]
                S_new_set = current_set - {a1, a2}
                
                # Try to add two elements from remaining
                found = False
                for b1 in remaining:
                    if can_add_fast(b1, S_new_list):
                        temp_list = S_new_list + [b1]
                        for b2 in remaining:
                            if b2 != b1 and can_add_fast(b2, temp_list):
                                new_set = S_new_set | {b1, b2}
                                new_list = temp_list + [b2]
                                new_list.sort()
                                return True, new_set, new_list
                if found:
                    break
            # Early exit if too slow
            if time.time() - start_time > deadline:
                break
        return False, current_set, current_list
    
    # Main local search loop
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < deadline:
        # 1. Try to add
        added, current_set, current_list = try_add_any(current_set, current_list)
        if added:
            no_improve_count = 0
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                best_list = list(current_set)
            continue
        
        # 2. Try 1-opt swap
        swapped, current_set, current_list = try_swap(current_set, current_list)
        if swapped:
            no_improve_count = 0
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                best_list = list(current_set)
            continue
        
        # 3. Try 2-opt
        if time.time() - start_time < deadline - 5:  # Need time for 2-opt
            two_opt, current_set, current_list = try_2opt(current_set, current_list)
            if two_opt:
                no_improve_count = 0
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    best_list = list(current_set)
                continue
        
        # No improvement
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Try restarting from a different random point
            # Or just break
            break
    
    # Final: ensure we return the best found
    if len(current_set) > len(best_set):
        best_set = current_set
    
    result = sorted(best_set)
    
    # Verify types: must be Python ints in a list
    result = [int(x) for x in result]
    
    return result
