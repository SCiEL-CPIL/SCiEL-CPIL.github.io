"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    n_candidates = len(candidates)
    
    # Precompute: for each x, the set of pairs (i, j) where i, j are indices in candidates
    # such that candidates[i] * candidates[j] is divisible by x
    # Actually, we need: for a given x and set S, check if any pair in S has product divisible by x
    # And: for a given x, a in S, check if there exists c in S, c != a, such that a | x*c
    
    # Let's precompute for each x in 2..256:
    # conflict_pairs[x] = set of (i, j) where i < j, and candidates[i] * candidates[j] % x == 0
    # This is O(N^3) precomputation but N=256 so 256^3/6 ~ 2.8M, might be slow
    # Let's do it more carefully
    
    # Actually, let's precompute:
    # For each x, multiples_of_x = [m for m in candidates if m % x == 0]
    # For the check "x | b*c", we need b*c to be a multiple of x
    # This is equivalent to: for prime factorization x = p1^a1 * p2^a2 * ...
    # b*c must have at least ai of prime pi for each i
    
    # Simpler approach: precompute for each x, the set of values v in 2..256 such that v % x == 0
    # Then for checking if x | b*c, we can check if (b*c) % x == 0 directly
    
    # Let's precompute a table: div_table[x][y] = True if x | y, for x, y in 2..256
    # This is 255*255 = 65025 entries, very fast
    div_table = [[False] * (N + 1) for _ in range(N + 1)]
    for x in range(2, N + 1):
        for y in range(x, N + 1, x):
            div_table[x][y] = True
    
    # For checking a | b*c:
    # We can use: a | b*c iff (a // gcd(a, b)) | c
    # Precompute gcd table? Or just compute on the fly.
    
    def is_valid_addition_fast(x, S_set, S_list):
        """
        Check if adding x to set S maintains the property.
        S_set is a set for O(1) lookup, S_list is a list for iteration.
        """
        n = len(S_list)
        
        # Case 1: x is the divisor. Check if x | b*c for any distinct b, c in S.
        if n >= 2:
            for i in range(n):
                b = S_list[i]
                for j in range(i + 1, n):
                    c = S_list[j]
                    if (b * c) % x == 0:
                        return False
        
        # Case 2: x is a multiplicand. For each a in S, check if a | x*c for some c in S, c != a.
        # a | x*c iff (a // gcd(a, x)) | c
        for a in S_list:
            g = gcd(a, x)
            req = a // g
            # Need c in S, c != a, such that c % req == 0
            # i.e., req | c
            if div_table[req][a]:
                # a is a multiple of req, but we need c != a
                # Check if there's another multiple of req in S
                found = False
                for c in S_list:
                    if c != a and div_table[req][c]:
                        found = True
                        break
                if found:
                    return False
            else:
                # a is not a multiple of req, so we just need any c in S (c != a) that is a multiple of req
                for c in S_list:
                    if c != a and div_table[req][c]:
                        return False
        
        return True

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    best_size = 0
    best_set = set()
    
    # Greedy construction with multiple orderings
    def greedy_construct(ordering):
        S = set()
        S_list = []
        for x in ordering:
            if is_valid_addition_fast(x, S, S_list):
                S.add(x)
                S_list.append(x)
        return S, S_list

    # Try different orderings
    orderings = []
    orderings.append(primes + composites)
    orderings.append(primes + sorted(composites, reverse=True))
    orderings.append(sorted(composites, key=lambda x: -x.bit_count()) + primes)
    orderings.append(sorted(composites, key=lambda x: x.bit_count()) + primes)
    
    # Add some random orderings
    for _ in range(10):
        rand_order = candidates[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for i, ordering in enumerate(orderings):
        if time.time() > deadline:
            break
        S, S_list = greedy_construct(ordering)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Local search: try to improve by swaps and additions
    current_set = set(best_set)
    current_list = list(current_set)
    current_size = len(current_set)
    
    time_limit_for_local = deadline - 3.0
    
    while time.time() < time_limit_for_local:
        improved = False
        
        # Try to add any element
        for x in candidates:
            if x not in current_set:
                if is_valid_addition_fast(x, current_set, current_list):
                    current_set.add(x)
                    current_list.append(x)
                    current_size += 1
                    improved = True
                    break
        
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            continue
        
        # No addition possible. Try swaps to find a configuration that allows addition.
        # Strategy: For each element u in current_set, temporarily remove it.
        # Then try to add elements that were previously blocked.
        # If we can add an element v after removing u, we have a same-size set.
        # Then try to add more.
        
        found_improvement = False
        for u in list(current_set):
            if time.time() > time_limit_for_local:
                break
            
            S_temp = current_set - {u}
            S_temp_list = [x for x in current_list if x != u]
            
            # Try to add elements that were blocked
            for v in candidates:
                if v in current_set and v != u:
                    continue
                if is_valid_addition_fast(v, S_temp, S_temp_list):
                    # We can add v after removing u
                    S_new = S_temp | {v}
                    S_new_list = S_temp_list + [v]
                    
                    # Now try to add more elements to S_new
                    added_any = False
                    for w in candidates:
                        if w in S_new:
                            continue
                        if is_valid_addition_fast(w, S_new, S_new_list):
                            S_new.add(w)
                            S_new_list.append(w)
                            added_any = True
                    
                    if added_any:
                        current_set = S_new
                        current_list = S_new_list
                        current_size = len(current_set)
                        found_improvement = True
                        break
            
            if found_improvement:
                break
        
        if found_improvement:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            continue
        
        # If no improvement found through the above, try random swaps to escape local optima
        if random.random() < 0.3:
            u = random.choice(list(current_set))
            v = random.choice([x for x in candidates if x not in current_set])
            
            S_temp = current_set - {u}
            S_temp_list = [x for x in current_list if x != u]
            
            if is_valid_addition_fast(v, S_temp, S_temp_list):
                current_set = S_temp | {v}
                current_list = S_temp_list + [v]
                
                # Try to add more
                for w in candidates:
                    if w not in current_set:
                        if is_valid_addition_fast(w, current_set, current_list):
                            current_set.add(w)
                            current_list.append(w)
                            current_size += 1
        
        # Check time
        if time.time() > time_limit_for_local:
            break
    
    # Final verification and cleanup
    result = list(best_set)
    return result

# EVOLVE-BLOCK-END
