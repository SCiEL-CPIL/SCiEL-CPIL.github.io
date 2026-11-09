import time
import random

def construct():
    n = 256
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    
    # Precompute conflict relationships
    # conflict[x] is a bitmask where bit y is set if x and y conflict
    # Two elements x, y conflict if there exists a valid triple where they participate
    # and the divisibility condition is violated.
    
    # Actually, let's think about this differently.
    # For a set S, the constraint is: for all a < b <= c in S, a does NOT divide (b+c).
    # 
    # When adding element x to set S, we need to check:
    # 1. x as 'a': for all b <= c in S with x < b, x does not divide (b+c)
    # 2. x as 'b': for all a in S with a < x, and all c in S with c >= x, a does not divide (x+c)
    # 3. x as 'c': for all a < b in S with b < x, a does not divide (b+x)
    #    Also for all a in S with a < x, a does not divide (x+x) = 2x (when b=c=x)
    
    # Let's precompute for each pair (a, b) with a < b, whether a divides 2b
    # And for each triple (a, b, c) with a < b <= c, whether a divides (b+c)
    
    # For efficiency, let's precompute:
    # divides[a][b] = True if a divides b
    # Then for the constraint: a divides (b+c) means (b+c) % a == 0
    
    # Let's precompute a conflict matrix:
    # For each element x, precompute which other elements y "conflict" with x
    # in the sense that having both in the set could violate the constraint.
    
    # Actually, the constraint involves triples, so pairwise conflicts aren't sufficient.
    # But we can still speed up the can_add check significantly.
    
    # Precompute: for each a, list of (b, c) pairs where a < b <= c and a | (b+c)
    # This tells us which pairs of larger elements are "bad" for a given smallest element.
    
    # Precompute divisors: divs[x] = list of divisors of x in [1, n]
    divs = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for multiple in range(a, n + 1, a):
            divs[multiple].append(a)
    
    # Precompute: for each a, bad_pairs[a] = set of (b, c) with a < b <= c <= n and a | (b+c)
    # This is too large. Instead, let's precompute:
    # For each a and b > a, the set of c >= b such that a | (b+c)
    # i.e., c ≡ -b (mod a), c >= b
    
    # For the can_add check, let's use a different approach:
    # Maintain the set as a sorted list and use early termination.
    
    k = 171
    base_tail = set(range(k, n + 1))
    
    def can_add(x, S):
        """Check if x can be added to set S while maintaining the property.
        S is a set of integers."""
        S_list = sorted(S)
        
        # Case 1: x is the smallest element (a=x)
        # Need x < b <= c in S, and x does NOT divide (b+c)
        # i.e., for all b <= c in S with b > x, (b+c) % x != 0
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for c in S_list[i:]:
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element (b=x)
        # Need a < x <= c in S, and a does NOT divide (x+c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element (c=x)
        # Subcase 3a: b = x, so a < x and a does NOT divide 2x
        for a in S_list:
            if a >= x:
                break
            if (2 * x) % a == 0:
                return False
        
        # Subcase 3b: b < x, so a < b < x, check a does NOT divide (b+x)
        for i, b in enumerate(S_list):
            if b >= x:
                break
            for a in S_list[:i]:
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def is_valid_set(S):
        """Verify that set S satisfies the property for all triples."""
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                for k_idx in range(j, len(S_sorted)):
                    c = S_sorted[k_idx]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Strategy: Start with the safe tail and greedily add elements.
    # Key insight: elements that are prime or have few divisors create fewer conflicts.
    
    best_set = None
    best_size = 0
    
    # Precompute number of divisors for prioritization
    num_divs = [0] * (n + 1)
    for i in range(1, n + 1):
        num_divs[i] = len(divs[i])
    
    strategies = []
    
    # Strategy 1: Descending (largest first)
    strategies.append(('desc', list(range(170, 0, -1))))
    
    # Strategy 2: Ascending
    strategies.append(('asc', list(range(1, 171))))
    
    # Strategy 3: By fewest divisors first (prime-like elements first)
    candidates = list(range(1, 171))
    candidates.sort(key=lambda x: (num_divs[x], x))
    strategies.append(('fewdiv', candidates))
    
    # Strategy 4: Random
    random.seed(123)
    candidates = list(range(1, 171))
    random.shuffle(candidates)
    strategies.append(('rand1', candidates))
    
    # Strategy 5: Another random
    random.seed(456)
    candidates = list(range(1, 171))
    random.shuffle(candidates)
    strategies.append(('rand2', candidates))
    
    # Strategy 6: Primes first
    def is_prime(m):
        if m < 2:
            return False
        for p in range(2, int(m**0.5) + 1):
            if m % p == 0:
                return False
        return True
    
    primes = [x for x in range(1, 171) if is_prime(x)]
    composites = [x for x in range(1, 171) if not is_prime(x)]
    strategies.append(('prime', primes + composites))
    
    for name, candidates in strategies:
        if time.time() > deadline:
            break
        
        current_set = set(base_tail)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if can_add(x, current_set):
                current_set.add(x)
        
        # Local search: try swaps
        if time.time() < deadline - 10.0:
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                current_list = sorted(current_set)
                
                for i, remove_x in enumerate(current_list):
                    if time.time() > deadline - 3.0:
                        break
                    trial_set = set(current_set)
                    trial_set.discard(remove_x)
                    
                    found = False
                    for add_x in range(1, n + 1):
                        if add_x in trial_set:
                            continue
                        if can_add(add_x, trial_set):
                            new_set = trial_set | {add_x}
                            if len(new_set) > len(current_set):
                                current_set = new_set
                                improved = True
                                found = True
                                break
                            elif len(new_set) == len(current_set):
                                # Same size - prefer if it allows more future additions
                                # Quick heuristic: count how many elements can be added
                                addable_new = 0
                                addable_cur = 0
                                for test_x in range(1, n + 1):
                                    if test_x not in new_set and can_add(test_x, new_set):
                                        addable_new += 1
                                        if addable_new >= 5:
                                            break
                                    if test_x not in current_set and can_add(test_x, current_set):
                                        addable_cur += 1
                                        if addable_cur >= 5:
                                            break
                                if addable_new > addable_cur:
                                    current_set = new_set
                                    improved = True
                                    found = True
                                    break
                    
                    if found:
                        break
        
        if best_set is None or len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # If we have time, do more aggressive local search on the best set
    if best_set is not None and time.time() < deadline - 15.0:
        current_set = set(best_set)
        improved = True
        iterations = 0
        max_iterations = 500
        
        while improved and iterations < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            current_list = sorted(current_set)
            
            for i, remove_x in enumerate(current_list):
                if time.time() > deadline - 5.0:
                    break
                trial_set = set(current_set)
                trial_set.discard(remove_x)
                
                for add_x in range(1, n + 1):
                    if add_x in trial_set:
                        continue
                    if can_add(add_x, trial_set):
                        new_set = trial_set | {add_x}
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            improved = True
                            break
                        elif len(new_set) == len(current_set):
                            # Try to find a swap that leads to a better structure
                            # Check if removing a different element from new_set allows adding more
                            pass
                if improved:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Final validation
    if best_set is None or not is_valid_set(best_set):
        return sorted(list(base_tail))
    
    return sorted(list(best_set))
