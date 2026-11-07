"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256, 'maximum_size': 30}

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    # We need a set of distinct integers in 1..256 such that no two distinct
    # nonempty subset sums divide one another.
    #
    # Key insight: if we use a superincreasing sequence (each element > sum of all
    # previous), then all subset sums are unique and well-separated. But we still
    # need to check divisibility.
    #
    # Better approach: use elements that are "large" relative to each other so that
    # subset sums don't land on multiples of each other.
    #
    # Strategy: 
    # 1. Start with a good seed set (found via careful construction)
    # 2. Use backtracking/greedy to try to extend
    # 3. Use randomized restarts with different orderings
    #
    # Critical optimization: incremental subset sum tracking and fast divisibility check.
    # Instead of O(n^2) pair check, for each new sum s, check if any existing sum t 
    # divides s or s divides t. We can precompute divisors.

    # Precompute: for each value v in 1..256*30=7680, store its divisors
    # Actually, max subset sum for 30 elements each <= 256 is 30*256 = 7680
    MAX_SUM = 30 * 256  # 7680
    
    # For fast divisibility: given a set of sums, when adding new sum s,
    # check: (1) does any existing sum divide s? (2) does s divide any existing sum?
    # We can maintain a sorted list and use the fact that if s is in the set,
    # we need to check s/k for integer k (divisors of s) and s*k (multiples of s).
    
    # Better: maintain the set of sums. When adding new sums, for each new sum ns:
    #   - Check if any existing sum es divides ns: i.e., ns % es == 0
    #   - Check if ns divides any existing sum es: i.e., es % ns == 0
    # To speed this up, we can keep the sums in a set and for divisibility check,
    # iterate over the smaller set.
    
    # Even better: precompute for each possible sum value its divisors that are
    # also in our range. But the set changes dynamically.
    
    # Let's use a different approach: maintain sorted list of sums.
    # When checking if adding element e creates conflicts:
    #   new_sums = existing_sums ∪ {s + e for s in existing_sums} ∪ {e}
    #   Then check all pairs in new_sums for divisibility.
    # This is O(n^2) per addition where n = number of subset sums.
    # For a set of size k, there are at most 2^k - 1 subset sums.
    # For k=10, that's 1023 sums, so ~500K pair checks per addition. Too slow for many restarts.
    
    # Optimized divisibility check: 
    # Given a set S of positive integers, check no two distinct elements divide each other.
    # Sort S. For each s in S (in order), check if any multiple of s (other than s) is in S.
    # Use a boolean array or set for membership.
    
    def is_divisibility_free(sums_set):
        """Check that no two distinct elements in sums_set divide each other."""
        if len(sums_set) <= 1:
            return True
        sums_list = sorted(sums_set)
        # For each sum, check its multiples
        # Use a set for O(1) lookup
        s_set = sums_set
        for s in sums_list:
            # Check multiples: 2s, 3s, ... up to max(sums_list)
            max_val = sums_list[-1]
            multiple = 2 * s
            while multiple <= max_val:
                if multiple in s_set:
                    return False
                multiple += s
        return True
    
    # Greedy construction with backtracking
    # Start with empty set, try to add elements one by one
    
    def try_greedy(candidate_order, current_sums, current_set, target_size):
        """Try to build a set of target_size using candidate_order."""
        if len(current_set) >= target_size:
            return current_set[:]
        
        for i, c in enumerate(candidate_order):
            if c in current_set:
                continue
            
            # Compute new sums
            new_sums = set(current_sums)
            new_sums.add(c)
            for s in current_sums:
                new_sums.add(s + c)
            
            # Check divisibility
            if is_divisibility_free(new_sums):
                new_set = current_set | {c}
                result = try_greedy(candidate_order[i+1:], new_sums, new_set, target_size)
                if result is not None and len(result) >= target_size:
                    return result
                # Even if we didn't reach target, return what we got
                if result is not None and len(result) > len(current_set):
                    return result
        
        return current_set[:] if current_set else None
    
    # This recursive approach is too slow. Let's use iterative greedy with randomization.
    
    def greedy_build(order, max_size=30):
        """Greedily add elements from order if they maintain the property."""
        current_set = set()
        current_sums = set()
        
        for c in order:
            if len(current_set) >= max_size:
                break
            if c in current_set:
                continue
            
            new_sums = set(current_sums)
            new_sums.add(c)
            for s in current_sums:
                new_sums.add(s + c)
            
            if is_divisibility_free(new_sums):
                current_set.add(c)
                current_sums = new_sums
        
        return list(current_set)
    
    # Strategy: 
    # 1. Try several different orderings (random, sorted descending, sorted ascending, etc.)
    # 2. Use a good initial seed
    # 3. Local search: try swapping elements
    
    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    # Try different strategies
    strategies = []
    
    # Strategy 1: Descending order (large numbers first)
    strategies.append(sorted(candidates, reverse=True))
    
    # Strategy 2: Ascending order
    strategies.append(sorted(candidates))
    
    # Strategy 3: Random orderings
    rng = random.Random(42)
    for _ in range(50):
        order = candidates[:]
        rng.shuffle(order)
        strategies.append(order)
    
    # Strategy 4: Weighted random (favor larger numbers)
    for _ in range(50):
        order = sorted(candidates, key=lambda x: -rng.random() * x)
        strategies.append(order)
    
    # Strategy 5: Primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, 257) if is_prime(p)]
    non_primes = [p for p in range(1, 257) if not is_prime(p)]
    strategies.append(primes + non_primes)
    strategies.append(non_primes + primes)
    
    # Run greedy for each strategy
    for order in strategies:
        if time.time() > deadline:
            break
        
        result = greedy_build(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Local search: try to improve by swapping or adding
    # For the best set found, try to add more elements
    if best_set:
        best_set_set = set(best_set)
        # Compute current sums
        best_sums = set()
        for mask in range(1, 1 << len(best_set)):
            s = 0
            for i in range(len(best_set)):
                if mask & (1 << i):
                    s += best_set[i]
            best_sums.add(s)
        
        # Try adding each remaining candidate
        remaining = [c for c in candidates if c not in best_set_set]
        for c in remaining:
            if time.time() > deadline:
                break
            if len(best_set) >= 30:
                break
            
            new_sums = set(best_sums)
            new_sums.add(c)
            for s in best_sums:
                new_sums.add(s + c)
            
            if is_divisibility_free(new_sums):
                best_set.append(c)
                best_set_set.add(c)
                best_sums = new_sums
                best_size = len(best_set)
    
    # More aggressive: try multiple random restarts with different seeds
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random ordering with bias toward larger numbers
        order = candidates[:]
        rng.shuffle(order)
        
        result = greedy_build(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Final local search: try to swap elements to free up room for more
    if best_set and len(best_set) < 30:
        best_set_set = set(best_set)
        
        # Compute current sums
        best_sums = set()
        for mask in range(1, 1 << len(best_set)):
            s = 0
            for i in range(len(best_set)):
                if mask & (1 << i):
                    s += best_set[i]
            best_sums.add(s)
        
        # Try removing one element and adding two others
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            
            removed = best_set[i]
            remaining_set = best_set_set - {removed}
            remaining_list = list(remaining_set)
            
            # Compute sums without removed
            rem_sums = set()
            for mask in range(1, 1 << len(remaining_list)):
                s = 0
                for j in range(len(remaining_list)):
                    if mask & (1 << j):
                        s += remaining_list[j]
                rem_sums.add(s)
            
            # Try to add two new elements
            candidates_to_add = [c for c in candidates if c not in remaining_set and c != removed]
            
            # First try adding one
            for c1 in candidates_to_add:
                new_sums1 = set(rem_sums)
                new_sums1.add(c1)
                for s in rem_sums:
                    new_sums1.add(s + c1)
                
                if is_divisibility_free(new_sums1):
                    # Now try adding a second
                    for c2 in candidates_to_add:
                        if c2 == c1:
                            continue
                        new_sums2 = set(new_sums1)
                        new_sums2.add(c2)
                        for s in new_sums1:
                            new_sums2.add(s + c2)
                        
                        if is_divisibility_free(new_sums2):
                            # We can replace removed with c1 and c2
                            new_best = remaining_set | {c1, c2}
                            if len(new_best) > len(best_set):
                                best_set = list(new_best)
                                best_set_set = new_best
                                best_size = len(best_set)
                                break
                if time.time() > deadline:
                    break
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in best_set]
    
    # Final validation
    if result:
        # Check all subset sums
        all_sums = set()
        n = len(result)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += result[i]
            all_sums.add(s)
        
        if not is_divisibility_free(all_sums):
            # Fallback: return empty or a known valid small set
            # Try a simple valid set
            fallback = [1, 2, 4, 8, 16, 32, 64, 128]
            # Check fallback
            fs = set()
            for mask in range(1, 1 << len(fallback)):
                s = 0
                for i in range(len(fallback)):
                    if mask & (1 << i):
                        s += fallback[i]
                fs.add(s)
            if is_divisibility_free(fs):
                result = fallback
            else:
                result = [1]
    
    return result
# EVOLVE-BLOCK-END
