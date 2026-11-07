import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill
    
    def get_subset_sums(S):
        """Get all non-empty subset sums of S."""
        sums = set()
        n = len(S)
        # Use iterative approach
        current_sums = set()
        for i in range(n):
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + S[i])
            new_sums.add(S[i])
            current_sums |= new_sums
        return current_sums
    
    def check_divisibility_conflict(sums):
        """Check if any two distinct sums in the set divide each other."""
        sums_list = sorted(sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            # a divides b for some b > a
            for j in range(i + 1, n):
                b = sums_list[j]
                if b > a * 2:  # optimization: if b > 2*a, a can't divide b unless... wait, a can divide b if b is a multiple of a
                    # Actually we need to check all
                    break  # This optimization is WRONG. Remove it.
                if b % a == 0:
                    return True
        return False
    
    def is_valid(S):
        """Check if all non-empty subset sums are divisibility-incomparable."""
        if not S:
            return True
        sums = get_subset_sums(S)
        sums_list = sorted(sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b % a == 0:
                    return False
        return True
    
    def can_add(S, x, existing_sums):
        """Check if we can add x to S, given existing_sums = get_subset_sums(S)."""
        # New sums: x, and s+x for each s in existing_sums
        new_sums = set(existing_sums)
        new_sums.add(x)
        for s in existing_sums:
            new_sums.add(s + x)
        
        # Check divisibility conflicts in new_sums
        sums_list = sorted(new_sums)
        n = len(sums_list)
        for i in range(n):
            a = sums_list[i]
            for j in range(i + 1, n):
                b = sums_list[j]
                if b % a == 0:
                    return False, None
        return True, new_sums
    
    # Candidate pool: focus on larger numbers to reduce divisibility issues
    # Numbers that are "hard to divide" - primes and near-primes in the upper range
    candidates = list(range(129, 257))  # 129 to 256
    
    best_set = []
    best_size = 0
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Greedy descending
    strategies.append(sorted(candidates, reverse=True))
    
    # Strategy 2: Greedy ascending from high end
    strategies.append(sorted(candidates, key=lambda x: -x))
    
    # Strategy 3: Random orderings
    for _ in range(20):
        c = candidates[:]
        random.shuffle(c)
        strategies.append(c)
    
    # Strategy 4: Focus on primes and numbers with few divisors
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
    
    primes_in_range = [x for x in candidates if is_prime(x)]
    strategies.append(primes_in_range)
    strategies.append(sorted(primes_in_range, reverse=True))
    
    # Also try a mix: large numbers with "awkward" divisibility
    # Numbers that are 1 mod small primes
    awkward = [x for x in candidates if x % 2 == 1 and x % 3 == 1 and x % 5 == 1]
    if awkward:
        strategies.append(sorted(awkward, reverse=True))
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        current_set = []
        current_sums = set()
        
        for x in order:
            if time.time() > deadline:
                break
            ok, new_sums = can_add(current_set, x, current_sums)
            if ok:
                current_set.append(x)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Local search: try to swap elements to potentially grow the set
    # Try removing one element and adding a different one
    if best_set and time.time() < deadline:
        current_set = best_set[:]
        current_sums = get_subset_sums(current_set)
        used = set(current_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                # Try removing current_set[i] and adding some other element
                removed = current_set[i]
                temp_set = current_set[:i] + current_set[i+1:]
                temp_sums = get_subset_sums(temp_set) if temp_set else set()
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if x in used:
                        continue
                    if x == removed:
                        continue
                    ok, new_sums = can_add(temp_set, x, temp_sums)
                    if ok:
                        # Check if this gives us a larger set (it won't, same size)
                        # But maybe it enables future growth - skip for now
                        pass
                
                # Try adding without removing (if we can)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if x in used:
                        continue
                    ok, new_sums = can_add(current_set, x, current_sums)
                    if ok:
                        current_set.append(x)
                        current_sums = new_sums
                        used.add(x)
                        improved = True
                        break
    
    # Final local search: try all pairs of swaps
    if best_set and time.time() < deadline:
        current_set = best_set[:]
        current_sums = get_subset_sums(current_set)
        used = set(current_set)
        
        for _ in range(5):  # multiple passes
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                removed = current_set[i]
                temp_set = current_set[:i] + current_set[i+1:]
                temp_sums = get_subset_sums(temp_set) if temp_set else set()
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if x in used:
                        continue
                    if x == removed:
                        continue
                    ok, new_sums = can_add(temp_set, x, temp_sums)
                    if ok:
                        # Try this swap - does it allow adding more?
                        new_set = temp_set + [x]
                        new_used = used - {removed} | {x}
                        
                        # Try to extend
                        extended = new_set[:]
                        ext_sums = new_sums
                        for y in candidates:
                            if y in new_used:
                                continue
                            ok2, ext2 = can_add(extended, y, ext_sums)
                            if ok2:
                                extended.append(y)
                                ext_sums = ext2
                                new_used.add(y)
                        
                        if len(extended) > len(current_set):
                            current_set = extended
                            current_sums = ext_sums
                            used = new_used
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(best_set)
    
    # Ensure we return at least something valid
    if not best_set or not is_valid(best_set):
        # Fallback: find a small valid set
        for size in range(30, 2, -1):
            if time.time() > deadline:
                break
            # Just return what we have if valid
            break
    
    # Final validation
    if best_set and is_valid(best_set):
        result = list(best_set)
        # Ensure all are Python int
        result = [int(x) for x in result]
        return result
    else:
        # Minimal valid set
        return [31, 32, 34]

# EVOLVE-BLOCK-END
