"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def _is_valid_set(S):
    """Check if set S satisfies the no-mean-of-subset property."""
    n = len(S)
    S_sorted = sorted(S)
    for i in range(n):
        a = S_sorted[i]
        others = [x for x in S_sorted if x != a]
        m = len(others)
        # For each possible subset size k from 2 to m
        # Check if any subset of size k has sum == k*a
        # Use bitset DP for subset sums
        # But m can be up to 47, and sums up to 47*512 = 24064
        # This is expensive per element. Let's use a smarter approach.
        
        # Actually, let's check: for a to be the mean of a subset T of others,
        # sum(T) = len(T) * a
        # Let's do DP: dp[k] = set of achievable sums using k elements from others
        # But this is O(m^2 * max_sum) which is too slow for 47 elements.
        
        # Alternative: for small k (2,3,4), check directly.
        # For k=2: need x+y=2a, i.e., y=2a-x. Check if 2a-x is in others and !=x.
        # For k=3: need x+y+z=3a. 
        # For larger k, it's harder.
        
        # Let's use a different approach: 
        # The condition is that a is NOT in the convex hull (in the arithmetic sense)
        # of any subset of size >= 2.
        
        # Practical approach: check all subset sizes up to some limit, 
        # and for larger sizes use the fact that if a is the mean of a large subset,
        # it's likely also the mean of a smaller subset or we can use symmetry.
        
        # Actually, let's just do the full check with early termination.
        # For a set of size up to 48, checking all subsets is 2^47 which is infeasible.
        # We need the DP approach.
        
        # dp[s] = bitmask of subset sizes that can achieve sum s? No.
        # Let's do: for each k from 2 to m, use DP to see if sum=k*a is achievable.
        # DP for fixed k: standard knapsack-like.
        
        # Given the time constraints, let's use a heuristic: 
        # Check k=2,3,4 explicitly, and for k>=5, assume it's unlikely 
        # unless the set has special structure. This is NOT rigorous.
        
        # Better: use the fact that we're building incrementally.
        # When adding a new element, we only need to check:
        # 1. The new element is not the mean of any subset of existing elements
        # 2. No existing element is the mean of any subset that includes the new element
        
        # This is still complex. Let's just do a full check with bitsets.
        
        # For a given target sum T = k*a, check if T is achievable by exactly k elements from others.
        # Use bitset: dp[i] = bitset where bit j is set if sum j is achievable with i elements.
        # But this is O(m * max_sum) per k, and we have m-1 values of k.
        
        # Given the 90s budget, let's limit the check to k up to 8 or so,
        # and for the final verification, do a more thorough check.
        
        # Actually, let's just do a straightforward check for small sets
        # and rely on the randomized construction to avoid obvious violations.
        
        # For now, let's check k=2 and k=3 thoroughly, and k=4,5 with sampling.
        
        # k=2: a is mean of x,y iff x+y=2a
        for x in others:
            y = 2*a - x
            if y != x and y in set(others):
                return False
        
        # k=3: a is mean of x,y,z iff x+y+z=3a
        others_set = set(others)
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                x, y = others[i], others[j]
                z = 3*a - x - y
                if z != x and z != y and z in others_set:
                    return False
        
        # k=4: a is mean of x,y,z,w iff x+y+z+w=4a
        # This is O(m^3) which for m=47 is ~160k iterations, acceptable
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                for k in range(j+1, len(others)):
                    x, y, z = others[i], others[j], others[k]
                    w = 4*a - x - y - z
                    if w != x and w != y and w != z and w in others_set:
                        return False
        
        # For k>=5, we'll skip in the incremental check but verify at the end
    return True

def _full_verify(S):
    """Full verification using DP for all subset sizes."""
    n = len(S)
    S_set = set(S)
    for a in S:
        others = [x for x in S if x != a]
        m = len(others)
        others_set = set(others)
        max_sum = sum(others)
        
        # For each k from 2 to m, check if k*a is achievable as sum of exactly k elements
        # Use DP: dp[k][s] = True if sum s achievable with k elements
        # To save memory, use a list of sets or bitsets
        
        # Bitset approach: dp[k] is a bitset (Python int) where bit s is set if achievable
        # dp[0] = 1 (bit 0 set)
        # dp[k] = OR over each element e in others of (dp[k-1] << e)
        
        # But we need exactly k elements, so we do:
        # dp[k] starts as 0
        # For each e in others: dp[k] |= (dp[k-1] << e) but this overcounts
        
        # Standard approach: iterate elements, update dp in reverse order of k
        # dp[k] = set of sums achievable with k elements
        
        # For efficiency with large sums, use a dict or set
        # dp[k] = set of achievable sums with k elements
        
        dp = [set() for _ in range(m+1)]
        dp[0] = {0}
        for e in others:
            for k in range(m-1, -1, -1):
                for s in list(dp[k]):
                    dp[k+1].add(s + e)
        
        for k in range(2, m+1):
            if k*a in dp[k]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    # Strategy: randomized greedy with multiple restarts
    # Order candidates in random order, add if valid
    # Use fast incremental check (k=2,3,4) for speed
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        current = []
        current_set = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if len(current) >= 48:
                break
            
            # Fast check: is x the mean of any subset of current?
            # And does adding x make any existing element the mean of a subset including x?
            
            # Check 1: x is not mean of any subset of current (size >= 2)
            # Check 2: for each a in current, a is not mean of any subset of (current - {a}) union {x} of size >= 2
            
            # For speed, do quick checks first
            valid = True
            
            # Quick check: x not mean of any pair in current
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if (current[i] + current[j]) == 2*x:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Quick check: no a in current is mean of {x, b} for b in current, b != a
                for a in current:
                    # a = (x + b)/2 => b = 2a - x
                    b = 2*a - x
                    if b != a and b != x and b in current_set:
                        valid = False
                        break
            
            if valid:
                # Check k=3: x not mean of any triple in current
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        for k in range(j+1, len(current)):
                            if current[i] + current[j] + current[k] == 3*x:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
            
            if valid:
                # Check k=3: no a in current is mean of {x, b, c}
                # This is O(n^2) per a, total O(n^3) which is too slow for n=47
                # Skip this for speed, rely on final verification
                
                if valid:
                    current.append(x)
                    current_set.add(x)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = current[:]
        
        # Try to improve best_set with local search
        if len(best_set) >= 40:
            # Try adding elements from 1..512 not in best_set
            if time.time() > deadline:
                break
            
            for x in range(1, 513):
                if time.time() > deadline:
                    break
                if x in best_set:
                    continue
                if len(best_set) >= 48:
                    break
                
                # Quick validity check
                valid = True
                for i in range(len(best_set)):
                    for j in range(i+1, len(best_set)):
                        if best_set[i] + best_set[j] == 2*x:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    for a in best_set:
                        b = 2*a - x
                        if b != a and b != x and b in best_set:
                            valid = False
                            break
                
                if valid:
                    # Do a more thorough check
                    test_set = best_set + [x]
                    if _is_valid_set(test_set):
                        best_set.append(x)
        
        if best_len >= 48:
            break
    
    # Final verification of best_set
    if best_set and _full_verify(best_set):
        return best_set
    else:
        # If full verify fails, return the largest valid subset we can find
        # Try removing elements one by one
        if best_set:
            while best_set and not _full_verify(best_set):
                # Remove the element that causes the most issues
                # For simplicity, remove the last one
                best_set.pop()
            return best_set
        return []

# EVOLVE-BLOCK-END
