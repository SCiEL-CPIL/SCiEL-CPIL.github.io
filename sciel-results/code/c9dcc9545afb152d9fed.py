import random
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    # Precompute LCM of 1..24
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # Precompute scaled reciprocals
    # val[i] = L // i
    val = [0] * 25
    for i in range(1, 25):
        val[i] = L // i

    def is_subsum_free(S):
        """
        Checks if S is subsum-free.
        S is a set of integers.
        """
        if not S:
            return True
        S_list = list(S)
        n = len(S_list)
        # For each x in S, check if 1/x is sum of subset of 1/y for y in S\{x}
        for i in range(n):
            x = S_list[i]
            target = val[x]
            # Get weights of others
            others = [val[y] for y in S_list if y != x]
            # Check if target is a subset sum of others
            # Since n is small (<=24), and we check subsets of size up to 23,
            # but typically we expect S to be around 16, so others size ~15.
            # 2^15 is 32768.
            # We can use a set of reachable sums.
            # Optimization: if target > sum(others), return False immediately for this x?
            # Actually if target > sum(others), it's not a subset sum.
            sum_others = sum(others)
            if target > sum_others:
                continue
            
            # Subset sum check
            # Use a set to track reachable sums
            # Start with {0}
            reachable = {0}
            for w in others:
                # We can prune sums that are > target
                new_reachable = set()
                for s in reachable:
                    if s <= target:
                        new_reachable.add(s)
                        if s + w <= target:
                            new_reachable.add(s + w)
                reachable = new_reachable
                if target in reachable:
                    return False
            # If loop finishes, target not found
            # Note: if target is not in reachable, it's fine for this x.
            
        return True

    best_set = set()
    
    # Try to find a set of size 16
    # We can try several random permutations
    # Or a specific heuristic order
    
    # Heuristic 1: Random
    for _ in range(500):
        if time.time() > deadline:
            break
        perm = list(range(1, 25))
        random.shuffle(perm)
        S = set()
        for x in perm:
            # Check if adding x keeps it valid
            # Optimization: only check constraints involving x?
            # No, adding x might create a new subsum for existing elements?
            # If 1/y = sum(1/z for z in S), and we add x, 
            # does 1/y = sum(1/z for z in S \cup {x})?
            # The RHS increases, so if it didn't equal before, it might now?
            # Wait. 1/y is fixed. The subset sum is over OTHERS.
            # If we add x to S, for an existing y, the set of others becomes (S \ {y}) \cup {x}.
            # So we might form a new sum.
            # So we must re-check the whole set.
            # However, we can optimize:
            # 1. Check if 1/x is sum of subset of S.
            # 2. Check if for any y in S, 1/y is sum of subset of (S \ {y}) \cup {x}.
            #    This is equivalent to: 1/y is sum of subset of S \ {y} (already false)
            #    OR 1/y is sum of subset of (S \ {y}) \cup {x} that includes x.
            #    If it includes x, then 1/y - 1/x must be sum of subset of S \ {y, x}.
            
            # Let's implement this optimized check
            if not S:
                S.add(x)
                continue
            
            # Check 1: Is 1/x a subset sum of S?
            target_x = val[x]
            others_x = [val[y] for y in S]
            if target_x <= sum(others_x):
                reachable = {0}
                found = False
                for w in others_x:
                    new_reachable = set()
                    for s in reachable:
                        if s <= target_x:
                            new_reachable.add(s)
                            if s + w <= target_x:
                                new_reachable.add(s + w)
                    reachable = new_reachable
                    if target_x in reachable:
                        found = True
                        break
                if found:
                    continue
            
            # Check 2: For each y in S, is 1/y a subset sum of (S \ {y}) \cup {x}?
            # This happens if:
            # a) 1/y is subset sum of S \ {y} (False, since S was valid)
            # b) 1/y is subset sum of (S \ {y}) \cup {x} using x.
            #    This means 1/y - 1/x is a subset sum of S \ {y, x}.
            
            valid_add = True
            for y in S:
                target_diff = val[y] - val[x]
                if target_diff < 0:
                    continue
                # Check if target_diff is subset sum of S \ {y, x}
                others_yx = [val[z] for z in S if z != y and z != x]
                if target_diff > sum(others_yx):
                    continue
                if target_diff == 0:
                    # 1/y = 1/x. Since elements are distinct, this is impossible.
                    # But if it were, empty subset sum is 0.
                    # So if val[y] == val[x], then 1/y - 1/x = 0.
                    # Is 0 a subset sum of others? Yes (empty set).
                    # So if val[y] == val[x], it would be invalid.
                    # But val[y] = L/y, distinct y -> distinct val[y].
                    # So target_diff != 0 for distinct y, x.
                    pass
                
                reachable = {0}
                found = False
                for w in others_yx:
                    new_reachable = set()
                    for s in reachable:
                        if s <= target_diff:
                            new_reachable.add(s)
                            if s + w <= target_diff:
                                new_reachable.add(s + w)
                    reachable = new_reachable
                    if target_diff in reachable:
                        found = True
                        break
                if found:
                    valid_add = False
                    break
            
            if valid_add:
                S.add(x)
                
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) >= 16:
                # We can stop early if we found a large set, but maybe there's a bigger one?
                # Max possible is 24.
                # Let's continue a bit to see if we can get 17, 18...
                # But 16 is a good target.
                pass

    # Fallback: if random didn't find good enough, try specific orders
    if len(best_set) < 16:
        # Try descending order
        S = set()
        for x in range(24, 0, -1):
            # Same optimized check logic
            if not S:
                S.add(x)
                continue
            # ... (reuse logic or just call is_subsum_free on S U {x})
            # For simplicity and safety, just use the full check if S is small
            # But optimized is better.
            # Let's just use the full is_subsum_free for the fallback as it's less code
            if is_subsum_free(S.union([x])):
                S.add(x)
        if len(S) > len(best_set):
            best_set = S
            
    if not best_set:
        # Should not happen
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END
