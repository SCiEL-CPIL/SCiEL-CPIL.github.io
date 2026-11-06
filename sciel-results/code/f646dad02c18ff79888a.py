import time
from math import gcd
from functools import reduce

_START_TIME = None
_TIME_LIMIT = 85.0

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) > _TIME_LIMIT

# Precompute LCM and scaled reciprocals
def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(_lcm, range(1, 25))
# V[i] is the scaled reciprocal of i (1/i * LCM)
_V = {i: _LCM // i for i in range(1, 25)}

def _greedy_initial():
    # Try to build a large set greedily
    # Heuristic: Add elements that cause fewest conflicts?
    # Or just try descending order
    best = []
    # Try a few orders
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        list(range(12, 0, -1)) + list(range(24, 12, -1))
    ]
    for order in orders:
        S = []
        Sigma = {0}
        Sigma_excl = {} # val: set
        # We need to maintain Sigma_excl correctly
        # Let's just use a simple check for greedy to keep it fast, or reuse the logic
        for x in order:
            if _time_exceeded():
                break
            # Check if x can be added
            # 1. V[x] not in Sigma
            if _V[x] in Sigma:
                continue
            # 2. For y in S, V[y] - V[x] not in Sigma_excl[y]
            valid = True
            for y in S:
                if _V[y] - _V[x] in Sigma_excl[y]:
                    valid = False
                    break
            
            if valid:
                # Update S, Sigma, Sigma_excl
                S.append(x)
                # Update Sigma
                new_sums = {s + _V[x] for s in Sigma}
                Sigma = Sigma.union(new_sums)
                
                # Update Sigma_excl
                new_excl = {}
                # For existing y in S (excluding x, which is new)
                for y in S:
                    if y == x:
                        new_excl[y] = Sigma # Sigma BEFORE adding x? No, Sigma_excl[x] should be Sigma of S\{x} which is the old Sigma
                        # Wait, logic:
                        # Sigma_excl[x] for the NEW set S' = S \cup {x} is the set of sums of S.
                        # So new_excl[x] = old Sigma.
                        pass # Handled below
                    else:
                        # y is in old S
                        # New Sigma_excl[y] = Old Sigma_excl[y] U {s + V[x] for s in Old Sigma_excl[y]}
                        old_set = Sigma_excl[y]
                        new_set = old_set.union({s + _V[x] for s in old_set})
                        new_excl[y] = new_set
                
                # Handle x
                new_excl[x] = Sigma # This is the Sigma of S (old)
                
                Sigma_excl = new_excl
                
        if len(S) > len(best):
            best = S[:]
            
    return best

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = _greedy_initial()
    best_size = len(best_set)
    
    # Backtracking
    # State: S (list), Sigma (set), Sigma_excl (dict)
    # To save memory and time, we might pass these by reference and backtrack, or copy them.
    # Copying sets of size 2^k is expensive.
    # Let's use a recursive function that copies the sets.
    # Since max size is ~15, 2^15 = 32768. Copying 15 sets of 32768 ints is ~500k ops.
    # This might be too slow if we do it at every node.
    
    # Alternative: Use a "persistent" data structure or just recompute?
    # Recomputing Sigma_excl from scratch for a given S is O(|S| * 2^{|S|}).
    # If we do this at every node, it's expensive.
    
    # Let's stick to copying but optimize the check.
    # Actually, we can just check validity using MIM on the fly for the "add x" step,
    # and only maintain Sigma for the "1/x in Sigma(S)" check which is O(1).
    # The expensive part was checking "V[y] - V[x] in Sigma(S \ {y})".
    # If we don't maintain Sigma_excl, we have to do a subset sum check for each y.
    # That is |S| MIM checks.
    # MIM is 2^(|S|/2).
    # If |S|=10, 2^5=32. 10 checks * 32 = 320 ops.
    # If |S|=15, 2^7.5 ~ 180. 15 checks * 180 = 2700 ops.
    # This is much faster than copying 500k ints!
    
    # So:
    # 1. Maintain Sigma (set of sums) for O(1) check of 1/x.
    # 2. For the second condition, perform MIM subset sum checks on the list of V[y] for y in S, y != current_y.
    
    def _subset_sum_mim(target, items):
        if target < 0:
            return False
        if target == 0:
            return True
        n = len(items)
        if n == 0:
            return False
        if n <= 10:
            # Brute force
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += items[i]
                if s == target:
                    return True
            return False
            
        half = n // 2
        left = items[:half]
        right = items[half:]
        
        left_sums = {0}
        for i in left:
            left_sums.update({s + i for s in list(left_sums)})
            
        right_sums = {0}
        for i in right:
            right_sums.update({s + i for s in list(right_sums)})
            
        for l in left_sums:
            if target - l in right_sums:
                return True
        return False

    def search(i, S, Sigma):
        nonlocal best_set, best_size
        if _time_exceeded():
            return
            
        if len(S) + (24 - i) <= best_size:
            return
            
        if i == 24:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        x = i + 1
        
        # Option 1: Exclude x
        search(i + 1, S, Sigma)
        
        # Option 2: Include x
        # Check 1: V[x] not in Sigma
        if _V[x] in Sigma:
            return # Prune, cannot add x
            
        # Check 2: For all y in S, V[y] - V[x] not in SubsetSums(S \ {y})
        can_add = True
        for y in S:
            target = _V[y] - _V[x]
            if target < 0:
                continue # Sum of positive numbers cannot be negative
            # Items are V[z] for z in S, z != y
            items = [_V[z] for z in S if z != y]
            if _subset_sum_mim(target, items):
                can_add = False
                break
                
        if can_add:
            # Update Sigma
            new_Sigma = Sigma.union({s + _V[x] for s in Sigma})
            S.append(x)
            search(i + 1, S, new_Sigma)
            S.pop()

    search(0, [], {0})
    
    if not best_set:
        return [24]
        
    return best_set
