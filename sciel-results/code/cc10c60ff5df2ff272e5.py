import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    weights = {s: L // s for s in range(1, 23)}
    
    # Map element to weight for faster access
    w_list = [0] * 23
    for s in range(1, 23):
        w_list[s] = weights[s]

    def is_dissociated(S):
        if not S:
            return True
        w = sorted([w_list[s] for s in S], reverse=True)
        sums = {0}
        for val in w:
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    def check_add(S, x, sums):
        """Check if adding x to S (with existing sums) is dissociated."""
        w = w_list[x]
        for s in sums:
            if s + w in sums:
                return False
        return True

    def get_sums(S):
        sums = {0}
        # Sort by weight descending for stability? Order doesn't matter for set of sums.
        for x in S:
            w = w_list[x]
            new_sums = {s + w for s in sums}
            sums.update(new_sums)
        return sums

    best_set = []
    best_size = 0
    best_sums = {0}

    # Strategy 1: Multiple greedy passes
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        current_sums = {0}
        for x in cand_order:
            if check_add(current_set, x, current_sums):
                w = w_list[x]
                current_sums.update({s + w for s in current_sums})
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            best_sums = get_sums(best_set)

    # Strategy 2: Local Search
    current_best = list(best_set)
    current_sums = get_sums(current_best)
    
    # To allow escaping local optima, we can use a "kick" strategy
    # or more complex swaps.
    
    for _ in range(500): # Iterations
        if time.time() > deadline - 2:
            break
            
        improved = False
        missing = [x for x in range(1, 23) if x not in current_best]
        
        # 1. Try to add
        if missing:
            # Try all missing, pick first valid
            for x in missing:
                if check_add(current_best, x, current_sums):
                    w = w_list[x]
                    current_sums.update({s + w for s in current_sums})
                    current_best.append(x)
                    improved = True
                    if len(current_best) > best_size:
                        best_size = len(current_best)
                        best_set = list(current_best)
                        best_sums = get_sums(best_set)
                    break
        
        # 2. Try 1-swap
        if not improved and len(current_best) > 0 and missing:
            for x_remove in current_best:
                for x_add in missing:
                    # Compute sums without x_remove
                    # This is expensive if done naively.
                    # Optimization: Precompute sums for subsets?
                    # For now, just do it if we are stuck.
                    sums_without = {0}
                    for x in current_best:
                        if x == x_remove: continue
                        w = w_list[x]
                        new_sums = {s + w for s in sums_without}
                        sums_without.update(new_sums)
                    
                    if check_add([x for x in current_best if x != x_remove], x_add, sums_without):
                        w_add = w_list[x_add]
                        new_sums = {s + w_add for s in sums_without}
                        current_sums = sums_without | new_sums
                        current_best = [x for x in current_best if x != x_remove] + [x_add]
                        improved = True
                        if len(current_best) > best_size:
                            best_size = len(current_best)
                            best_set = list(current_best)
                            best_sums = get_sums(best_set)
                        break
                if improved:
                    break
        
        # 3. Try 2-swap (remove 2, add 2) - only if stuck
        if not improved and len(current_best) > 2 and len(missing) >= 2:
            # Sample a few
            for _ in range(5):
                if time.time() > deadline - 2: break
                x_rem1, x_rem2 = random.sample(current_best, 2)
                x_add1, x_add2 = random.sample(missing, 2)
                
                sums_without = {0}
                for x in current_best:
                    if x == x_rem1 or x == x_rem2: continue
                    w = w_list[x]
                    new_sums = {s + w for s in sums_without}
                    sums_without.update(new_sums)
                
                # Check if we can add x_add1
                w1 = w_list[x_add1]
                ok1 = True
                for s in sums_without:
                    if s + w1 in sums_without:
                        ok1 = False
                        break
                if not ok1: continue
                
                sums_after1 = sums_without | {s + w1 for s in sums_without}
                
                # Check if we can add x_add2
                w2 = w_list[x_add2]
                ok2 = True
                for s in sums_after1:
                    if s + w2 in sums_after1:
                        ok2 = False
                        break
                if not ok2: continue
                
                new_sums = sums_after1 | {s + w2 for s in sums_after1}
                current_sums = new_sums
                current_best = [x for x in current_best if x != x_rem1 and x != x_rem2] + [x_add1, x_add2]
                improved = True
                if len(current_best) > best_size:
                    best_size = len(current_best)
                    best_set = list(current_best)
                    best_sums = get_sums(best_set)
                break

        # 4. Kick: Random restart from best_set if stuck for a while
        if not improved and random.random() < 0.05:
            current_best = list(best_set)
            current_sums = get_sums(current_best)

    return best_set if best_set else [1]
