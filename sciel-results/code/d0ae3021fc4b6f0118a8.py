"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    best_set = None
    best_len = 0
    
    # Precompute: for each a in [1,256], and each possible sum s in [2,512],
    # store whether a divides s. This helps fast conflict checking.
    # But more useful: for each a, precompute the set of (b,c) pairs where b>=a, c>=b, and a | (b+c)
    # Actually, let's precompute for each a, the forbidden sums: sums s where a|s
    # Then when checking if we can add x to a set, we need to check all triples (a,b,c) in set∪{x}
    # with a<b<=c and a|(b+c)
    
    # Better approach: precompute for each a, the set of residue classes mod a that are forbidden
    # For a given a, b+c must not be divisible by a. So b+c mod a != 0.
    
    # Let's precompute: for each a in 1..256, for each r in 0..a-1, 
    # but this is complex. Instead, let's optimize the can_add function.
    
    def check_set_validity_fast(s):
        """Check if the entire set satisfies the constraint using sorted list."""
        s_sorted = sorted(s)
        n = len(s_sorted)
        for i in range(n):
            a = s_sorted[i]
            # For this a, check all pairs (b,c) with b>a, c>=b
            for j in range(i+1, n):
                b = s_sorted[j]
                # For fixed a,b, c must satisfy a | (b+c), i.e., c ≡ -b (mod a)
                # So c = b + k*a for some k>=0, and c>=b, c in set, c>=b
                # c ranges from b to max_set
                target = (-b) % a
                # c must be in set and c >= b and c ≡ target (mod a)
                # Since c >= b and c ≡ target (mod a), c = target + m*a where m*a >= b - target
                # But simpler: iterate c from j to n-1 and check
                for k in range(j, n):
                    c = s_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add_fast(x, current_sorted, current_set):
        """Check if x can be added to current set. current_sorted is sorted list of current set."""
        if x in current_set:
            return False
        
        # Build new sorted list by inserting x
        # Instead of full rebuild, check conflicts involving x
        # x can play role of a, b, or c
        # Case 1: x is the smallest (a = x)
        #   Then for all b,c in current with b > x, c >= b: check x | (b+c)
        for i in range(len(current_sorted)):
            b = current_sorted[i]
            if b <= x:
                continue
            for j in range(i, len(current_sorted)):
                c = current_sorted[j]
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is b or c, and a is in current (a < x)
        # For each a in current with a < x:
        #   x can be b: then c >= x, c in current, check a | (x + c)
        #   x can be c: then b < x, b in current, b > a, check a | (b + x)
        for a in current_sorted:
            if a >= x:
                break  # since current_sorted is sorted, all subsequent are >= x
            # x as b: need c >= x, c in current, c > a (auto since c >= x > a)
            for c in current_sorted:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
            # x as c: need b < x, b in current, b > a
            for b in current_sorted:
                if b <= a:
                    continue
                if b >= x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def greedy_extend_optimized(start_set, order='ascending'):
        """Try to extend a set by adding smaller elements."""
        current = set(start_set)
        current_sorted = sorted(current)
        
        # Try adding elements from 1 to min(current) - 1
        min_cur = min(current)
        candidates = list(range(1, min_cur))
        
        if order == 'ascending':
            pass
        elif order == 'descending':
            candidates.reverse()
        else:
            random.shuffle(candidates)
        
        added_any = True
        while added_any:
            added_any = False
            current_sorted = sorted(current)
            min_cur = min(current)
            candidates = [x for x in range(1, min_cur) if x not in current]
            if order == 'ascending':
                pass
            elif order == 'descending':
                candidates.reverse()
            else:
                random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return current
                if can_add_fast(x, current_sorted, current):
                    current.add(x)
                    added_any = True
                    break  # restart with new min
        
        return current
    
    # Try multiple starting points with different strategies
    for start in range(150, 257):
        if time.time() > deadline:
            break
        
        # Start with upper tail from start to 256
        start_set = list(range(start, 257))
        
        # Verify base set is valid
        if not check_set_validity_fast(start_set):
            continue
        
        if len(start_set) > best_len:
            best_set = list(start_set)
            best_len = len(start_set)
        
        # Try to extend with different orderings
        for order in ['ascending', 'descending', 'random']:
            if time.time() > deadline:
                break
            extended = greedy_extend_optimized(set(start_set), order)
            if len(extended) > best_len:
                best_set = sorted(extended)
                best_len = len(best_set)
    
    # Additional: try local search on the best set found
    if best_set is not None and time.time() < deadline - 10:
        # Try removing one element and adding two others
        best_set_set = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current = set(best_set_set)
            current_sorted = sorted(current)
            
            for a in list(current):
                if time.time() > deadline - 5:
                    break
                # Try removing a
                trial = current - {a}
                trial_sorted = sorted(trial)
                
                # Try adding any x not in trial
                for x in range(1, 257):
                    if x in trial:
                        continue
                    if can_add_fast(x, trial_sorted, trial):
                        new_set = trial | {x}
                        if len(new_set) > len(best_set_set):
                            best_set_set = new_set
                            best_set = sorted(new_set_set)
                            best_len = len(best_set)
                            improved = True
                            break
                if improved:
                    break
    
    # If we didn't find anything, return a known valid set
    if best_set is None:
        best_set = list(range(192, 257))
    
    return best_set
# EVOLVE-BLOCK-END
