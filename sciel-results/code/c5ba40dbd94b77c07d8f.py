import time
import sys

def _verify_distinct_subset_sums(vals):
    sums = {0}
    for v in vals:
        new_sums = {s + v for s in sums}
        if len(sums) + len(new_sums) != len(sums.union(new_sums)):
            return False
        sums = sums.union(new_sums)
    return True

def construct():
    n = 12
    best_max = 2048
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    def search(depth, current_vals, current_sums, current_max):
        nonlocal best_max
        if time.time() > deadline:
            return
        
        if depth == n:
            if current_max < best_max:
                best_max = current_max
            return
        
        # Lower bound for next value
        # Must be distinct and positive.
        # To maintain sorted order and avoid permutations, 
        # we can require next_val > current_vals[-1] if current_vals else next_val >= 1
        if current_vals:
            lb = current_vals[-1] + 1
        else:
            lb = 1
            
        # Upper bound is best_max
        ub = best_max
        
        # Try values from lb to ub
        # Optimization: If we find a solution, we might want to try smaller values first?
        # Yes, iterating lb to ub does that.
        
        for v in range(lb, ub + 1):
            if time.time() > deadline:
                return
            
            # Check if v creates distinct sums
            # new_sums = current_sums U (current_sums + v)
            # Disjoint if current_sums and (current_sums + v) are disjoint
            
            # Efficient check:
            # If v is in (current_sums - current_sums), it's a collision.
            # But computing difference set is O(N^2).
            # Checking intersection of sets is O(N).
            # N is 2^depth.
            
            # We can just generate new_sums and check length
            # Or check if any s + v in current_sums
            
            # Faster: 
            # If v > max(current_sums), then disjoint.
            # max(current_sums) is sum(current_vals)
            sum_current = sum(current_vals)
            
            if v > sum_current:
                # Always valid
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s)
                    new_sums.add(s + v)
                search(depth + 1, current_vals + [v], new_sums, max(current_max, v))
            else:
                # Check intersection
                # We need to know if (current_sums + v) intersects current_sums
                # This is equivalent to v in {s1 - s2 | s1, s2 in current_sums}
                # Or just iterate
                valid = True
                # Optimization: if v is small, likely collision.
                # We can check a few samples? No, must be exact.
                
                # Generate new sums and check size
                # This is O(2^depth)
                # Since depth is small, this is fine.
                
                # Actually, we can just do:
                # if any((s + v) in current_sums for s in current_sums):
                #     valid = False
                
                # But we need new_sums for recursion.
                # So we generate them anyway.
                
                new_sums = set(current_sums)
                for s in current_sums:
                    ns = s + v
                    if ns in current_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                
                if valid:
                    search(depth + 1, current_vals + [v], new_sums, max(current_max, v))
                    
    search(0, [], {0}, 0)
    
    # If no better found, return powers of 2
    if best_max == 2048:
        return [1 << i for i in range(n)]
    else:
        # We need to return the actual values, not just the max.
        # The search above only tracks max.
        # We need to store the best_vals.
        pass

def construct():
    n = 12
    best_max = 2048
    best_vals = [1 << i for i in range(n)]
    start_time = time.time()
    deadline = start_time + 50.0
    
    def search(depth, current_vals, current_sums, current_max):
        nonlocal best_max, best_vals
        if time.time() > deadline:
            return
        
        if depth == n:
            if current_max < best_max:
                best_max = current_max
                best_vals = list(current_vals)
            return
        
        if current_vals:
            lb = current_vals[-1] + 1
        else:
            lb = 1
            
        ub = best_max
        
        for v in range(lb, ub + 1):
            if time.time() > deadline:
                return
            
            sum_current = sum(current_vals)
            
            if v > sum_current:
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s)
                    new_sums.add(s + v)
                search(depth + 1, current_vals + [v], new_sums, max(current_max, v))
            else:
                valid = True
                new_sums = set(current_sums)
                for s in current_sums:
                    ns = s + v
                    if ns in current_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                
                if valid:
                    search(depth + 1, current_vals + [v], new_sums, max(current_max, v))

    search(0, [], {0}, 0)
    
    return best_vals
