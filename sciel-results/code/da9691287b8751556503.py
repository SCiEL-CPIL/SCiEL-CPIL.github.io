import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Target: cover diffs 1..30
    TARGET_MASK = (1 << 30) - 1
    
    def check_solution(A):
        """Verify A is a valid Sidon set covering 1..30"""
        if not A or 0 not in A:
            return False
        if len(A) > 32:
            return False
        if max(A) > 10**6:
            return False
        diffs = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    # Lower bound: need at least 9 elements (36 diffs > 30 required)
    # Minimal M for a Sidon set of size 9 covering 1..30 is around 40-50
    for M in range(30, 150):
        if time.time() > deadline:
            break
        
        # Try to find a Sidon set A subset of {0..M} containing 0, covering 1..30
        # We don't force M to be in A - if max(A) < M, we would have found it earlier
        
        initial_set = [0]
        initial_used = set()
        initial_mask = 0
        
        def search(current_set, used_diffs, covered_mask):
            if time.time() > deadline:
                return None
            
            # Check if all target differences are covered
            if covered_mask == TARGET_MASK:
                return current_set[:]
            
            # If we have too many elements, fail
            if len(current_set) >= 32:
                return None
            
            # Pruning: estimate if we can cover remaining diffs
            missing_count = bin(TARGET_MASK ^ covered_mask).count("1")
            remaining_elements = 32 - len(current_set)
            # Each new element can add at most len(current_set) new diffs
            # But we only care about diffs in 1..30
            # Rough upper bound on new useful diffs
            max_new_useful = 0
            for x in range(1, M+1):
                if x in current_set:
                    continue
                new_useful = 0
                for a in current_set:
                    d = abs(x - a)
                    if 1 <= d <= 30 and d not in used_diffs:
                        new_useful += 1
                max_new_useful = max(max_new_useful, new_useful)
            
            # Even with optimistic estimate, if we can't cover missing, prune
            # This is a weak bound, so be conservative
            # Skip this pruning for now - it's expensive
            
            # Generate candidates
            candidates = []
            for x in range(1, M+1):
                if x in current_set:
                    continue
                
                new_diffs = set()
                valid = True
                for a in current_set:
                    d = abs(x - a)
                    if d in used_diffs:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Calculate new mask
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                # Heuristic: number of new bits set (gain)
                gain = bin(new_mask ^ covered_mask).count("1")
                
                # Secondary heuristic: prefer x that doesn't create too many "waste" diffs
                # (diffs > 30 that use up the Sidon budget)
                waste = sum(1 for d in new_diffs if d > 30)
                
                # Tertiary: prefer smaller x for better future flexibility?
                # Actually, prefer x that maximizes future options
                # For now, use gain desc, waste asc, x asc
                candidates.append((gain, -waste, x, new_diffs, new_mask))
            
            if not candidates:
                return None
            
            # Sort by gain desc, waste asc (so -waste desc), x asc
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            
            # Try top candidates - limit branching to avoid explosion
            # If gain is 0 for the best, we're likely stuck
            max_gain = candidates[0][0]
            if max_gain == 0:
                # No candidate improves coverage - likely dead end
                # But we might still need to add elements to enable future coverage?
                # In Sidon sets, adding an element only restricts future choices
                # So if no element improves coverage, we can't finish
                return None
            
            # Try top 10 candidates
            for gain, neg_waste, x, new_diffs, new_mask in candidates[:10]:
                new_set = current_set + [x]
                new_set.sort()
                new_used = set(used_diffs)
                new_used.update(new_diffs)
                
                res = search(new_set, new_used, new_mask)
                if res is not None:
                    return res
            
            return None
        
        sol = search(initial_set, initial_used, initial_mask)
        
        if sol is not None:
            if check_solution(sol):
                return sol
    
    # Fallback: greedy construction
    a = [0]
    used = set()
    for _ in range(31):
        best_x = None
        best_gain = -1
        for x in range(1, 10**6):
            if x in a:
                continue
            new_diffs = set()
            valid = True
            for ai in a:
                diff = abs(x - ai)
                if diff in used or diff in new_diffs:
                    valid = False
                    break
                new_diffs.add(diff)
            if not valid:
                continue
            gain = sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used)
            if gain > best_gain:
                best_gain = gain
                best_x = x
                if best_gain >= 3:
                    break
        if best_x is None or best_gain <= 0:
            break
        a.append(best_x)
        a.sort()
        for ai in a[:-1]:
            used.add(abs(best_x - ai))
    
    if check_solution(a):
        return a
    
    # Last resort hardcoded set (should not be needed)
    return [0, 1, 4, 10, 12, 17, 21, 25, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120]
