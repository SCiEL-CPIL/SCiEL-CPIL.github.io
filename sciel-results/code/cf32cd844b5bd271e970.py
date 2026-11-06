import sys
import time
import random

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill
    
    def check_time():
        return time.time() > deadline
    
    # Known good solution from literature: cyclic difference set
    # For perfect difference sets, we can use Singer's construction
    # But let's try a smarter search first
    
    def check_sidon_and_cover(s):
        """Check if s is a Sidon set covering all diffs 1..30"""
        s = sorted(s)
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        # Check all diffs 1..30 are covered
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    def greedy_construct():
        """Greedy construction that tries to minimize the span"""
        # Start with 0
        s = [0]
        used_diffs = set()
        
        # We need to cover diffs 1..30
        # Greedily add elements that cover the most missing diffs
        for step in range(50):
            if check_time():
                break
            
            missing = [d for d in range(1, 31) if d not in used_diffs]
            if not missing:
                break
            
            # Try candidate values in increasing order
            best_candidate = None
            best_missing_count = 0
            
            # Search up to a reasonable bound
            max_search = max(s) + 100 if s else 100
            
            for x in range(max(s) + 1 if s else 1, max_search + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for y in s:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Count how many missing diffs this would cover
                covers_missing = sum(1 for d in new_diffs if d in missing)
                
                if covers_missing > best_missing_count:
                    best_missing_count = covers_missing
                    best_candidate = x
                    best_new_diffs = new_diffs
                    
                    # Early termination if we cover all missing
                    if covers_missing == len(missing):
                        break
            
            if best_candidate is None:
                # No valid candidate found, try larger range
                max_search = max(s) + 200
                for x in range(max(s) + 1 if s else 1, max_search + 1):
                    valid = True
                    new_diffs = []
                    for y in s:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    covers_missing = sum(1 for d in new_diffs if d in missing)
                    
                    if covers_missing > best_missing_count:
                        best_missing_count = covers_missing
                        best_candidate = x
                        best_new_diffs = new_diffs
                    
                    if covers_missing == len(missing):
                        break
            
            if best_candidate is None:
                # Give up on greedy, return what we have
                break
            
            s.append(best_candidate)
            used_diffs.update(best_new_diffs)
        
        return s if check_sidon_and_cover(s) else None
    
    def backtracking_search(max_val, min_size=8, max_size=12):
        """Backtracking search with aggressive pruning"""
        
        def dfs(current_set, used_diffs_mask, depth, target_size):
            if check_time():
                return None
            
            # Check if all diffs 1..30 are covered
            # used_diffs_mask has bit d-1 set if diff d is used
            target_mask = (1 << 30) - 1
            if (used_diffs_mask & target_mask) == target_mask:
                return list(current_set)
            
            if depth == target_size:
                return None
            
            last_val = current_set[-1] if current_set else 0
            
            # Pruning: can we still cover all missing diffs?
            missing_mask = target_mask & ~used_diffs_mask
            missing_count = missing_mask.bit_count()
            
            remaining = target_size - depth
            # Max new diffs we can add
            max_new_diffs = remaining * (remaining - 1) // 2 + remaining * depth
            if missing_count > max_new_diffs:
                return None
            
            # Generate candidates: values that would cover at least one missing diff
            candidates = set()
            for d in range(1, 31):
                if (missing_mask >> (d - 1)) & 1:
                    for y in current_set:
                        x = y + d
                        if last_val < x <= max_val:
                            candidates.add(x)
            
            # Also try some random candidates for diversity
            for _ in range(min(10, remaining * 2)):
                if current_set:
                    x = random.randint(last_val + 1, max_val)
                    candidates.add(x)
                else:
                    candidates.add(random.randint(1, max_val))
            
            # Sort by how many missing diffs they cover (most constrained first)
            def candidate_score(x):
                count = 0
                for d in range(1, 31):
                    if (missing_mask >> (d - 1)) & 1:
                        for y in current_set:
                            if x - y == d:
                                count += 1
                                break
                return -count  # Negate for ascending sort (most covering first)
            
            sorted_candidates = sorted(candidates, key=candidate_score)
            
            for x in sorted_candidates:
                # Check Sidon property
                valid = True
                new_diffs = []
                for y in current_set:
                    d = x - y
                    if d > 0 and (used_diffs_mask >> (d - 1)) & 1:
                        valid = False
                        break
                    if d > 0:
                        new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Update mask
                new_mask = used_diffs_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                res = dfs(current_set + [x], new_mask, depth + 1, target_size)
                if res:
                    return res
            
            return None
        
        # Try different sizes
        for size in range(min_size, max_size + 1):
            if check_time():
                return None
            res = dfs([0], 0, 1, size)
            if res:
                return res
        
        return None
    
    # Strategy: Try backtracking search for increasing max_val
    # Start from a reasonable lower bound
    # Minimum size for 30 diffs: need k such that k*(k-1)/2 >= 30, so k>=9
    # But Sidon constraint makes it harder
    
    best_solution = None
    best_max_val = float('inf')
    
    # Try greedy first for a quick valid solution
    greedy_result = greedy_construct()
    if greedy_result:
        best_solution = greedy_result
        best_max_val = max(greedy_result)
    
    # Now try to improve with backtracking
    # Start from a lower bound and increase
    # Theoretical lower bound: if we have k elements, max diff is at least ~sqrt(30)*k
    # For k=9, need max diff >= ~17, but Sidon constraint pushes it higher
    
    for max_val in range(35, 120):
        if check_time():
            break
        
        if best_solution and max_val >= best_max_val:
            # We already have a better solution
            break
        
        sol = backtracking_search(max_val, min_size=9, max_size=11)
        if sol:
            sol_max = max(sol)
            if sol_max < best_max_val:
                best_solution = sol
                best_max_val = sol_max
    
    # If we still don't have a solution, use greedy
    if best_solution is None:
        best_solution = greedy_construct()
    
    # Final fallback: construct a valid set
    if best_solution is None:
        # Simple construction: use powers of 2 scaled
        # This won't be optimal but will be valid
        s = [0]
        used = set()
        for d in range(1, 31):
            # Find smallest x such that x-y not in used for all y in s, and x-y=d for some y
            found = False
            for y in s:
                x = y + d
                valid = True
                for z in s:
                    if x != z and (x - z) in used:
                        valid = False
                        break
                if valid:
                    s.append(x)
                    used.add(d)
                    for z in s[:-1]:
                        used.add(x - z)
                    found = True
                    break
            if not found:
                # Add a new element that creates diff d
                x = max(s) + d + 1
                while True:
                    valid = True
                    for z in s:
                        if (x - z) in used:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                s.append(x)
                for z in s[:-1]:
                    used.add(x - z)
        best_solution = s
    
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(res)
