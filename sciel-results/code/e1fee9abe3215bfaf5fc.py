import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = 135  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    def valid_sidon_with_coverage(S):
        if not S or 0 not in S:
            return False
        if len(S) > 32:
            return False
        if max(S) > 10**6:
            return False
        if not is_sidon(S):
            return False
        if not covers_all(S):
            return False
        return True

    best_max = float('inf')
    best_set = None
    
    # CSP Backtracking with Forward Checking
    # State: (elements_list, used_diffs_set, covered_mask)
    # We build the set incrementally, always adding the next smallest element
    
    def backtrack(elements, used_diffs, covered_mask, depth):
        nonlocal best_max, best_set
        
        if time.time() - start_time > deadline:
            return False
        
        if covered_mask == full_mask:
            current_max = max(elements) if elements else 0
            if current_max < best_max:
                best_max = current_max
                best_set = elements[:]
            return True
        
        if len(elements) >= 32:
            return False
        
        # Generate candidate values for the next element
        # The next element must be > max(elements)
        last = max(elements) if elements else -1
        
        # Strategy: try values that cover missing differences
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        # Generate candidates from missing differences
        candidates = set()
        for d in missing_diffs:
            for s in elements:
                x = s + d
                if x > last and x <= 10**6:
                    candidates.add(x)
        
        # Also try some values near the current max
        for offset in range(1, 50):
            x = last + offset
            if x <= 10**6:
                candidates.add(x)
        
        # Sort candidates by value (prefer smaller to keep max low)
        candidates = sorted(candidates)
        
        # Forward checking: for each candidate, check if it maintains Sidon property
        for x in candidates:
            if time.time() - start_time > deadline:
                return False
            
            # Check Sidon condition: x - s must not be in used_diffs for any s in elements
            valid = True
            new_diffs = []
            for s in elements:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Update covered mask
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            # Pruning: if we can't possibly cover all remaining diffs with remaining elements
            remaining_slots = 32 - len(elements) - 1
            remaining_missing = 30 - bin(new_mask).count('1')
            # Each new element can cover at most len(elements)+1 new diffs (differences to existing elements)
            # But we need to be optimistic: assume each new element covers all remaining missing
            if remaining_missing > 0 and remaining_slots == 0:
                continue
            
            # Recursive call
            elements.append(x)
            new_used = used_diffs.union(new_diffs)
            
            if backtrack(elements, new_used, new_mask, depth + 1):
                return True
            
            elements.pop()
        
        return False
    
    # Start with [0]
    initial_elements = [0]
    initial_used = set()
    initial_mask = 0
    
    # Try to find a solution
    found = backtrack(initial_elements, initial_used, initial_mask, 0)
    
    if not found and best_set is None:
        # Fallback: greedy construction
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32 and time.time() - start_time < deadline:
            missing = target_diffs - covered
            best_candidate = None
            for d in sorted(missing):
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        if best_candidate is None or c < best_candidate:
                            best_candidate = c
            if best_candidate is not None:
                S.append(best_candidate)
                S.sort()
                covered = get_diffs(S)
            else:
                c = max(S) + 1
                while c <= 10**6 and time.time() - start_time < deadline:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
                else:
                    break
        return S
    
    if best_set:
        # Local search polish: try to reduce the maximum element
        S = best_set[:]
        S.sort()
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # Try to reduce each element starting from the largest
            for i in range(len(S) - 1, 0, -1):
                if (time.time() - start_time) > deadline:
                    break
                    
                s_i = S[i]
                S_rest = S[:i] + S[i+1:]
                
                # Try replacing s_i with smaller values
                lower_bound = S[i-1] + 1 if i > 0 else 1
                upper_bound = s_i - 1
                
                if upper_bound < lower_bound:
                    continue
                
                # Try values that create useful differences
                trial_values = set()
                for d in range(1, 31):
                    for s in S_rest:
                        trial_val = s + d
                        if lower_bound <= trial_val < s_i:
                            trial_values.add(trial_val)
                
                # Also try values just below s_i
                for offset in range(1, min(15, s_i - lower_bound + 1)):
                    trial_values.add(s_i - offset)
                
                for new_val in sorted(trial_values, reverse=True):
                    if new_val <= lower_bound or new_val >= s_i:
                        continue
                    
                    S_new = S_rest + [new_val]
                    S_new.sort()
                    
                    if is_sidon(S_new) and covers_all(S_new):
                        S = S_new
                        improved = True
                        break
                
                if improved:
                    break  # Restart with new S
            
            # Try removing elements if possible
            if not improved and len(S) > 2:
                for i in range(len(S) - 1, 0, -1):
                    if (time.time() - start_time) > deadline:
                        break
                    S_try = S[:i] + S[i+1:]
                    if is_sidon(S_try) and covers_all(S_try):
                        S = S_try
                        improved = True
                        break
        
        return S
    
    return [0]
