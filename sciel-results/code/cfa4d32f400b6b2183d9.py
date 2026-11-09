import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
    
    def check_sidon_bitset(elements):
        """Fast Sidon check using difference tracking"""
        n = len(elements)
        if n < 2:
            return True
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = elements[j] - elements[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def get_diff_mask(elements):
        """Get bitmask of all positive differences"""
        mask = 0
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                d = elements[j] - elements[i]
                if d <= 30:
                    mask |= (1 << d)
        return mask
    
    def is_valid(elements):
        """Full validity check"""
        if len(elements) < 2 or len(elements) > 32:
            return False
        if 0 not in elements:
            return False
        if max(elements) > 10**6:
            return False
        if not check_sidon_bitset(elements):
            return False
        if not (get_diff_mask(elements) & target_mask) == target_mask:
            return False
        return True
    
    def fast_backtrack(D, time_limit=None):
        """
        Fast backtracking with bitset difference tracking.
        Uses aggressive pruning: if a difference is already used, 
        any candidate creating that difference is immediately rejected.
        """
        if time_limit is None:
            time_limit = deadline - 5
        
        best_result = [None]
        
        def dfs(elements, used_diff_mask, covered_mask, last_elem):
            if time.time() > time_limit:
                return False
            
            if (covered_mask & target_mask) == target_mask:
                best_result[0] = elements[:]
                return True
            
            k = len(elements)
            if k >= 32:
                return False
            
            missing_mask = target_mask & ~covered_mask
            if missing_mask == 0:
                best_result[0] = elements[:]
                return True
            
            # Pruning: remaining capacity
            # Each new element can add at most k new differences with existing elements
            # We need to cover popcount(missing_mask) more differences
            remaining_slots = 32 - k
            # Very conservative: each element covers at most k new target diffs
            # Count bits in missing_mask
            missing_count = bin(missing_mask).count('1')
            if missing_count > remaining_slots * k:
                return False
            
            # Generate candidates in order of coverage potential
            # Try values that create many new covered differences
            candidates = []
            for x in range(last_elem + 1, D + 1):
                # Check against all existing elements
                valid = True
                new_covered = 0
                for y in elements:
                    d = x - y
                    if d > 30:
                        continue
                    bit = 1 << d
                    if (used_diff_mask & bit) != 0:
                        valid = False
                        break
                    if (target_mask & bit) != 0:
                        new_covered += 1
                
                if not valid:
                    continue
                
                if new_covered > 0:
                    candidates.append((new_covered, x))
            
            # Sort by coverage descending, then value ascending
            candidates.sort(key=lambda c: (-c[0], c[1]))
            
            # Limit for performance
            max_cands = 40
            if len(candidates) > max_cands:
                candidates = candidates[:max_cands]
            
            for _, x in candidates:
                # Compute new used_diff_mask
                new_used = used_diff_mask
                for y in elements:
                    d = x - y
                    if d <= 30:
                        new_used |= (1 << d)
                
                new_covered = covered_mask
                for y in elements:
                    d = x - y
                    if d <= 30:
                        new_covered |= (1 << d)
                
                if dfs(elements + [x], new_used, new_covered, x):
                    return True
            
            return False
        
        dfs([0], 0, 0, 0)
        return best_result[0]
    
    def local_search_minimize(S, time_limit=None):
        """
        Local search to minimize max element.
        Tries to replace elements with smaller values.
        """
        if time_limit is None:
            time_limit = deadline - 10
        
        if S is None:
            return None
        
        best_S = sorted(S)
        best_max = max(best_S)
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < time_limit and no_improve < max_no_improve:
            improved = False
            S_sorted = sorted(best_S)
            n = len(S_sorted)
            
            for i in range(n - 1, 0, -1):
                if time.time() > time_limit:
                    break
                
                s_i = S_sorted[i]
                S_rest = S_sorted[:i] + S_sorted[i + 1:]
                
                # Try smaller values for position i
                lower = S_sorted[i - 1] + 1 if i > 0 else 1
                upper = s_i - 1
                
                if upper <= lower:
                    continue
                
                # Try a range of smaller values
                step = max(1, (upper - lower) // 50)
                trials = set()
                for v in range(upper, lower - 1, -step):
                    trials.add(v)
                trials.add(lower)
                
                for new_val in sorted(trials, reverse=True):
                    if new_val < lower or new_val >= s_i:
                        continue
                    
                    S_new = S_rest + [new_val]
                    S_new.sort()
                    
                    if not check_sidon_bitset(S_new):
                        continue
                    if not (get_diff_mask(S_new) & target_mask) == target_mask:
                        continue
                    
                    new_max = max(S_new)
                    if new_max < best_max:
                        best_S = S_new
                        best_max = new_max
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                no_improve = 0
            else:
                no_improve += 1
        
        return best_S
    
    # Main: try increasing D values
    best_result = None
    best_score = float('inf')
    
    # Start from a reasonable lower bound
    # For covering differences 1..30 with a Sidon set, 
    # the minimum diameter is at least around 30 (trivially)
    # but practically much larger. Start from 50.
    for D in range(50, 200):
        if time.time() > deadline - 30:
            break
        
        result = fast_backtrack(D)
        if result is not None:
            if is_valid(result):
                score = max(result)
                if score < best_score:
                    best_score = score
                    best_result = result[:]
                    # Try to optimize
                    optimized = local_search_minimize(best_result)
                    if optimized is not None and is_valid(optimized):
                        opt_score = max(optimized)
                        if opt_score < best_score:
                            best_score = opt_score
                            best_result = optimized[:]
                    break
    
    # If backtracking didn't find anything, use greedy fallback
    if best_result is None:
        S = [0]
        used_diffs = set()
        covered = set()
        
        for step in range(32):
            if time.time() > deadline - 15:
                break
            
            missing = target_diffs - covered
            if not missing:
                break
            
            best_x = None
            best_coverage = -1
            
            for x in range(1, 1000):
                if x in S:
                    continue
                valid = True
                new_cov = 0
                for y in S:
                    d = abs(x - y)
                    if d in used_diffs:
                        valid = False
                        break
                    if d in missing:
                        new_cov += 1
                
                if not valid:
                    continue
                
                if new_cov > best_coverage:
                    best_coverage = new_cov
                    best_x = x
            
            if best_x is None:
                # Add a large element
                found = False
                base = max(S) + 1
                for offset in range(1, 200):
                    x = base + offset
                    valid = True
                    for y in S:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        S.append(x)
                        S.sort()
                        used_diffs = set()
                        covered = set()
                        for i in range(len(S)):
                            for j in range(i + 1, len(S)):
                                d = S[j] - S[i]
                                used_diffs.add(d)
                                if d in target_diffs:
                                    covered.add(d)
                        found = True
                        break
                if not found:
                    break
            else:
                S.append(best_x)
                S.sort()
                used_diffs = set()
                covered = set()
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        d = S[j] - S[i]
                        used_diffs.add(d)
                        if d in target_diffs:
                            covered.add(d)
        
        if is_valid(S):
            best_result = S
            best_score = max(S)
    
    # Final polish
    if best_result is not None:
        final = local_search_minimize(best_result)
        if final is not None and is_valid(final):
            if max(final) < max(best_result):
                best_result = final
    
    # Ensure we return a valid set
    if best_result is None or not is_valid(best_result):
        # Last resort: build a guaranteed valid set
        S = [0]
        used_diffs = set()
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            for x in range(1, 5000):
                if x in S:
                    continue
                trial = sorted(S + [x])
                new_diffs = set()
                valid = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        diff = trial[j] - trial[i]
                        if diff in new_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    if not valid:
                        break
                if valid:
                    S = trial
                    used_diffs = new_diffs
                    for dd in new_diffs:
                        if dd in target_diffs:
                            covered.add(dd)
                    found = True
                    break
            if not found:
                y = 2 * max(S) + 3 * d + 10
                while True:
                    trial = sorted(S + [y])
                    new_diffs = set()
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            diff = trial[j] - trial[i]
                            if diff in new_diffs:
                                valid = False
                                break
                            new_diffs.add(diff)
                        if not valid:
                            break
                    if valid:
                        S = trial
                        used_diffs = new_diffs
                        for dd in new_diffs:
                            if dd in target_diffs:
                                covered.add(dd)
                        break
                    y += 1
        
        best_result = S
    
    return best_result
