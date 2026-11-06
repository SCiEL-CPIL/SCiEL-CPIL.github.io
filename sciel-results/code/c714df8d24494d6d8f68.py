import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # Generate 2-3-5 smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute compatibility (divisibility antichain check)
    # compat[i][j] is True if smooth[i] and smooth[j] are compatible (neither divides the other)
    compat = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i+1, n_smooth):
            a, b = smooth[i], smooth[j]
            if a % b == 0 or b % a == 0:
                compat[i][j] = False
                compat[j][i] = False
    
    # Precompute index for quick lookup
    idx_map = {val: i for i, val in enumerate(smooth)}
    
    answers = [None] * 128

    # Strategy: Pre-enumerate valid antichains of small size (1, 2, 3)
    # and assign them to targets if not already assigned.
    # This ensures we find the minimal term count for as many targets as possible
    # without per-target expensive search, leveraging the fact that most targets
    # can be represented by <= 3 terms.

    # Level 1: 1 term
    for s in smooth:
        if s <= 128:
            answers[s - 1] = [s]

    # Level 2: 2 terms
    # Iterate pairs (i, j) with i < j.
    # If compatible, sum = smooth[i] + smooth[j].
    # If sum <= 128 and answers[sum-1] is None, assign.
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            if compat[i][j]:
                if answers[s - 1] is None:
                    answers[s - 1] = [a, b]

    # Check if all targets are covered
    if all(ans is not None for ans in answers):
        return answers

    # Level 3: 3 terms
    # Iterate triples (i, j, k) with i < j < k.
    # Check antichain property for all 3 pairs.
    # If valid and sum <= 128, assign if not already assigned.
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if not compat[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    break
                # Check remaining pairs (i,k) and (j,k)
                if compat[i][k] and compat[j][k]:
                    if answers[s - 1] is None:
                        answers[s - 1] = [a, b, c]

    # Check again
    if all(ans is not None for ans in answers):
        return answers

    # Fallback: For any remaining targets, perform a targeted backtracking search
    # starting from count 4 upwards.
    # This is unlikely to be needed for 1..128, but ensures validity.
    for target_idx in range(128):
        if answers[target_idx] is not None:
            continue
        
        if time.time() > deadline:
            # Emergency: if we run out of time, we must still return something.
            # But the task requires valid representations.
            # If we haven't found one, we have a problem.
            # However, given the density, 3 terms should cover everything.
            # Let's try a quick brute force for the specific target with limited depth.
            pass

        target = target_idx + 1
        candidates = [x for x in smooth if x <= target]
        
        # Try counts from 4 to 10 (should be more than enough)
        found = False
        for count in range(4, 11):
            if time.time() > deadline:
                break
            if len(candidates) < count:
                break
            
            # Pruning
            min_sum = sum(candidates[:count])
            max_sum = sum(candidates[-count:])
            if min_sum > target or max_sum < target:
                continue

            # Backtracking search for this specific count
            # We use a recursive helper for efficiency
            def backtrack(start_idx, current_terms, remaining):
                if len(current_terms) == count:
                    if remaining == 0:
                        return list(current_terms)
                    return None
                
                # Pruning: if remaining < 0, fail
                if remaining < 0:
                    return None
                
                # If not enough elements left to reach count, fail
                if start_idx + (count - len(current_terms)) > len(candidates):
                    return None
                
                # Try each candidate from start_idx
                for idx in range(start_idx, len(candidates)):
                    val = candidates[idx]
                    if val > remaining:
                        continue
                    
                    # Check compatibility with all existing terms
                    ok = True
                    for existing_val in current_terms:
                        existing_idx = idx_map[existing_val]
                        if not compat[existing_idx][idx]:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    current_terms.append(val)
                    result = backtrack(idx + 1, current_terms, remaining - val)
                    if result:
                        return result
                    current_terms.pop()
                
                return None

            result = backtrack(0, [], target)
            if result:
                answers[target_idx] = result
                found = True
                break
        
        if not found:
            # This should not happen for 1..128
            # As a last resort, if target is smooth, use [target] (though we checked count 1)
            # Or raise an error? The task says "All targets require valid representations".
            # If we fail, the score is 0.
            # Let's try a broader search if this happens.
            # But for now, leave it as None which will cause an error in evaluation if not handled.
            # Actually, let's just try combinations up to 15 if backtracking failed.
            for count in range(11, 20):
                if time.time() > deadline:
                    break
                if len(candidates) < count:
                    break
                min_sum = sum(candidates[:count])
                max_sum = sum(candidates[-count:])
                if min_sum > target or max_sum < target:
                    continue
                
                # Use itertools.combinations for simplicity in fallback
                for terms in combinations(candidates, count):
                    if sum(terms) == target:
                        # Check antichain
                        idxs = [idx_map[t] for t in terms]
                        ok = True
                        for ii in range(len(idxs)):
                            for jj in range(ii+1, len(idxs)):
                                if not compat[idxs[ii]][idxs[jj]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            answers[target_idx] = list(terms)
                            found = True
                            break
                if found:
                    break

    return answers
