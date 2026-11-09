# EVOLVE-BLOCK-START
import time
from functools import lru_cache
import itertools

def construct():
    """
    Returns 128 lists representing targets 1..128.
    Each list contains distinct 5-smooth numbers summing to the target.
    The set of numbers in each list forms a divisibility antichain.
    Minimizes the total number of summands across all targets.
    """
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit

    # 1. Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_numbers.sort()
    n_smooth = len(smooth_numbers)
    
    # 2. Precompute divisibility relation
    divides = [[False]*n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth_numbers[j] % smooth_numbers[i] == 0:
                divides[i][j] = True
    
    # Precompute incompatibility: i and j incompatible if divides[i][j] or divides[j][i]
    incompatible = [[False]*n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                incompatible[i][j] = divides[i][j] or divides[j][i]
    
    # 3. Search for minimal antichain representation for each target
    best_answers = [None] * 129
    best_counts = [float('inf')] * 129
    unsolved = set(range(1, 129))
    
    # Helper to check if a list of indices forms an antichain
    def is_antichain(indices):
        for a in range(len(indices)):
            for b in range(a+1, len(indices)):
                if incompatible[indices[a]][indices[b]]:
                    return False
        return True
    
    # k=1
    for i in range(n_smooth):
        val = smooth_numbers[i]
        if val <= 128 and val in unsolved:
            best_answers[val] = [val]
            best_counts[val] = 1
            unsolved.remove(val)
    
    if time.time() - start_time > timeout:
        return _finalize(best_answers, smooth_numbers, incompatible)
    
    # k=2
    for i in range(n_smooth):
        vi = smooth_numbers[i]
        for j in range(i+1, n_smooth):
            vj = smooth_numbers[j]
            s = vi + vj
            if s > 128:
                break
            if s in unsolved and not incompatible[i][j]:
                best_answers[s] = [vi, vj]
                best_counts[s] = 2
                unsolved.remove(s)
    
    if time.time() - start_time > timeout:
        return _finalize(best_answers, smooth_numbers, incompatible)
    
    # k=3
    for i in range(n_smooth):
        vi = smooth_numbers[i]
        for j in range(i+1, n_smooth):
            vj = smooth_numbers[j]
            s2 = vi + vj
            if s2 > 128: break
            if incompatible[i][j]: continue
            for k in range(j+1, n_smooth):
                vk = smooth_numbers[k]
                s3 = s2 + vk
                if s3 > 128: break
                if s3 in unsolved:
                    if not incompatible[i][k] and not incompatible[j][k]:
                        best_answers[s3] = [vi, vj, vk]
                        best_counts[s3] = 3
                        unsolved.remove(s3)
    
    if time.time() - start_time > timeout:
        return _finalize(best_answers, smooth_numbers, incompatible)
    
    # k=4
    for i in range(n_smooth):
        vi = smooth_numbers[i]
        for j in range(i+1, n_smooth):
            vj = smooth_numbers[j]
            s2 = vi + vj
            if s2 > 128: break
            if incompatible[i][j]: continue
            for k in range(j+1, n_smooth):
                vk = smooth_numbers[k]
                s3 = s2 + vk
                if s3 > 128: break
                if incompatible[i][k] or incompatible[j][k]: continue
                for l in range(k+1, n_smooth):
                    vl = smooth_numbers[l]
                    s4 = s3 + vl
                    if s4 > 128: break
                    if s4 in unsolved:
                        if not incompatible[i][l] and not incompatible[j][l] and not incompatible[k][l]:
                            best_answers[s4] = [vi, vj, vk, vl]
                            best_counts[s4] = 4
                            unsolved.remove(s4)
    
    if time.time() - start_time > timeout:
        return _finalize(best_answers, smooth_numbers, incompatible)
    
    # k=5 and beyond: use backtracking for remaining unsolved
    if unsolved:
        for target in sorted(unsolved):
            if time.time() - start_time > timeout:
                break
            ans = _find_antichain_backtrack(target, smooth_numbers, incompatible, n_smooth)
            if ans:
                best_answers[target] = ans
                best_counts[target] = len(ans)
                unsolved.remove(target)
    
    return _finalize(best_answers, smooth_numbers, incompatible)

def _find_antichain_backtrack(target, smooth_numbers, incompatible, n_smooth):
    """Find an antichain representation of target using backtracking with pruning."""
    # Try k from 1 to 10
    for k in range(1, 11):
        result = _backtrack_search(target, smooth_numbers, incompatible, n_smooth, k)
        if result is not None:
            return result
    return None

def _backtrack_search(target, smooth_numbers, incompatible, n_smooth, k):
    """Backtrack to find k distinct smooth numbers summing to target, forming an antichain."""
    if k == 0:
        return [] if target == 0 else None
    
    # Pruning: if target < smallest possible sum of k remaining elements, fail
    # If target > largest possible sum, fail
    
    # Use recursion with index to avoid duplicates
    def backtrack(start_idx, remaining, chosen):
        if len(chosen) == k:
            return list(chosen) if remaining == 0 else None
        
        for i in range(start_idx, n_smooth):
            val = smooth_numbers[i]
            if val > remaining:
                break
            
            # Check if adding this element maintains antichain property
            valid = True
            for j in chosen:
                if incompatible[i][j]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Pruning: check if remaining can still be achieved
            # The minimum we can add with (k - len(chosen) - 1) more elements
            # is the sum of the smallest (k - len(chosen) - 1) smooth numbers > smooth_numbers[i]
            # But this is complex; skip for now
            
            chosen.append(i)
            result = backtrack(i + 1, remaining - val, chosen)
            if result is not None:
                return [smooth_numbers[x] for x in result]
            chosen.pop()
        
        return None
    
    result = backtrack(0, target, [])
    return result

def _finalize(best_answers, smooth_numbers, incompatible):
    """Finalize answers, ensuring all targets have valid representations."""
    final_answers = []
    for n in range(1, 129):
        if best_answers[n] is not None:
            final_answers.append(best_answers[n])
        else:
            # Fallback: find any valid antichain
            found = False
            for k in range(1, 11):
                ans = _find_antichain_backtrack(n, smooth_numbers, incompatible, len(smooth_numbers))
                if ans is not None:
                    final_answers.append(ans)
                    found = True
                    break
            if not found:
                # This should not happen, but handle gracefully
                # Find any valid representation
                for k in range(1, 15):
                    for combo in itertools.combinations([x for x in smooth_numbers if x <= n], k):
                        if sum(combo) == n:
                            # Check antichain
                            valid = True
                            for a in range(k):
                                for b in range(a+1, k):
                                    # Find indices
                                    ia = smooth_numbers.index(combo[a])
                                    ib = smooth_numbers.index(combo[b])
                                    if incompatible[ia][ib]:
                                        valid = False
                                        break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if valid:
                                final_answers.append(list(combo))
                                found = True
                                break
                    if found:
                        break
                if not found:
                    # Last resort: should not happen
                    final_answers.append([n])
    return final_answers

# EVOLVE-BLOCK-END
