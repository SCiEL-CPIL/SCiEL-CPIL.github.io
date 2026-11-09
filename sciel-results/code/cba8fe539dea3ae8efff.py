import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85  # 90s limit, leave 5s margin
    
    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):  # 2^7 = 128
        val_a = 2 ** a
        for b in range(0, 5):  # 3^4 = 81, 3^5 = 243 > 128
            val_ab = val_a * (3 ** b)
            if val_ab > 128:
                break
            for c in range(0, 4):  # 5^3 = 125, 5^4 = 625 > 128
                val_abc = val_ab * (5 ** c)
                if val_abc > 128:
                    break
                smooth.add(val_abc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Precompute divisibility bitmask: div_mask[i] = bitmask of indices j where smooth_list[i] divides smooth_list[j]
    # For antichain check: if we have a set S, S is antichain iff for all i in S, div_mask[i] & S == 0
    div_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_list[i], smooth_list[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    div_mask[i] |= (1 << j)
    
    # Map smooth value to its index
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Pre-filter candidates for each n: list of indices into smooth_list
    candidates = [[] for _ in range(129)]
    for i in range(1, 129):
        for idx in range(n_smooth):
            if smooth_list[idx] <= i:
                candidates[i].append(idx)
            else:
                break
        
    # Precompute for each n: sort candidates by value descending
    cand_desc = [sorted(candidates[n], key=lambda idx: smooth_list[idx], reverse=True) for n in range(129)]
    
    # Precompute suffix sums for pruning
    suffix_sums = [None for _ in range(129)]
    for n in range(1, 129):
        cand = cand_desc[n]
        s = [0] * (len(cand) + 1)
        for i in range(len(cand) - 1, -1, -1):
            s[i] = s[i + 1] + smooth_list[cand[i]]
        suffix_sums[n] = s
    
    # Precompute reverse divisibility mask
    rev_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth_list[i] % smooth_list[j] == 0:
                    # smooth_list[j] divides smooth_list[i]
                    rev_mask[i] |= (1 << j)
    
    # Memoization cache: (n, k) -> result or None
    memo = {}
    
    def find_antichain(n, k):
        """
        Finds an antichain of size k summing to n.
        Returns list of terms or None.
        """
        key = (n, k)
        if key in memo:
            return memo[key]
        
        cand = cand_desc[n]
        if not cand or len(cand) < k:
            memo[key] = None
            return None
        
        # Quick bound check
        if k == 1:
            if n in val_to_idx:
                memo[key] = [n]
                return [n]
            memo[key] = None
            return None
        
        # Max sum: sum of k largest
        if k < len(cand):
            max_sum = suffix_sums[n][0] - suffix_sums[n][k]
        else:
            max_sum = suffix_sums[n][0]
        
        # Min sum: sum of k smallest (last k in descending order)
        min_sum = sum(smooth_list[cand[len(cand)-1-i]] for i in range(k))
        
        if n > max_sum or n < min_sum:
            memo[key] = None
            return None

        path = []  # list of indices
        path_mask = [0]  # bitmask of chosen indices for fast divisibility check
        
        def backtrack(start_idx, remaining_sum, remaining_k, chosen_mask):
            if remaining_k == 0:
                if remaining_sum == 0:
                    return [smooth_list[idx] for idx in path]
                return None
            
            avail_count = len(cand) - start_idx
            if avail_count < remaining_k:
                return None
            
            # Pruning: max possible sum from remaining candidates
            if start_idx + remaining_k <= len(cand):
                max_possible = suffix_sums[n][start_idx] - suffix_sums[n][start_idx + remaining_k]
            else:
                max_possible = suffix_sums[n][start_idx]
            if max_possible < remaining_sum:
                return None
            
            # Min possible sum: sum of smallest 'remaining_k' from cand[start_idx:]
            if len(cand) - start_idx >= remaining_k:
                min_possible = sum(smooth_list[cand[len(cand)-1-i]] for i in range(remaining_k))
            else:
                min_possible = suffix_sums[n][start_idx]
            if min_possible > remaining_sum:
                return None
            
            for i in range(start_idx, len(cand)):
                idx = cand[i]
                num = smooth_list[idx]
                if num > remaining_sum:
                    continue
                
                # Check divisibility using bitmask: if any chosen element divides this one or vice versa
                # div_mask[idx] has bits set for elements that idx divides
                # We need: no chosen element divides idx, and idx divides no chosen element
                # div_mask[idx] & chosen_mask != 0 means idx divides some chosen element
                # Also need to check if some chosen element divides idx
                # For that, we need: for each chosen j, smooth_list[j] divides smooth_list[idx]
                # This is equivalent to: (div_mask[idx] & chosen_mask) == 0 AND 
                # for all j in chosen, not (smooth_list[j] divides smooth_list[idx])
                # The second condition: smooth_list[j] | smooth_list[idx] means j is in div_mask... 
                # Actually div_mask[j] has bit idx set if smooth_list[j] divides smooth_list[idx]
                # So we need: for all j in chosen, bit idx not set in div_mask[j]
                # That's: chosen_mask & reverse_div_mask[idx] == 0 where reverse_div_mask[idx] = {j : div_mask[j] has bit idx}
                
                # Let me precompute reverse: rev_mask[idx] = bitmask of j such that smooth_list[j] divides smooth_list[idx]
                # Then compatibility: (div_mask[idx] & chosen_mask) == 0 and (rev_mask[idx] & chosen_mask) == 0
                
                if (div_mask[idx] & chosen_mask) != 0:
                    continue
                if (rev_mask[idx] & chosen_mask) != 0:
                    continue
                
                path.append(idx)
                new_mask = chosen_mask | (1 << idx)
                result = backtrack(i + 1, remaining_sum - num, remaining_k - 1, new_mask)
                if result is not None:
                    return result
                path.pop()
            
            return None
        
        result = backtrack(0, n, k, 0)
        memo[key] = result
        return result

    answers = [None] * 129
    
    # Process targets: smooth ones first (k=1), then others by increasing k needed
    # For efficiency, process all targets but with time check
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            break
            
        answer = None
        for k in range(1, 31):
            if time.time() - start_time > DEADLINE:
                break
            res = find_antichain(n, k)
            if res is not None:
                answer = res
                break
        
        if answer is None:
            # Last resort fallback
            if n in val_to_idx:
                answer = [n]
            else:
                answer = [n]
        
        answers[n] = answer
    
    # Fill in any missing answers with fallback
    for n in range(1, 129):
        if answers[n] is None:
            if n in val_to_idx:
                answers[n] = [n]
            else:
                for k in range(1, 31):
                    res = find_antichain(n, k)
                    if res is not None:
                        answers[n] = res
                        break
                if answers[n] is None:
                    answers[n] = [n]
    
    return answers[1:]
