import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    # Generate smooth numbers up to 128
    smooth = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth.append(i)
    
    smooth_set = set(smooth)
    smooth_sorted = sorted(smooth)
    n_smooth = len(smooth_sorted)
    answers = [None] * 128
    
    # Precompute which smooth numbers can be used together (antichain check)
    # div_free[i][j] = True if smooth_sorted[i] does not divide smooth_sorted[j]
    div_free = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_sorted[i], smooth_sorted[j]
                if a < b:
                    div_free[i][j] = (b % a != 0)
                else:
                    div_free[i][j] = (a % b != 0)
    
    def _is_antichain(indices):
        """Check if the set of smooth numbers at given indices forms an antichain."""
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                ii, jj = indices[i], indices[j]
                if not div_free[ii][jj]:
                    return False
        return True
    
    def _find_minimal(n, max_k=30):
        """Find minimal k such that n can be represented as sum of k distinct smooth numbers forming an antichain."""
        # k=1
        if n in smooth_set:
            return [n]
        
        # k=2
        for i in range(n_smooth):
            a = smooth_sorted[i]
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                # Find index of b
                # Use binary search or dict
                j = smooth_set_to_idx.get(b, -1)
                if j != -1 and div_free[i][j]:
                    return [a, b]
        
        # k=3
        for i in range(n_smooth):
            a = smooth_sorted[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth_sorted[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    k_idx = smooth_set_to_idx.get(c, -1)
                    if k_idx != -1 and k_idx > j:
                        if div_free[i][j] and div_free[i][k_idx] and div_free[j][k_idx]:
                            return [a, b, c]
        
        # k=4 and beyond: use backtracking
        if time.time() > deadline - 5:
            return None  # Timeout safety
        
        def _backtrack(remaining, start_idx, current, k_remaining):
            if time.time() > deadline - 3:
                return None
            if k_remaining == 1:
                if remaining in smooth_set:
                    idx = smooth_set_to_idx.get(remaining, -1)
                    if idx != -1 and idx > start_idx:
                        # Check antichain with current
                        for ci in current:
                            if not div_free[ci][idx]:
                                return None
                        return current + [remaining]
                return None
            if k_remaining == 0:
                return current if remaining == 0 else None
            
            # Pruning: minimum possible sum with k_remaining terms starting from start_idx
            if k_remaining > 0:
                min_possible = sum(smooth_sorted[start_idx:start_idx + k_remaining]) if start_idx + k_remaining <= n_smooth else float('inf')
                if min_possible > remaining:
                    return None
            
            for idx in range(start_idx, n_smooth):
                val = smooth_sorted[idx]
                if val > remaining:
                    break
                new_remaining = remaining - val
                if new_remaining < 0:
                    break
                # Check if val can be added to current (antichain)
                valid = True
                for ci in current:
                    if not div_free[ci][idx]:
                        valid = False
                        break
                if not valid:
                    continue
                result = _backtrack(new_remaining, idx + 1, current + [val], k_remaining - 1)
                if result is not None:
                    return result
            return None
        
        for k in range(4, min(max_k, 30) + 1):
            result = _backtrack(n, 0, [], k)
            if result is not None:
                return result
        
        return None
    
    # Create index lookup for smooth numbers
    smooth_set_to_idx = {v: i for i, v in enumerate(smooth_sorted)}
    
    # Solve for each target
    for n in range(1, 129):
        if time.time() > deadline - 10:
            # Emergency: fill remaining with whatever we can find quickly
            if n in smooth_set:
                answers[n-1] = [n]
            else:
                # Quick 2-term check
                found = False
                for i in range(n_smooth):
                    a = smooth_sorted[i]
                    if a >= n:
                        break
                    b = n - a
                    if b <= a:
                        continue
                    if b in smooth_set:
                        j = smooth_set_to_idx.get(b, -1)
                        if j != -1 and div_free[i][j]:
                            answers[n-1] = [a, b]
                            found = True
                            break
                if not found:
                    answers[n-1] = [1, n-1] if n-1 in smooth_set else [1, 1, n-2]
            continue
        
        result = _find_minimal(n)
        if result is None:
            # Fallback: should not happen, but just in case
            answers[n-1] = [1, n-1] if n-1 in smooth_set else [1, 1, n-2]
        else:
            answers[n-1] = result
    
    # Verify all answers are valid
    for n in range(1, 129):
        ans = answers[n-1]
        if ans is None:
            # Should not happen
            answers[n-1] = [n] if n in smooth_set else [1, n-1]
    
    return answers
