# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    smooth_idx = {v: i for i, v in enumerate(smooth)}
    n_smooth = len(smooth)
    
    # Precompute divisibility: divs[i] = set of indices j where smooth[i] divides smooth[j]
    divs = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0:
                divs[i].add(j)
                divs[j].add(i)
    
    # Precompute compatible pairs: compat[i] = set of indices j > i where neither divides the other
    compat = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if j not in divs[i]:
                compat[i].add(j)
                compat[j].add(i)
    
    def find_min_rep(n, max_k=30):
        """Find representation of n with minimum number of terms forming an antichain."""
        # Try k=1
        if n in smooth_set:
            return [n]
        
        # Try k=2
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                j = smooth_idx[b]
                if j not in divs[i]:
                    return [a, b]
        
        # Try k=3
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    k = smooth_idx[c]
                    if k not in divs[i] and k not in divs[j] and j not in divs[i]:
                        return [a, b, c]
        
        # Try k=4
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d in smooth_set:
                        m = smooth_idx[d]
                        if (m not in divs[i] and m not in divs[j] and 
                            m not in divs[k] and k not in divs[i] and 
                            k not in divs[j] and j not in divs[i]):
                            return [a, b, c, d]
        
        # Try k=5 with improved backtracking using compat sets
        def backtrack5(target, start_idx, current_sum, current_list, current_indices, depth_limit):
            if current_sum == target:
                return current_list[:]
            if current_sum > target:
                return None
            if len(current_list) >= depth_limit:
                return None
            if start_idx >= n_smooth:
                return None
            
            remaining = target - current_sum
            
            for idx in range(start_idx, n_smooth):
                term = smooth[idx]
                if current_sum + term > target:
                    break
                
                # Check antichain with existing using precomputed compat
                valid = True
                for existing_idx in current_indices:
                    if idx not in compat[existing_idx]:
                        valid = False
                        break
                if not valid:
                    continue
                
                current_list.append(term)
                current_indices.append(idx)
                res = backtrack5(target, idx + 1, current_sum + term, current_list, current_indices, depth_limit)
                if res:
                    return res
                current_list.pop()
                current_indices.pop()
            
            return None
        
        # Try k=5
        res = backtrack5(n, 0, 0, [], [], 5)
        if res:
            return res
        
        # Try k=6
        res = backtrack5(n, 0, 0, [], [], 6)
        if res:
            return res
        
        # Try k=7
        res = backtrack5(n, 0, 0, [], [], 7)
        if res:
            return res
        
        # Try k=8
        res = backtrack5(n, 0, 0, [], [], 8)
        if res:
            return res
        
        # Fallback: try up to 12 terms
        res = backtrack5(n, 0, 0, [], [], 12)
        if res:
            return res
        
        # Last resort fallback
        return [n] if n in smooth_set else [1]
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline - 2:
            # Time critical: use a fast fallback
            if n in smooth_set:
                answers.append([n])
            else:
                found = None
                for a in smooth:
                    if a >= n:
                        break
                    b = n - a
                    if b > a and b in smooth_set:
                        j = smooth_idx[b]
                        i = smooth_idx[a]
                        if j not in divs[i]:
                            found = [a, b]
                            break
                if found is None:
                    def quick_backtrack(target, start_idx, current_sum, current_list, current_indices):
                        if current_sum == target:
                            return current_list[:]
                        if current_sum > target:
                            return None
                        if len(current_list) >= 10:
                            return None
                        for idx in range(start_idx, n_smooth):
                            term = smooth[idx]
                            if current_sum + term > target:
                                break
                            valid = True
                            for existing_idx in current_indices:
                                if idx not in compat[existing_idx]:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            current_list.append(term)
                            current_indices.append(idx)
                            res = quick_backtrack(target, idx + 1, current_sum + term, current_list, current_indices)
                            if res:
                                return res
                            current_list.pop()
                            current_indices.pop()
                        return None
                    found = quick_backtrack(n, 0, 0, [], [])
                    if found is None:
                        found = [n] if n in smooth_set else [1]
                answers.append(found)
            continue
        
        found = find_min_rep(n)
        if found is None:
            found = [n] if n in smooth_set else [1]
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
