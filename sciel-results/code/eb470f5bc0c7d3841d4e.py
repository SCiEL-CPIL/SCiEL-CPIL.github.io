import time

def _is_smooth(n):
    y = n
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _precompute():
    # 1. Generate smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    n_smooth = len(smooth)
    
    # 2. Precompute compatibility
    # comp[i][j] is True if smooth[i] and smooth[j] are compatible (antichain pair)
    comp = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth[j] % smooth[i] == 0 or smooth[i] % smooth[j] == 0:
                comp[i][j] = False
                comp[j][i] = False
                
    # 3. Find minimal representation for each target 1..128
    answers = [None] * 129
    
    # Helper to check if a subset of indices is a valid antichain
    # We will do this incrementally in DFS
    
    def find_min_rep(t):
        # Try k=1
        for i in range(n_smooth):
            if smooth[i] == t:
                return [smooth[i]]
        
        # Try k=2, 3, ...
        # Max k is likely small. Let's try up to 10 to be safe, though 5 is probably enough.
        for k in range(2, 11):
            # DFS
            found = None
            # Pruning helper: min sum of 'count' distinct numbers from smooth[start_idx:]
            # We can precompute suffix sums of smallest 'm' elements? 
            # Or just compute on the fly. Since n_smooth is 38, it's fast.
            
            def dfs(start_idx, current_sum, chosen_indices):
                nonlocal found
                if found is not None:
                    return
                terms_needed = k - len(chosen_indices)
                remaining_sum = t - current_sum
                
                if terms_needed == 0:
                    if remaining_sum == 0:
                        found = [smooth[i] for i in chosen_indices]
                    return
                
                if terms_needed < 0 or remaining_sum < 0:
                    return
                    
                # Not enough numbers left
                if start_idx + terms_needed > n_smooth:
                    return
                
                # Pruning: Minimum possible sum of 'terms_needed' distinct numbers 
                # from smooth[start_idx:] must be <= remaining_sum
                # The smallest 'terms_needed' numbers are smooth[start_idx : start_idx + terms_needed]
                min_possible_sum = sum(smooth[start_idx : start_idx + terms_needed])
                if min_possible_sum > remaining_sum:
                    return
                
                # Iterate next index
                for next_idx in range(start_idx, n_smooth):
                    # If this number alone is too big, break
                    if smooth[next_idx] > remaining_sum:
                        break
                    
                    # Check compatibility with all chosen
                    compatible = True
                    for c_idx in chosen_indices:
                        if not comp[c_idx][next_idx]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    chosen_indices.append(next_idx)
                    dfs(next_idx + 1, current_sum + smooth[next_idx], chosen_indices)
                    chosen_indices.pop()
                    if found is not None:
                        return
            
            dfs(0, 0, [])
            if found is not None:
                return found
        
        # If not found (should not happen for 1..128 with k up to 10), return None
        return None

    for t in range(1, 129):
        answers[t] = find_min_rep(t)
        
    return answers

# Precompute at module load
_ANSWERS = _precompute()

def construct():
    return _ANSWERS[1:129]
