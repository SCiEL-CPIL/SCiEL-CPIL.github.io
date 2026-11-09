import time

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
    
    # Precompute compatibility matrix
    # compatible[i][j] = True if smooth_list[i] and smooth_list[j] can coexist in an antichain
    # i.e., neither divides the other.
    compatible = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_list[i], smooth_list[j]
            if a % b == 0 or b % a == 0:
                compatible[i][j] = False
                compatible[j][i] = False
            
    # Map smooth value to its index
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    def find_min_antichain(n):
        """
        Finds the minimum number of terms and the terms themselves for target n.
        Returns (k, terms) or None if not found.
        """
        # Get candidate indices: smooth numbers <= n
        # Sorted ascending by value
        candidates = [i for i in range(n_smooth) if smooth_list[i] <= n]
        
        # If n itself is smooth, k=1 is possible
        if n in val_to_idx:
            return 1, [n]
            
        # Try k = 2, 3, ...
        # Max k is bounded by the number of candidates
        for k in range(2, len(candidates) + 1):
            # Pruning: Is it possible to sum k distinct numbers from candidates to n?
            # Min sum of k smallest
            # Max sum of k largest
            
            # candidates are indices into smooth_list (which is sorted ascending)
            # So candidates[0] is the smallest index in the filtered list?
            # No, candidates is a list of indices. 
            # Since we iterated i from 0 to n_smooth, candidates is in ascending order of index.
            # And smooth_list is sorted ascending.
            # So smooth_list[candidates[0]] <= smooth_list[candidates[1]] ...
            
            k_smallest_vals = [smooth_list[candidates[i]] for i in range(k)]
            min_sum = sum(k_smallest_vals)
            if min_sum > n:
                break # No solution for this k or larger k (since min sum grows with k)
                
            k_largest_vals = [smooth_list[candidates[i]] for i in range(len(candidates)-k, len(candidates))]
            max_sum = sum(k_largest_vals)
            if max_sum < n:
                continue # Need more terms to reach n
            
            # Backtracking search for a valid antichain of size k
            found_terms = None
            path = [] # List of indices
            
            def backtrack(start_idx, remaining_sum, remaining_k):
                if remaining_k == 0:
                    if remaining_sum == 0:
                        return [smooth_list[idx] for idx in path]
                    return None
                
                avail_count = len(candidates) - start_idx
                if avail_count < remaining_k:
                    return None
                
                # Pruning:
                # Max possible sum: sum of largest 'remaining_k' from candidates[start_idx:]
                # Min possible sum: sum of smallest 'remaining_k' from candidates[start_idx:]
                
                # Candidates are sorted ascending.
                # Largest remaining_k in candidates[start_idx:]:
                # They are at the end of the list.
                max_possible = sum(smooth_list[candidates[i]] for i in range(len(candidates)-remaining_k, len(candidates)))
                if max_possible < remaining_sum:
                    return None
                
                min_possible = sum(smooth_list[candidates[i]] for i in range(start_idx, start_idx + remaining_k))
                if min_possible > remaining_sum:
                    return None
                
                for i in range(start_idx, len(candidates)):
                    idx = candidates[i]
                    num = smooth_list[idx]
                    
                    if num > remaining_sum:
                        break # Since sorted ascending, no larger num will fit
                    
                    # Check compatibility with all previously chosen
                    is_compatible = True
                    for prev_idx in path:
                        if not compatible[prev_idx][idx]:
                            is_compatible = False
                            break
                    
                    if is_compatible:
                        path.append(idx)
                        result = backtrack(i + 1, remaining_sum - num, remaining_k - 1)
                        if result is not None:
                            return result
                        path.pop()
                
                return None
            
            found_terms = backtrack(0, n, k)
            
            if found_terms is not None:
                return k, found_terms
                
        # Fallback if not found (should not happen for 1..128)
        # This case should never be reached as all 1..128 are representable.
        return 1, [n] if n in val_to_idx else [1, n-1]

    answers = []
    for n in range(1, 129):
        # Check time
        if time.time() - start_time > DEADLINE:
             # If we run out of time, we need to return *something* valid.
             # We can't just break, we need 128 rows.
             # However, the algorithm is very fast.
             pass
        
        k, terms = find_min_antichain(n)
        answers.append(terms)
        
    return answers
