import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # 1. Generate 2,3,5-smooth numbers up to 128
    smooth = set()
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.add(x)
    
    smooth_list = sorted(smooth)
    # Map value to index for quick checking? Or just use values.
    # Values are unique.
    
    # Precompute optimal antichain representations for sums 1..128
    # best_rep[sum] = (length, tuple)
    # We want to minimize length.
    # Initialize with infinity
    INF = 100
    best_len = [INF] * 129
    best_terms = [None] * 129
    
    # best_len[0] = 0, best_terms[0] = ()
    best_len[0] = 0
    best_terms[0] = ()
    
    # We will use DFS to generate all valid antichains with sum <= 128.
    # To avoid generating too many, we prune if sum > 128.
    # Also, we can prune if current length >= best_len[sum] - (max possible reduction?)
    # Actually, just generating all is likely fine given the small constraint.
    # Number of smooth numbers is 37.
    # Antichains in this poset are limited.
    
    def dfs(start_idx, current_sum, current_terms):
        # current_terms is a tuple of values, sorted
        if current_sum > 128:
            return
            
        # Update best if we have a better representation for this sum
        if len(current_terms) < best_len[current_sum]:
            best_len[current_sum] = len(current_terms)
            best_terms[current_sum] = current_terms
            
        # Try adding more terms
        # We must pick terms > last term to avoid duplicates/permutations
        # And terms must not be divisible by any term in current_terms
        # Since we pick in increasing order, we only need to check if any term in current_terms divides the candidate.
        
        last_val = current_terms[-1] if current_terms else 0
        
        # Iterate through smooth_list from start_idx
        # We can start from start_idx to ensure increasing order
        for i in range(start_idx, len(smooth_list)):
            val = smooth_list[i]
            if val <= last_val:
                continue
            new_sum = current_sum + val
            if new_sum > 128:
                break # Since smooth_list is sorted, further vals will be larger
            
            # Check antichain condition: val must not be divisible by any in current_terms
            # Since val > all in current_terms, we only check if any y in current_terms divides val
            valid = True
            for y in current_terms:
                if val % y == 0:
                    valid = False
                    break
            
            if valid:
                dfs(i + 1, new_sum, current_terms + (val,))

    dfs(0, 0, ())
    
    # Construct answers
    answers = []
    for n in range(1, 129):
        if best_terms[n] is not None:
            answers.append(list(best_terms[n]))
        else:
            # Fallback: should not happen, but if it does, find a valid one via brute force search
            # This indicates a bug in precomputation or constraint issue
            # For safety, do a quick search
            found = None
            # Try sizes 1 to 10
            candidates = [x for x in smooth_list if x < n]
            for k in range(1, 11):
                import itertools
                for combo in itertools.combinations(candidates, k):
                    if sum(combo) == n:
                        # Check antichain
                        ok = True
                        for i in range(len(combo)):
                            for j in range(i+1, len(combo)):
                                a, b = combo[i], combo[j]
                                if a > b: a, b = b, a
                                if b % a == 0:
                                    ok = False
                                    break
                            if not ok: break
                        if ok:
                            found = list(combo)
                            break
                if found:
                    break
            if found:
                answers.append(found)
            else:
                # Extreme fallback: return empty list (will fail validation but avoid crash)
                # Or raise error? Task requires valid.
                # If precomputation failed, something is wrong.
                answers.append([])

    return answers
