import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    N = 255
    
    # The set of all numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    # Candidates must be divisible by 3, 5, or 17.
    
    # Helper to check if a number is compatible with a set
    def is_compatible(num, current_set):
        for x in current_set:
            if math.gcd(num, x) == 1:
                return False
        return True

    # Strategy 1: Base set of multiples of 3
    # This is a known large clique.
    base_set = set(x for x in range(1, N + 1) if x % 3 == 0)
    
    best_set = set(base_set)
    best_size = len(best_set)
    
    # Strategy 2: Local Search / Hill Climbing
    # Try to improve the base set by swapping elements
    # We iterate through elements in the current set, try removing one,
    # and then try to add as many compatible candidates as possible.
    
    # Candidates that are NOT in the base set
    # They must be divisible by 5 or 17 (since if div by 3, they are in base)
    # Actually, any number not in base_set but valid candidate (div by 5 or 17)
    candidates = [x for x in range(1, N + 1) if x not in base_set and (x % 5 == 0 or x % 17 == 0)]
    
    # To make local search faster, we can precompute which candidates are compatible with which base elements
    # But N is small (255), so O(N^2) is fine.
    
    current_set = set(base_set)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try removing each element in the current set
        for x in list(current_set):
            if time.time() > deadline - 5:
                break
            
            # Remove x
            new_set = set(current_set)
            new_set.remove(x)
            
            # Try to add candidates
            # Sort candidates by "potential" (e.g., number of factors) to find good additions quickly?
            # Or just try all.
            added_any = False
            # We need to check compatibility with the ENTIRE new_set
            for y in candidates:
                if y in new_set:
                    continue
                if is_compatible(y, new_set):
                    new_set.add(y)
                    added_any = True
            
            # If we added more than we removed (1), size increases
            # If we added exactly 1, size is same (no gain, but might enable future gains? 
            # Usually, if size doesn't increase, we don't swap unless it's a tie-breaking heuristic)
            if len(new_set) > len(current_set):
                current_set = new_set
                improved = True
                best_set = set(current_set)
                best_size = len(current_set)
                # Update candidates list? The set of candidates is fixed (all valid numbers not in initial base).
                # But some candidates might have been added to current_set.
                # The loop `if y in new_set` handles this.
                
    # Strategy 3: Greedy Construction from Scratch
    # Try different orderings to see if we can find a larger clique
    all_candidates = [x for x in range(1, N + 1) if math.gcd(x, 255) > 1]
    
    # Precompute gcd matrix for speed in greedy
    n_cand = len(all_candidates)
    gcd_matrix = [[0]*n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i, n_cand):
            gcd_matrix[i][j] = math.gcd(all_candidates[i], all_candidates[j])
            gcd_matrix[j][i] = gcd_matrix[i][j]
            
    # Index of 255
    idx_255 = all_candidates.index(255)
    
    # Try a few greedy strategies
    # 1. Most connected first
    # 2. Ascending
    # 3. Descending
    
    orderings = [
        lambda idx: -sum(1 for j in range(n_cand) if gcd_matrix[idx][j] > 1),
        lambda idx: all_candidates[idx],
        lambda idx: -all_candidates[idx]
    ]
    
    for order_func in orderings:
        if time.time() > deadline - 10:
            break
            
        current = {idx_255}
        remaining = set(range(n_cand)) - current
        
        while remaining:
            if time.time() > deadline - 10:
                break
            best_v = -1
            best_score = -1
            for v in remaining:
                # Score: number of connections to current set
                score = sum(1 for u in current if gcd_matrix[v][u] > 1)
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v == -1 or best_score == 0:
                break
            current.add(best_v)
            remaining.remove(best_v)
            
        # Convert to values
        current_vals = set(all_candidates[i] for i in current)
        if len(current_vals) > best_size:
            best_size = len(current_vals)
            best_set = current_vals

    # Final Verification
    # Ensure 255 is in the set
    if 255 not in best_set:
        best_set.add(255)
        
    # Ensure all pairs are non-coprime
    valid = True
    vals = list(best_set)
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            if math.gcd(vals[i], vals[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to the safe, known valid set of multiples of 3
        return sorted(x for x in range(1, N + 1) if x % 3 == 0)
        
    return sorted(best_set)

construct()
