import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s out of 60s, safety margin
    
    n = 20
    max_val = 10000
    
    def compute_sets(A):
        """Compute sumset and product set for a list of n distinct positive integers."""
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(n):
                sums.add(ai + A[j])
                prods.add(ai * A[j])
        return sums, prods
    
    def score(A):
        """Compute max(|A+A|, |A*A|) for a valid set of n distinct positive integers <= max_val."""
        if len(set(A)) != n or max(A) > max_val or min(A) < 1:
            return float('inf')
        sums, prods = compute_sets(A)
        return max(len(sums), len(prods))
    
    def incremental_score(A, idx, new_val):
        """
        Compute the new score if A[idx] is replaced by new_val.
        Assumes A is currently valid and new_val is not in A (except at idx).
        Uses the existing sumset/productset to compute only the delta.
        """
        old_val = A[idx]
        if new_val == old_val:
            return score(A)
        
        # Check validity
        if new_val < 1 or new_val > max_val:
            return float('inf')
        if new_val in A and A.index(new_val) != idx:
            return float('inf')
        
        # Get current sets
        sums, prods = compute_sets(A)
        
        # Remove pairs involving old_val
        for j in range(n):
            sums.discard(old_val + A[j])
            prods.discard(old_val * A[j])
        
        # Add pairs involving new_val
        for j in range(n):
            sums.add(new_val + A[j])
            prods.add(new_val * A[j])
        
        return max(len(sums), len(prods))
    
    # Helper to generate GP
    def gen_gp(ratio, start, length):
        A = []
        val = start
        for _ in range(length):
            A.append(val)
            val *= ratio
        return A
    
    best_score = float('inf')
    best_set = list(range(1, n+1))
    
    # Candidate 1: Single GP
    for ratio in [2, 3]:
        for start in [1, 2]:
            A = gen_gp(ratio, start, n)
            if max(A) <= max_val:
                s = score(A)
                if s < best_score:
                    best_score = s
                    best_set = A
    
    # Candidate 2: Two clusters GP
    for ratio in [2, 3]:
        for start1 in [1, 2]:
            for start2 in [3, 4, 5, 10]:
                A1 = gen_gp(ratio, start1, 10)
                A2 = gen_gp(ratio, start2, 10)
                A = A1 + A2
                if len(set(A)) == n and max(A) <= max_val:
                    s = score(A)
                    if s < best_score:
                        best_score = s
                        best_set = A
    
    # Local search with incremental scoring
    def local_search(A, time_limit):
        A = list(A)
        current_score = score(A)
        
        # Precompute current sets for incremental updates
        sums, prods = compute_sets(A)
        
        while time.time() < time_limit:
            improved = False
            i = random.randint(0, n-1)
            old_val = A[i]
            
            # Try replacing A[i] with a new value
            # First try values near the current value
            candidates = []
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                cand = old_val + delta
                if 1 <= cand <= max_val and cand != old_val:
                    candidates.append(cand)
            
            # Add some random candidates
            for _ in range(10):
                candidates.append(random.randint(1, max_val))
            
            for new_val in candidates:
                if new_val < 1 or new_val > max_val:
                    continue
                if new_val in A and A.index(new_val) != i:
                    continue
                
                # Incremental score computation
                # Remove pairs involving old_val
                sums_copy = sums.copy()
                prods_copy = prods.copy()
                for j in range(n):
                    sums_copy.discard(old_val + A[j])
                    prods_copy.discard(old_val * A[j])
                
                # Add pairs involving new_val
                for j in range(n):
                    sums_copy.add(new_val + A[j])
                    prods_copy.add(new_val * A[j])
                
                new_score = max(len(sums_copy), len(prods_copy))
                
                if new_score < current_score:
                    # Update
                    A[i] = new_val
                    current_score = new_score
                    sums = sums_copy
                    prods = prods_copy
                    improved = True
                    break
            
            # If no single replacement improved, try a random pair replacement
            if not improved:
                j = random.randint(0, n-1)
                if j == i:
                    continue
                old_val2 = A[j]
                
                new_val1 = random.randint(1, max_val)
                new_val2 = random.randint(1, max_val)
                
                # Check for duplicates
                if new_val1 in A and A.index(new_val1) != i:
                    continue
                if new_val2 in A and A.index(new_val2) != j:
                    continue
                if new_val1 == new_val2:
                    continue
                
                # Incremental score computation for pair replacement
                sums_copy = sums.copy()
                prods_copy = prods.copy()
                
                # Remove pairs involving old_val and old_val2
                for k in range(n):
                    sums_copy.discard(old_val + A[k])
                    prods_copy.discard(old_val * A[k])
                    sums_copy.discard(old_val2 + A[k])
                    prods_copy.discard(old_val2 * A[k])
                
                # Add pairs involving new_val1 and new_val2
                # First, temporarily set the new values
                A[i] = new_val1
                A[j] = new_val2
                
                for k in range(n):
                    sums_copy.add(new_val1 + A[k])
                    prods_copy.add(new_val1 * A[k])
                    sums_copy.add(new_val2 + A[k])
                    prods_copy.add(new_val2 * A[k])
                
                new_score = max(len(sums_copy), len(prods_copy))
                
                if new_score < current_score:
                    current_score = new_score
                    sums = sums_copy
                    prods = prods_copy
                    improved = True
                else:
                    # Revert
                    A[i] = old_val
                    A[j] = old_val2
            
            if time.time() > time_limit - 2:
                break
        
        return A, current_score

    # Run local search on best found so far
    if time.time() < deadline - 5:
        best_set, best_score = local_search(best_set, deadline - 5)
        
    return best_set
