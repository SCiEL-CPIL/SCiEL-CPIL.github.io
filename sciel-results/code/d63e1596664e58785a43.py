import numpy as np
import random

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    
    def get_representations(x):
        # x is boolean array of length 129
        conv = np.convolve(x, x, mode='full')
        # conv[k] is ordered count
        R = np.zeros(len(targets))
        for i, t in enumerate(targets):
            if t % 2 == 0:
                mid = t // 2
                # conv[t] = 2 * pairs_distinct + x[mid]
                # R = pairs_distinct + x[mid]
                # pairs_distinct = (conv[t] - x[mid]) / 2
                # R = (conv[t] - x[mid]) / 2 + x[mid] = (conv[t] + x[mid]) / 2
                R[i] = (conv[t] + x[mid]) / 2.0
            else:
                # conv[t] = 2 * pairs
                # R = pairs = conv[t] / 2
                R[i] = conv[t] / 2.0
        return R

    def get_cost(x, R):
        # Penalty for missing coverage
        missing = np.sum(R < 1)
        if missing > 0:
            return 1000000 + missing * 1000 + np.max(R)
        
        peak = np.max(R)
        # Secondary objective: minimize sum of squares to flatten distribution
        sq_sum = np.sum(R**2)
        return peak * 1000 + sq_sum

    # 1. Greedy Construction
    # Start with empty set
    A = set()
    x = np.zeros(n, dtype=np.int8)
    uncovered = set(range(target_min, target_max + 1))
    
    # To speed up greedy, we can start with a few random elements or just run greedy from scratch
    # Greedy from scratch might be slow if we check all 129 candidates * iterations.
    # Iterations ~ 50. 129 * 50 * (cost of checking coverage)
    # Checking coverage: for each candidate c, we need to know which uncovered targets are covered by c+a for a in A.
    # This is a set intersection.
    
    max_iterations = 100
    for _ in range(max_iterations):
        if not uncovered:
            break
            
        best_c = -1
        best_gain = 0
        
        for c in range(n):
            if c in A:
                continue
            
            # Calculate new covered targets
            # New pairs involving c: {c+a | a in A} and {2c}
            # We only care about targets in 'uncovered'
            
            # Optimization: Precompute sums?
            # For each a in A, s = c + a. If s in uncovered, it's a gain.
            # Also 2c.
            
            gain = 0
            # Check 2c
            if 2 * c in uncovered:
                gain += 1
                
            for a in A:
                s = c + a
                if s in uncovered:
                    gain += 1
                    # Note: if multiple a give same s, we only count it once for 'uncovered'
                    # But 'uncovered' is a set, so we should check membership.
                    # To avoid double counting same s from different a, we should track which uncovered targets are hit.
                    # However, for greedy selection, overestimating gain is often okay, but let's be precise.
                    # Actually, if c+a1 == c+a2, then a1==a2. So sums are distinct for distinct a.
                    # So gain is just count of a in A such that c+a in uncovered, plus 1 if 2c in uncovered.
            
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1 or best_gain == 0:
            # If we can't cover more, we might be stuck? 
            # If A is empty, gain is always 0. 
            # We need to add at least 2 elements to form a sum.
            # If A is empty, add 0 and 1? Or just random.
            if not A:
                # Add two elements that might help
                # e.g., 64 and 64? No, distinct elements in set.
                # Add 0 and 128? Sum 128.
                A.add(0)
                A.add(128)
                x[0] = 1
                x[128] = 1
                # Recalculate uncovered
                s = 128
                if s in uncovered:
                    uncovered.remove(s)
                continue
            
            # If stuck, add a random element to break symmetry
            c = random.randint(0, n - 1)
            if c not in A:
                A.add(c)
                x[c] = 1
                # Update uncovered
                for a in A:
                    s = c + a
                    if s in uncovered:
                        uncovered.remove(s)
                s = 2 * c
                if s in uncovered:
                    uncovered.remove(s)
                continue
                
        # Add best_c
        A.add(best_c)
        x[best_c] = 1
        
        # Update uncovered
        for a in A:
            s = best_c + a
            if s in uncovered:
                uncovered.remove(s)
        s = 2 * best_c
        if s in uncovered:
            uncovered.remove(s)

    # Ensure coverage is complete. If greedy failed to cover all, force add
    # Calculate current R
    R = get_representations(x)
    missing_targets = targets[R < 1]
    
    if len(missing_targets) > 0:
        # Fallback: Add elements randomly until covered
        while len(missing_targets) > 0:
            # Pick a missing target t
            t = random.choice(missing_targets.tolist())
            # Find a pair (a, b) such that a+b=t. 
            # Ideally we want to add one element. 
            # If we add c, we need t-c in A.
            # So pick c such that t-c in A.
            # Or pick c such that t-c is a new element? No, we need pair in A.
            # So we must add c where t-c is already in A.
            
            valid_cs = []
            for c in range(n):
                if c in A:
                    continue
                other = t - c
                if 0 <= other < n and other in A:
                    valid_cs.append(c)
            
            if valid_cs:
                c = random.choice(valid_cs)
                A.add(c)
                x[c] = 1
                # Update R and missing
                R = get_representations(x)
                missing_targets = targets[R < 1]
            else:
                # No single element can complete a pair for t with existing A.
                # We need to add two elements? Or change A.
                # Simpler: Add 0 and t? If t <= 128, 0 and t. But t is in [128, 256].
                # If t=128, add 0, 128.
                # If t=256, add 128, 128 (but set, so just 128? No, need two 128s? No, set allows 128 once. Pair (128,128) requires 128 in A. Yes.)
                # So for any t, if we add t//2 (if integer) and ... 
                # Actually, just add a random pair that sums to t.
                # Let a = t // 2. If 2a == t, add a.
                # Else add a and t-a.
                a = t // 2
                b = t - a
                if a == b:
                    if a not in A:
                        A.add(a)
                        x[a] = 1
                else:
                    if a not in A:
                        A.add(a)
                        x[a] = 1
                    if b not in A:
                        A.add(b)
                        x[b] = 1
                        
                R = get_representations(x)
                missing_targets = targets[R < 1]

    # 2. Local Optimization
    # Minimize peak, then sum of squares
    current_R = get_representations(x)
    current_cost = get_cost(x, current_R)
    
    # Local Search
    # Try removing elements first to reduce size
    changed = True
    while changed:
        changed = False
        for a in list(A):
            # Try removing a
            x[a] = 0
            new_R = get_representations(x)
            new_cost = get_cost(x, new_R)
            if new_cost < current_cost:
                A.remove(a)
                current_R = new_R
                current_cost = new_cost
                changed = True
            else:
                x[a] = 1

    # Try swapping
    for _ in range(2000): # Fixed number of iterations
        # Pick a random element in A
        a = random.choice(list(A))
        # Pick a random element not in A
        b_candidates = [i for i in range(n) if i not in A]
        if not b_candidates:
            break
        b = random.choice(b_candidates)
        
        # Swap a with b
        x[a] = 0
        x[b] = 1
        new_R = get_representations(x)
        new_cost = get_cost(x, new_R)
        
        if new_cost < current_cost:
            A.remove(a)
            A.add(b)
            current_R = new_R
            current_cost = new_cost
        else:
            # Revert
            x[a] = 1
            x[b] = 0

    # Try adding elements (if it reduces peak)
    for _ in range(500):
        b_candidates = [i for i in range(n) if i not in A]
        if not b_candidates:
            break
        b = random.choice(b_candidates)
        
        x[b] = 1
        new_R = get_representations(x)
        new_cost = get_cost(x, new_R)
        
        if new_cost < current_cost:
            A.add(b)
            current_R = new_R
            current_cost = new_cost
        else:
            x[b] = 0

    result = list(A)
    if not result:
        return [64]
    return result
