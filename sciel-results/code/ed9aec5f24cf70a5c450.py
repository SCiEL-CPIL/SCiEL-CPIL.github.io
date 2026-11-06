"""Program for B_3[1] set construction with unique triple sums."""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    n_max = 4096
    target_size = 80
    
    # Start with powers of 4 skeleton: 1, 4, 16, 64, 256, 1024, 4096
    # Actually 4096 is too large as it's the max, let's use up to 1024
    # Powers of 4: 1, 4, 16, 64, 256, 1024 (6 elements)
    # These guarantee unique triple sums because in base 4, each "digit" contributes independently
    
    selected = [1, 4, 16, 64, 256, 1024]
    
    # Precompute all triple sums for current set
    def compute_triple_sums(S):
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    sums.add(S[i] + S[j] + S[l])
        return sums
    
    existing_sums = compute_triple_sums(selected)
    
    # Candidates to try: all integers from 1 to 4096 not already selected
    selected_set = set(selected)
    candidates = [x for x in range(1, n_max + 1) if x not in selected_set]
    
    # For efficiency, we'll try candidates in a strategic order
    # Try smaller gaps first, then larger
    # Actually, let's just try all candidates and pick valid ones greedily
    
    # To check if x can be added, we need to verify that no new triple sum
    # involving x collides with existing sums or with other new triple sums involving x
    
    def can_add(x, S, existing_sums):
        """Check if x can be added to S without creating triple sum collisions."""
        # New triple sums involving x:
        # 1. (x, x, x) -> 3x
        # 2. (x, x, s) for s in S -> 2x + s
        # 3. (x, s, s) for s in S -> x + 2s
        # 4. (x, s, t) for s, t in S, s <= t -> x + s + t
        
        new_sums = set()
        
        # Type 1: 3x
        new_sums.add(3 * x)
        
        # Types 2-4: involve x and elements of S
        for s in S:
            new_sums.add(2 * x + s)
            new_sums.add(x + 2 * s)
        
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.add(x + S[i] + S[j])
        
        # Check for collisions with existing sums
        if new_sums & existing_sums:
            return False
        
        # Check for internal collisions among new sums
        if len(new_sums) != len(new_sums):
            return False
        
        # Also need to check: do any two different new triples produce the same sum?
        # We already added to a set, so if len matches, no internal collision
        # But we need to count multiplicities properly
        
        # Actually, let's count the number of triples vs unique sums
        # Number of new triples involving x:
        # (x,x,x): 1
        # (x,x,s): |S|
        # (x,s,s): |S|
        # (x,s,t) for s<t: C(|S|,2)
        # (x,s,s) already counted above
        # Total new triples: 1 + |S| + |S| + C(|S|,2) = 1 + 2|S| + |S|(|S|-1)/2
        
        k = len(S)
        expected_count = 1 + 2 * k + k * (k - 1) // 2
        
        if len(new_sums) != expected_count:
            return False
        
        return True
    
    # Greedily add elements
    # Try candidates in increasing order, but we can also try different orderings
    # Let's try multiple strategies within the time budget
    
    best_selected = list(selected)
    best_size = len(selected)
    
    # Strategy 1: Try candidates in increasing order
    for x in candidates:
        if time.time() > deadline:
            break
        if can_add(x, selected, existing_sums):
            selected.append(x)
            selected_set.add(x)
            # Recompute existing sums
            existing_sums = compute_triple_sums(selected)
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = list(selected)
            if len(selected) >= target_size:
                break
    
    # If we didn't reach target, try a different ordering
    if len(selected) < target_size and time.time() < deadline:
        # Try candidates in a different order - maybe larger values first in gaps
        # Reset to skeleton
        selected = [1, 4, 16, 64, 256, 1024]
        selected_set = set(selected)
        existing_sums = compute_triple_sums(selected)
        
        # Try candidates sorted by some heuristic - maybe by "gap" they fill
        # For now, try reverse order
        candidates2 = [x for x in range(1, n_max + 1) if x not in selected_set]
        candidates2.sort(reverse=True)
        
        for x in candidates2:
            if time.time() > deadline:
                break
            if can_add(x, selected, existing_sums):
                selected.append(x)
                selected_set.add(x)
                existing_sums = compute_triple_sums(selected)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                if len(selected) >= target_size:
                    break
    
    # Try yet another strategy: interleave
    if len(best_selected) < target_size and time.time() < deadline:
        selected = [1, 4, 16, 64, 256, 1024]
        selected_set = set(selected)
        existing_sums = compute_triple_sums(selected)
        
        candidates3 = [x for x in range(1, n_max + 1) if x not in selected_set]
        # Sort by distance to nearest power of 4
        def gap_score(x):
            # Find nearest power of 4
            best = float('inf')
            p = 1
            while p <= n_max:
                best = min(best, abs(x - p))
                p *= 4
            return best
        
        candidates3.sort(key=gap_score)
        
        for x in candidates3:
            if time.time() > deadline:
                break
            if can_add(x, selected, existing_sums):
                selected.append(x)
                selected_set.add(x)
                existing_sums = compute_triple_sums(selected)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                if len(selected) >= target_size:
                    break
    
    return best_selected
