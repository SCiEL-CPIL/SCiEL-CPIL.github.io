import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    all_subsets = list(range(1 << n))
    
    # Helper: check if a is strict subset of b
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    # Helper: count comparables between mask and a list of other masks
    def count_comparables(mask, others):
        count = 0
        for o in others:
            if mask == o:
                continue
            if (mask & o) == mask:
                count += 1
            elif (mask & o) == o:
                count += 1
        return count
    
    # Step 1: Build a good initial solution using greedy
    # Start with a maximal chain and its "dual" structure
    current_masks = []
    current_set = set()
    
    # Add empty and full
    current_masks.append(0)
    current_set.add(0)
    current_masks.append(full_mask)
    current_set.add(full_mask)
    
    # Add all singletons
    for i in range(n):
        current_masks.append(1 << i)
        current_set.add(1 << i)
    
    # Add all co-singletons (size n-1)
    for i in range(n):
        s = full_mask ^ (1 << i)
        current_masks.append(s)
        current_set.add(s)
    
    # Greedily add remaining sets
    while len(current_masks) < m:
        if time.time() > deadline:
            break
        best_gain = -1
        best_mask = None
        for c in all_subsets:
            if c in current_set:
                continue
            gain = count_comparables(c, current_masks)
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is None or best_gain <= 0:
            break
        current_masks.append(best_mask)
        current_set.add(best_mask)
    
    # Ensure we have exactly m sets (fill with random if needed)
    while len(current_masks) < m:
        c = random.choice(all_subsets)
        if c not in current_set:
            current_masks.append(c)
            current_set.add(c)
    
    # Step 2: Precompute comparability matrix for fast delta updates
    # comp[i][j] = 1 if current_masks[i] and current_masks[j] are comparable (strict subset either way)
    N = m
    
    def build_comp_matrix(masks):
        comp = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                a, b = masks[i], masks[j]
                if a == b:
                    continue
                if (a & b) == a or (a & b) == b:
                    comp[i][j] = 1
                    comp[j][i] = 1
        return comp
    
    def compute_score_from_matrix(comp):
        count = 0
        for i in range(N):
            for j in range(i + 1, N):
                count += comp[i][j]
        return count
    
    comp = build_comp_matrix(current_masks)
    current_score = compute_score_from_matrix(comp)
    
    # Precompute comparability of each candidate with each current mask position
    # For a candidate mask c and position idx, the "contribution" is the number of j != idx
    # such that c and current_masks[j] are comparable.
    # We can precompute for each candidate c: comp_count[c][idx] = number of j != idx where c comparable to masks[j]
    # But that's 4096 * 40 = 163840 entries. Let's compute on the fly but efficiently.
    
    # Actually, let's precompute a lookup: for each mask value, store its comparability with each position.
    # But positions change as we swap. Instead, let's just compute delta efficiently.
    
    # For a swap at index idx: remove old_mask, add new_mask.
    # Old contribution: sum of comp[idx][j] for j != idx
    # New contribution: number of j != idx where new_mask comparable to current_masks[j]
    
    # To compute new contribution fast, we can precompute for each candidate mask c
    # a bitmask of which positions it's comparable to. But positions change.
    
    # Alternative: maintain an array contrib[j] = number of i < j where comp[i][j]=1
    # No, let's just compute the new contribution by checking new_mask against all other masks.
    # That's O(N) per swap evaluation, which is 40 checks. Still much better than O(N^2).
    
    # Let's optimize the inner loop:
    def new_contrib(new_mask, idx):
        count = 0
        for j in range(N):
            if j == idx:
                continue
            o = current_masks[j]
            if new_mask == o:
                continue
            if (new_mask & o) == new_mask or (new_mask & o) == o:
                count += 1
        return count
    
    # Local Search with fast delta
    last_improvement = time.time()
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            # Try a restart with a different random seed or perturbation
            # For now, just continue but increase max_no_improve
            max_no_improve *= 2
            no_improve_count = 0
        
        # Pick a random index to replace
        idx = random.randint(0, m - 1)
        old_mask = current_masks[idx]
        
        # Pick a random replacement
        new_mask = random.choice(all_subsets)
        
        if new_mask == old_mask:
            continue
        
        if new_mask in current_set:
            continue
        
        # Calculate old contribution: sum of comp[idx][j] for j != idx
        old_contrib = 0
        for j in range(N):
            if j != idx:
                old_contrib += comp[idx][j]
        
        # Calculate new contribution
        new_contrib_val = new_contrib(new_mask, idx)
        
        delta = new_contrib_val - old_contrib
        
        if delta > 0:
            # Accept the swap
            # Update comp matrix
            for j in range(N):
                if j == idx:
                    continue
                # Remove old comparability
                comp[idx][j] = 0
                comp[j][idx] = 0
                # Add new comparability
                o = current_masks[j]
                if new_mask != o:
                    if (new_mask & o) == new_mask or (new_mask & o) == o:
                        comp[idx][j] = 1
                        comp[j][idx] = 1
            
            current_masks[idx] = new_mask
            current_set.remove(old_mask)
            current_set.add(new_mask)
            current_score += delta
            last_improvement = time.time()
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Convert back to lists
    result = []
    for mask in current_masks:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
