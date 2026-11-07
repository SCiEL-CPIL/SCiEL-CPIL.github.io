import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 12
    m = 40
    
    # Precompute bitmasks for all subsets? 2^12 = 4096. Easy.
    # But we only need to generate candidates.
    
    # Initial solution: Chain + Heuristic
    # Chain: {0}, {0,1}, ..., {0..11}
    chain_masks = [(1 << i) - 1 for i in range(n + 1)]
    
    # We need 40 sets. Chain has 13. Need 27 more.
    # Heuristic: For each size k, pick sets that are "neighbors" of C_k
    # A set S of size k is comparable with C_j if S subset C_j or C_j subset S.
    # If S has size k, it is comparable with C_j for j >= k if S subset C_j.
    # It is comparable with C_j for j < k if C_j subset S.
    
    # To maximize comparability with the chain, we want S to contain small C_j or be subset of large C_j.
    # Sets that contain {0..k-2} and one extra element are good candidates.
    
    current_sets = list(chain_masks)
    
    # Add 27 sets
    # Let's distribute them across sizes to avoid too many duplicates of structure
    # Or just pick random valid sets and improve.
    # A good start: for each k in 0..11, add some sets of size k.
    # Let's add sets that are "shifted" chains.
    # E.g., {1..k}, {0..k-2, k+1}, etc.
    
    # Simple heuristic: pick 27 random subsets, ensuring distinctness.
    # Then optimize.
    
    all_masks = list(range(1 << n))
    random.shuffle(all_masks)
    
    # We already have chain masks.
    chain_set = set(chain_masks)
    
    for mask in all_masks:
        if len(current_sets) >= m:
            break
        if mask not in chain_set:
            current_sets.append(mask)
            chain_set.add(mask)
            
    # Ensure we have exactly 40
    # The above loop might not fill if random picks collide with chain, but all_masks is shuffled and large.
    # 4096 masks, 13 in chain, 27 needed. 4096-13 > 27. So it will fill.
    
    def count_comparable(sets):
        cnt = 0
        n_sets = len(sets)
        for i in range(n_sets):
            a = sets[i]
            for j in range(i + 1, n_sets):
                b = sets[j]
                # Strict subset: a < b or b < a
                # a is strict subset of b if (a & b) == a and a != b
                # Since all sets are distinct, a != b is always true.
                if (a & b) == a or (a & b) == b:
                    cnt += 1
        return cnt
    
    best_score = count_comparable(current_sets)
    best_sets = list(current_sets)
    
    # Local Search
    # Try swapping each set with a random other set of the same size?
    # Or just a random set?
    # Swapping with a set of the same size is a good neighborhood.
    
    # Precompute masks by size for faster sampling
    masks_by_size = {k: [] for k in range(n + 1)}
    for mask in all_masks:
        masks_by_size[mask.bit_count()].append(mask)
        
    # Shuffle lists for random access
    for k in masks_by_size:
        random.shuffle(masks_by_size[k])
        
    current_score = best_score
    
    # Hill Climbing
    # We can afford a few thousand iterations.
    # 40 sets * 40 pairs = 1600 checks per evaluation.
    # 10000 evaluations = 16M checks. Very fast in Python if optimized? 
    # Python loops are slow. 16M operations might take ~1-2 seconds.
    # So we can do more iterations.
    
    # Optimization: Incremental score update?
    # Too complex for the time limit. Just re-evaluate.
    
    # Let's try 5000 iterations of swapping one set with a random set of same size.
    # And also try random restarts if stuck.
    
    max_iterations = 20000
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random index to change
        idx = random.randint(0, m - 1)
        old_mask = current_sets[idx]
        old_size = old_mask.bit_count()
        
        # Pick a random new mask of the same size
        # Ensure it's not the same as current
        # And not already in the set (unless we swap it out, which we do)
        # We need to ensure the new set is not equal to any other set in the list.
        
        # Find a valid replacement
        # Sample from masks_by_size[old_size]
        # Check if it collides with other sets
        # If collision, try again or pick different size?
        # Let's just pick a random mask from all_masks, check validity.
        
        new_mask = None
        # Try a few times to find a valid distinct mask
        for _ in range(20):
            cand = random.choice(masks_by_size[old_size])
            if cand == old_mask:
                continue
            # Check if cand is in current_sets (excluding idx)
            # This is O(m).
            is_dup = False
            for k in range(m):
                if k != idx and current_sets[k] == cand:
                    is_dup = True
                    break
            if not is_dup:
                new_mask = cand
                break
        
        if new_mask is None:
            continue
            
        # Swap
        current_sets[idx] = new_mask
        
        new_score = count_comparable(current_sets)
        
        if new_score > current_score:
            current_score = new_score
            best_score = new_score
            best_sets = list(current_sets)
        else:
            # Revert
            current_sets[idx] = old_mask
            
    # Convert masks back to lists of ints
    result = []
    for mask in best_sets:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
