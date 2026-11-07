import time
import random

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    sets[i] is a strict subset of sets[j] or sets[j] is a strict subset of sets[i].
    """
    n = P['n']
    m = len(sets)
    masks = [0] * m
    for i in range(m):
        mask = 0
        for v in sets[i]:
            if 0 <= v < n:
                mask |= (1 << v)
        masks[i] = mask
    
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            if (mi & mj) == mi or (mi & mj) == mj:
                if mi != mj: # Strict subset check is redundant if we know they are distinct, 
                             # but (mi & mj) == mi implies mi is subset. 
                             # If mi == mj, they are not strict. 
                             # Since sets are distinct, mi != mj is guaranteed.
                    count += 1
    return count

def _score_with_swap(sets, idx, new_mask):
    """
    Calculate the change in comparable pairs if sets[idx] is replaced by new_mask.
    sets is a list of masks.
    """
    m = len(sets)
    old_mask = sets[idx]
    
    deg_old = 0
    deg_new = 0
    
    for j in range(m):
        if j == idx:
            continue
        mj = sets[j]
        
        # Check if old_mask is comparable with mj
        if (old_mask & mj) == old_mask or (old_mask & mj) == mj:
            deg_old += 1
            
        # Check if new_mask is comparable with mj
        if (new_mask & mj) == new_mask or (new_mask & mj) == mj:
            deg_new += 1
            
    return deg_new - deg_old

def _random_initial_solution():
    n = P['n']
    m = P['m']
    all_masks = [0] * (1 << n)
    # We need m distinct random masks
    # Using a set to track chosen masks
    chosen = set()
    while len(chosen) < m:
        mask = random.randint(0, (1 << n) - 1)
        chosen.add(mask)
    return list(chosen)

def _local_search(initial_masks, deadline):
    n = P['n']
    m = P['m']
    current_masks = list(initial_masks)
    current_score = _count_comparable_pairs([[b for b in range(n) if (mask>>b)&1] for mask in current_masks])
    
    best_score = current_score
    best_masks = list(current_masks)
    
    # Precompute all possible masks? 2^12 = 4096. Yes.
    all_possible_masks = list(range(1 << n))
    
    # To speed up random selection of a new mask not in current set, 
    # we can just pick randomly and check membership.
    # Membership check in a set is O(1).
    
    current_set = set(current_masks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try a few swaps per "epoch" or just iterate
        # Let's do a full pass of random attempts
        for _ in range(1000): # Fixed number of attempts per restart/epoch
            if time.time() >= deadline:
                return best_score, best_masks
            
            idx = random.randint(0, m - 1)
            # Pick a random new mask
            new_mask = random.choice(all_possible_masks)
            if new_mask in current_set:
                continue
            
            delta = _score_with_swap(current_masks, idx, new_mask)
            if delta > 0:
                # Accept
                old_mask = current_masks[idx]
                current_masks[idx] = new_mask
                current_set.remove(old_mask)
                current_set.add(new_mask)
                current_score += delta
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(current_masks)
                    
    return best_score, best_masks

def construct():
    deadline = time.time() + 50.0 # 50s safety margin
    
    best_overall_score = -1
    best_overall_masks = None
    
    # Multiple restarts
    while time.time() < deadline:
        init_masks = _random_initial_solution()
        score, masks = _local_search(init_masks, deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_masks = masks
            
        # If we hit a very high score, we might break early? 
        # Max possible pairs is C(40,2) = 780. 
        # A chain of 13 has C(13,2)=78. 
        # If we have multiple chains, say 3 chains of 13, 13, 14? 
        # C(13,2)+C(13,2)+C(14,2) = 78+78+91 = 247.
        # If we can get close to 780, that's great.
        # But with n=12, we can't have a chain of 40.
        # Max chain length is 13.
        # So we are limited.
        # Let's just run until deadline.
        
        # If the score is very high, e.g., > 400, maybe stop? 
        # No, keep searching for better.
        
    # Convert best_overall_masks to lists of integers
    n = P['n']
    result = []
    for mask in best_overall_masks:
        subset = []
        for b in range(n):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
        
    return result
