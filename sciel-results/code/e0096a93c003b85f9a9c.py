"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

N = 12
M = 40

def _count_comparable(family):
    """Count unordered pairs (i,j) with i<j such that family[i] < family[j] or family[j] < family[i]."""
    masks = [0] * len(family)
    for i, s in enumerate(family):
        m = 0
        for v in s:
            m |= (1 << v)
        masks[i] = m
    count = 0
    n = len(family)
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            if mi & mj == mi or mi & mj == mj:
                count += 1
    return count

def _greedy_construct():
    """
    Greedy construction over B_6 (subsets of {0..5}).
    We want to select 40 subsets to maximize comparable pairs.
    Strategy: start with empty set, then greedily add sets that maximize marginal gain.
    """
    vertices = list(range(6))
    # Generate all 64 subsets of {0..5} as masks
    all_masks = []
    for mask in range(64):
        all_masks.append(mask)
    
    # We need to pick 40 out of 64
    # Greedy: start with a good seed, then iteratively add the set with highest marginal gain
    
    # Let's try a different approach: 
    # The comparable pairs in a family F is sum over pairs where one is subset of other.
    # If we think of the family as a poset, we want to maximize the number of comparable pairs.
    # This is equivalent to maximizing the sum over each element of the number of elements below it + above it.
    
    # Greedy algorithm:
    # 1. Start with empty set (mask 0)
    # 2. At each step, add the mask that gives the largest increase in comparable pairs
    
    selected = [0]  # Start with empty set
    selected_set = {0}
    
    # Precompute for each pair of masks whether they are comparable
    # comp[i][j] = True if mask i is subset of mask j or vice versa
    comp = [[False] * 64 for _ in range(64)]
    for i in range(64):
        for j in range(64):
            if (i & j) == i or (i & j) == j:
                comp[i][j] = True
    
    # For efficiency, precompute the contribution of adding a new mask m to a current selected set S:
    # gain(m, S) = number of s in S such that s and m are comparable
    
    while len(selected) < M:
        best_mask = -1
        best_gain = -1
        for m in range(64):
            if m in selected_set:
                continue
            gain = 0
            for s in selected:
                if comp[m][s]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        if best_mask == -1:
            break
        selected.append(best_mask)
        selected_set.add(best_mask)
    
    # Convert masks to lists of vertices
    result = []
    for mask in selected:
        subset = [b for b in range(6) if (mask >> b) & 1]
        result.append(subset)
    
    return result

def _local_search(family, max_iters=5000):
    """
    Local search: try to swap one set in the family with one not in the family to improve count.
    """
    current = list(family)
    current_masks = []
    for s in current:
        m = 0
        for v in s:
            m |= (1 << v)
        current_masks.append(m)
    
    current_count = _count_comparable(current)
    
    # All possible masks in B_6
    all_masks = list(range(64))
    selected_set = set(current_masks)
    
    # Precompute comparability
    comp = [[False] * 64 for _ in range(64)]
    for i in range(64):
        for j in range(64):
            if (i & j) == i or (i & j) == j:
                comp[i][j] = True
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(len(current_masks)):
            mi = current_masks[i]
            for m in all_masks:
                if m in selected_set:
                    continue
                # Compute new count if we replace mi with m
                # Old contribution of mi: number of j != i such that comp[mi][mj]
                # New contribution of m: number of j != i such that comp[m][mj]
                old_contrib = 0
                new_contrib = 0
                for j in range(len(current_masks)):
                    if j == i:
                        continue
                    mj = current_masks[j]
                    if comp[mi][mj]:
                        old_contrib += 1
                    if comp[m][mj]:
                        new_contrib += 1
                delta = new_contrib - old_contrib
                if delta > 0:
                    # Make the swap
                    current_masks[i] = m
                    selected_set.remove(mi)
                    selected_set.add(m)
                    current_count += delta
                    improved = True
                    break
            if improved:
                break
    
    # Convert back to lists
    result = []
    for mask in current_masks:
        subset = [b for b in range(6) if (mask >> b) & 1]
        result.append(subset)
    
    return result, current_count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline to be safe
    
    # Start with greedy construction
    family = _greedy_construct()
    best_count = _count_comparable(family)
    
    # Local search refinement
    family, count = _local_search(family, max_iters=10000)
    if count > best_count:
        best_count = count
    
    # Try a few random restarts with local search if time permits
    random.seed(42)
    while time.time() < deadline:
        # Random initial family: pick 40 random subsets from B_6
        all_masks = list(range(64))
        random.shuffle(all_masks)
        init_family_masks = all_masks[:M]
        init_family = []
        for mask in init_family_masks:
            subset = [b for b in range(6) if (mask >> b) & 1]
            init_family.append(subset)
        
        family, count = _local_search(init_family, max_iters=5000)
        if count > best_count:
            best_count = count
            family = family
    
    # Ensure types are correct
    result = []
    for s in family:
        result.append([int(v) for v in s])
    
    return result
# EVOLVE-BLOCK-END
