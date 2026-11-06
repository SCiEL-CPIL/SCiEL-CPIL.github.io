# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """
    Find the largest Sidon set containing {1, 2, 5, 11} in [1, 284].
    
    Approach:
    1. Formulate as Maximum Independent Set (MIS) on a conflict graph.
    2. Use bitmask representation for fast neighbor checks.
    3. Solve with Branch and Bound using heuristic ordering and pruning.
    """
    
    mandatory = [1, 2, 5, 11]
    start_time = time.time()
    time_limit = 110  # seconds
    max_val = 284
    
    # Precompute sums for mandatory elements
    mand_sums = set()
    for i in range(len(mandatory)):
        for j in range(i, len(mandatory)):
            mand_sums.add(mandatory[i] + mandatory[j])
    
    # Generate candidates
    candidates = []
    for x in range(1, max_val + 1):
        if x in mandatory:
            continue
        valid = True
        if 2 * x in mand_sums:
            valid = False
        else:
            for m in mandatory:
                if x + m in mand_sums:
                    valid = False
                    break
        if valid:
            candidates.append(x)
    
    n_cand = len(candidates)
    cand_index = {c: i for i, c in enumerate(candidates)}
    
    # Build conflict graph using bitmasks
    # mask[i] has bit j set if i and j conflict
    masks = [0] * n_cand
    
    for i in range(n_cand):
        u = candidates[i]
        sums_with_u = set(mand_sums)
        sums_with_u.add(2 * u)
        for m in mandatory:
            sums_with_u.add(u + m)
            
        for j in range(i + 1, n_cand):
            v = candidates[j]
            if 2 * v in sums_with_u or u + v in sums_with_u:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
                
    # Greedy initialization to get a strong lower bound
    def greedy_init():
        best_size = 0
        best_mask = 0
        # Try a few random orderings
        indices = list(range(n_cand))
        for _ in range(200):
            random.shuffle(indices)
            chosen_mask = 0
            blocked_mask = 0
            size = 0
            for i in indices:
                if (chosen_mask >> i) & 1: continue
                if (blocked_mask >> i) & 1: continue
                chosen_mask |= (1 << i)
                size += 1
                blocked_mask |= masks[i]
            if size > best_size:
                best_size = size
                best_mask = chosen_mask
        return best_size, best_mask

    init_size, init_mask = greedy_init()
    best_size = init_size
    best_mask = init_mask
    
    # B&B with bitmasks
    sys.setrecursionlimit(3000)
    
    def popcount(x):
        return bin(x).count("1")
        
    def branch_and_bound(avail_mask, chosen_mask, chosen_size):
        nonlocal best_size, best_mask
        nonlocal start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        # Prune: if even taking all available doesn't beat best
        if chosen_size + popcount(avail_mask) <= best_size:
            return
            
        if avail_mask == 0:
            if chosen_size > best_size:
                best_size = chosen_size
                best_mask = chosen_mask
            return
            
        # Reduction: Isolated vertices
        # A vertex v is isolated in avail if (masks[v] & avail_mask) == 0
        # We can greedily pick all isolated vertices
        isolated_mask = 0
        # Iterate through set bits in avail_mask
        temp = avail_mask
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            v_idx = lsb.bit_length() - 1
            if (masks[v_idx] & avail_mask) == 0:
                isolated_mask |= lsb
            temp ^= lsb
            
        if isolated_mask:
            # Add all isolated to chosen
            new_chosen_mask = chosen_mask | isolated_mask
            new_chosen_size = chosen_size + popcount(isolated_mask)
            # Remove isolated from avail
            new_avail_mask = avail_mask & ~isolated_mask
            branch_and_bound(new_avail_mask, new_chosen_mask, new_chosen_size)
            return
            
        # Heuristic Branching
        # Pick vertex with highest degree in avail
        best_v = -1
        max_deg = -1
        temp = avail_mask
        while temp:
            lsb = temp & -temp
            v_idx = lsb.bit_length() - 1
            deg = popcount(masks[v_idx] & avail_mask)
            if deg > max_deg:
                max_deg = deg
                best_v = v_idx
            temp ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New avail = avail - {v} - neighbors(v)
        new_avail_include = avail_mask & ~(v_bit | masks[best_v])
        new_chosen_include = chosen_mask | v_bit
        branch_and_bound(new_avail_include, new_chosen_include, chosen_size + 1)
        
        # Branch 2: Exclude best_v
        new_avail_exclude = avail_mask & ~v_bit
        branch_and_bound(new_avail_exclude, chosen_mask, chosen_size)
        
    # Start B&B
    # Initial avail is all candidates
    all_mask = (1 << n_cand) - 1
    branch_and_bound(all_mask, 0, 0)
    
    # Construct final set
    final_set = set(mandatory)
    # Extract indices from best_mask
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        final_set.add(candidates[idx])
        temp ^= lsb
        
    return sorted(final_set)

# EVOLVE-BLOCK-END
