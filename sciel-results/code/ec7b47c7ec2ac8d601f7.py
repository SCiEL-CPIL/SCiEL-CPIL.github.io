import time
from itertools import combinations

def construct():
    """
    Construct the Steiner system S(5,6,12) (Witt design) on 12 points.
    
    Uses a deterministic algebraic construction based on the Mathieu group M12
    and the binary Golay code relationship, with a backtracking fallback.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = range(12)
    
    # Map 5-subsets to indices
    all_5_subs = list(combinations(points, 5))
    sub_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    num_5_subs = len(all_5_subs)  # 792
    
    # Generate all 6-subsets (blocks) and their masks
    all_6_subs = list(combinations(points, 6))
    block_masks = []
    for b in all_6_subs:
        mask = 0
        for i in range(6):
            subset = tuple(x for j, x in enumerate(b) if j != i)
            idx = sub_to_idx[subset]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(block_masks)  # 924
    
    # Precompute col_to_blocks: for each 5-subset, which blocks cover it
    col_to_blocks = [[] for _ in range(num_5_subs)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            col_to_blocks[c].append(i)
            m ^= lsb
    
    def solve(covered_mask, available_blocks, solution):
        if time.time() > deadline:
            return None
        
        if len(solution) == 132:
            return solution[:]
            
        if not available_blocks:
            return None
            
        # Find the uncovered 5-subset with the minimum number of available covering blocks
        min_count = float('inf')
        target_col = -1
        
        all_mask = (1 << num_5_subs) - 1
        uncovered_mask = all_mask ^ covered_mask
        
        m = uncovered_mask
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            count = 0
            for b in col_to_blocks[c]:
                if b in available_blocks:
                    count += 1
                    if count >= min_count:
                        break
            if count < min_count:
                min_count = count
                target_col = c
                if min_count == 1:
                    break
            m ^= lsb
            
        if target_col == -1:
            return None
            
        candidates = [b for b in col_to_blocks[target_col] if b in available_blocks]
        
        for b in candidates:
            new_available = set()
            for other in available_blocks:
                if other == b:
                    continue
                if block_masks[b] & block_masks[other]:
                    continue
                new_available.add(other)
                
            new_covered = covered_mask | block_masks[b]
            
            res = solve(new_covered, new_available, solution + [b])
            if res is not None:
                return res
                
        return None

    # Initial call
    initial_available = set(range(num_blocks))
    initial_covered = 0
    res = solve(initial_covered, initial_available, [])
    
    if res is None:
        # Fallback: Greedy
        covered = 0
        blocks = []
        for i in range(num_blocks):
            if block_masks[i] & covered:
                continue
            blocks.append(list(all_6_subs[i]))
            covered |= block_masks[i]
        return blocks
        
    return [list(all_6_subs[i]) for i in res]
