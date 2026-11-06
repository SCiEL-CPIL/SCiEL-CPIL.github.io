import time
from itertools import combinations

def _solve_s4511():
    """
    Construct S(4,5,11) - Steiner system with 66 blocks of size 5 on 11 points.
    Each 4-subset appears in exactly one block.
    """
    n = 11
    points = list(range(n))
    
    # Precompute all 4-subsets and 5-subsets
    all_4subsets = list(combinations(points, 4))
    all_5subsets = list(combinations(points, 5))
    
    num_4 = len(all_4subsets) # 330
    num_5 = len(all_5subsets) # 462
    
    # Map 4-subset to index
    s4_index = {s4: i for i, s4 in enumerate(all_4subsets)}
    
    # For each 5-subset, store the indices of the 4-subsets it contains
    # This is a list of lists of indices
    s5_to_s4s = []
    for s5 in all_5subsets:
        s4s = []
        for s4 in combinations(s5, 4):
            s4s.append(s4_index[s4])
        s5_to_s4s.append(s4s)
        
    # For each 4-subset, store the indices of the 5-subsets that contain it
    # This is a list of lists of indices
    s4_to_s5s = [[] for _ in range(num_4)]
    for s5_idx, s4s in enumerate(s5_to_s4s):
        for s4_idx in s4s:
            s4_to_s5s[s4_idx].append(s5_idx)
            
    # Precompute a bitmask for each 5-subset representing the 4-subsets it covers
    s5_masks = []
    for s4s in s5_to_s4s:
        mask = 0
        for idx in s4s:
            mask |= (1 << idx)
        s5_masks.append(mask)
        
    # Target number of blocks
    target_blocks = 66
    
    # Symmetry breaking: The first block is fixed to (0, 1, 2, 3, 4)
    # Find its index in all_5subsets
    first_block = (0, 1, 2, 3, 4)
    try:
        first_block_idx = all_5subsets.index(first_block)
    except ValueError:
        # This should not happen
        return None
        
    initial_mask = s5_masks[first_block_idx]
    blocks = [first_block]
    
    start_time = time.time()
    
    def backtrack(mask, current_blocks):
        if time.time() - start_time > 10: # Safety margin
            return False
            
        if len(current_blocks) == target_blocks:
            return True
            
        # Find an uncovered 4-subset with the fewest valid candidate 5-subsets (MRV)
        min_cands = float('inf')
        best_s4_idx = -1
        best_cands = []
        
        # Iterate through all 4-subsets
        # Optimization: We can skip if the bit is set in mask
        for s4_idx in range(num_4):
            if mask & (1 << s4_idx):
                continue
                
            # Find valid 5-subsets containing this 4-subset
            # A 5-subset is valid if its mask is disjoint from the current mask
            valid_cands = []
            for s5_idx in s4_to_s5s[s4_idx]:
                if s5_masks[s5_idx] & mask:
                    continue
                valid_cands.append(s5_idx)
                
            c = len(valid_cands)
            if c < min_cands:
                min_cands = c
                best_s4_idx = s4_idx
                best_cands = valid_cands
                if c == 0:
                    return False
                if c == 1:
                    # Can't get better than 1
                    break
                    
        if best_s4_idx == -1:
            # All 4-subsets are covered, but we haven't reached target blocks?
            # This implies we covered all 4-subsets with fewer than 66 blocks, which is impossible
            # for a Steiner system, but might happen in a partial search.
            # If all 4-subsets are covered, we are done only if len == target.
            # Since we return False if len != target at the start of the function,
            # this case means we are stuck.
            return False
            
        # Try each valid candidate
        for s5_idx in best_cands:
            new_mask = mask | s5_masks[s5_idx]
            current_blocks.append(all_5subsets[s5_idx])
            
            if backtrack(new_mask, current_blocks):
                return True
                
            current_blocks.pop()
            
        return False
        
    if backtrack(initial_mask, blocks):
        return blocks
        
    return None

def _witt_s5612_from_s4511(s4511_blocks):
    """
    Construct S(5,6,12) from S(4,5,11) by adding point 11.
    For each 5-block B in S(4,5,11), create 6-block B ∪ {11}.
    Also create 6-blocks from the complements: for each 5-block B,
    the complement in {0..10} has size 6, and we add that as a block too.
    """
    if s4511_blocks is None:
        return None
    
    blocks = []
    for b in s4511_blocks:
        # Block with infinity (11)
        blocks.append(list(b) + [11])
        # Complement block (in 0..10)
        # b is a tuple of 5 distinct ints from 0..10
        # We need the 6 ints from 0..10 not in b
        b_set = set(b)
        comp = [i for i in range(11) if i not in b_set]
        blocks.append(comp)
    
    return blocks

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Try the algebraic construction via S(4,5,11)
    try:
        s4511 = _solve_s4511()
        if s4511 is not None and len(s4511) == 66:
            witt = _witt_s5612_from_s4511(s4511)
            if witt is not None and len(witt) == 132:
                # Verify
                covered = set()
                valid = True
                for b in witt:
                    for s5 in combinations(b, 5):
                        if s5 in covered:
                            valid = False
                            break
                        covered.add(s5)
                    if not valid:
                        break
                if valid:
                    return witt
    except Exception:
        pass
    
    # Fallback: Greedy with randomized restarts if time permits
    # This is unlikely to reach 132, but better than nothing.
    # However, given the strict requirement to maximize, and the fact that
    # the algebraic construction is deterministic and fast, this is just a safety net.
    
    if time.time() < deadline:
        best_result = []
        # Simple greedy
        covered, result = set(), []
        for b in combinations(range(12), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                result.append(list(b))
                covered.update(subsets)
                if time.time() > deadline - 5:
                    break
        if len(result) > len(best_result):
            best_result = result
            
        return best_result
        
    return []

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} blocks")
    if len(res) > 0:
        # Quick validation
        covered = set()
        valid = True
        for b in res:
            for s5 in combinations(b, 5):
                if s5 in covered:
                    valid = False
                    break
                covered.add(s5)
            if not valid:
                break
        print(f"Valid: {valid}")
