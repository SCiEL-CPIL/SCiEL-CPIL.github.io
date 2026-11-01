"""Covering design C(24,7,3) via bitset-accelerated greedy with Variable Neighborhood Search."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin before 150s kill
    seed = 42
    random.seed(seed)
    
    # Precompute all t-subsets (triples) and map to indices
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Precompute for each block (k-subset) the list of t-set indices it covers
    # We'll compute this on the fly or cache if needed, but for v=24,k=7,t=3:
    # Number of possible blocks = C(24,7) = 346104. Too many to precompute all.
    # But we only need to evaluate candidates.
    
    # Helper: get bitmask of covered t-sets for a given block
    # Represent coverage as a Python int (bitset)
    # Since n_tsets = 2024, we can use Python's arbitrary precision integers.
    
    def block_to_bitmask(block):
        """Convert a block (list of ints) to a bitmask of covered t-sets."""
        bm = 0
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            bm |= (1 << idx)
        return bm
    
    # Precompute all t-sets as a full mask
    full_mask = (1 << n_tsets) - 1
    
    # Greedy construction with bitsets
    def greedy_construct():
        """
        Greedy set cover: repeatedly pick the block that covers the most uncovered t-sets.
        To speed up, we don't evaluate all C(v,k) blocks. Instead, we use a heuristic:
        Start with a random t-set, build a block around it, then try to improve.
        Actually, standard greedy for covering designs is expensive if we check all blocks.
        
        Alternative:
        1. Start with empty set of blocks.
        2. While there are uncovered t-sets:
           a. Pick a random uncovered t-set.
           b. Construct a block containing this t-set that maximizes coverage of *other* uncovered t-sets.
              How? Try all extensions of the t-set to a k-set?
              Number of extensions: C(v-t, k-t) = C(21, 4) = 5985.
              For each candidate block, compute how many new t-sets it covers.
              This is 5985 * C(7,3) = 5985 * 35 ~ 200k ops per step.
              Number of steps ~ 76. Total ~ 15M ops. This is feasible.
        """
        uncovered_mask = full_mask
        blocks = []
        block_bitmasks = []
        
        while uncovered_mask and time.time() < deadline - 10:
            # Pick a random uncovered t-set to anchor the new block
            # Find an index where bit is set in uncovered_mask
            # We can iterate through set bits, but picking random is better for diversity
            # Let's pick a random index from 0 to n_tsets-1 and check if it's uncovered.
            # Or, more efficiently, pick a random t-set from the list and check.
            # Actually, we can just pick a random t-set from all_tsets and check if it's in uncovered_mask.
            # But we need to ensure we make progress.
            
            # Find an uncovered t-set
            # For efficiency in Python, iterating set bits of a large int is slow.
            # Let's maintain a list of uncovered indices? No, that's also slow to update.
            # Let's just pick a random t-set index.
            # If it's covered, pick another.
            # Since we start with 2024 and remove ~35 each time, the density is high initially.
            
            anchor_idx = -1
            for _ in range(100):  # Try up to 100 times to find an uncovered one
                idx = random.randint(0, n_tsets - 1)
                if (uncovered_mask >> idx) & 1:
                    anchor_idx = idx
                    break
            
            if anchor_idx == -1:
                # Fallback: find any set bit
                # This is rare if random sampling fails
                tmp = uncovered_mask
                while tmp:
                    idx = (tmp & -tmp).bit_length() - 1
                    if (uncovered_mask >> idx) & 1:
                        anchor_idx = idx
                        break
                    tmp &= tmp - 1
            
            if anchor_idx == -1:
                break # Should not happen if uncovered_mask is non-zero
                
            anchor_tset = all_tsets[anchor_idx]
            
            # Generate all blocks containing anchor_tset
            # anchor_tset has t elements. We need to add k-t more from v-t remaining.
            remaining_points = [x for x in range(v) if x not in anchor_tset]
            best_block = None
            best_gain = -1
            best_bm = 0
            
            # To avoid generating all 5985 candidates every time (which is slow in Python),
            # we can sample or use a smarter heuristic.
            # However, 5985 is small enough.
            # Let's optimize the inner loop.
            
            # Precompute the mask of the anchor t-set itself? No, we need the whole block's mask.
            
            for combo in itertools.combinations(remaining_points, k - t):
                block = list(anchor_tset) + list(combo)
                bm = block_to_bitmask(block)
                # Gain is number of new t-sets covered
                gain = (bm & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_bm = bm
            
            if best_block is not None:
                best_block.sort()
                blocks.append(best_block)
                block_bitmasks.append(best_bm)
                uncovered_mask &= ~best_bm
        
        return blocks, block_bitmasks, uncovered_mask
    
    # Local Search / VNS
    def local_search(blocks, block_bitmasks, uncovered_mask, deadline_ls):
        """
        Variable Neighborhood Search.
        Neighborhood r: replace r blocks with r new blocks.
        Start with r=1.
        """
        current_blocks = list(blocks)
        current_bitmasks = list(block_bitmasks)
        current_uncovered = uncovered_mask
        current_size = len(current_blocks)
        
        # Precompute the union mask of all current blocks
        union_mask = 0
        for bm in current_bitmasks:
            union_mask |= bm
        
        # If not fully covered, we can't really do LS effectively without fixing coverage first.
        # But the greedy should produce a valid covering.
        # Let's assume current_uncovered is 0.
        
        no_improve = 0
        r = 1
        
        while time.time() < deadline_ls and no_improve < 20:
            improved = False
            
            # Neighborhood r: remove r blocks, add r new blocks.
            # For r=1:
            # Pick a block i to remove.
            # Find a new block j that covers all t-sets that were ONLY covered by block i.
            # And ideally covers some t-sets covered by other blocks to enable further removal?
            # Actually, if we replace block i with block j, the new coverage is:
            # (Union \ Mask_i) | Mask_j
            # We need this to be full_mask (or at least not lose coverage).
            # If it loses coverage, it's invalid.
            
            if r == 1:
                # Try to replace each block
                for i in range(len(current_blocks)):
                    if time.time() >= deadline_ls:
                        break
                    
                    # What t-sets are uniquely covered by block i?
                    # unique_mask_i = Mask_i & ~(Union of all other blocks)
                    # Union of all other blocks = Union_mask ^ (Mask_i & Union_mask) ? No.
                    # Union_others = Union_mask & ~Mask_i  -- this is wrong if other blocks overlap with i.
                    # Correct: Union_others = Union_mask without i's contribution?
                    # Actually, we can compute Union_others by OR-ing all bitmasks except i.
                    # This is O(n) per i, so O(n^2) total. n ~ 76. 76^2 = 5776. Fast.
                    
                    union_others = 0
                    for j in range(len(current_blocks)):
                        if j != i:
                            union_others |= current_bitmasks[j]
                    
                    # T-sets that would be lost if we remove i
                    lost = current_bitmasks[i] & ~union_others
                    
                    # We need a new block j such that Mask_j covers 'lost'.
                    # And to improve, we want to reduce the total number of blocks?
                    # Replacing 1 with 1 doesn't reduce count.
                    # But it might enable removing *other* blocks later?
                    # Or, we can try to find a block j that covers 'lost' AND is "better"
                    # such that it allows pruning.
                    
                    # Strategy: Find a block j that covers 'lost'.
                    # Among those, pick one that maximizes overlap with other blocks' unique parts?
                    # Or simply, if we find such a j, we swap. Then we run a pruning pass.
                    
                    # How to find such a j?
                    # We can't iterate all C(24,7) blocks.
                    # Heuristic:
                    # The new block must contain all t-sets in 'lost'? No, just cover them.
                    # A t-set in 'lost' must be a subset of j.
                    # So j must contain the union of all t-sets in 'lost'? No, just each one individually.
                    # But if 'lost' contains disjoint t-sets, j might not contain all of them.
                    # However, 'lost' consists of t-sets that are ONLY in block i.
                    # Since block i has size k=7, it contains C(7,3)=35 t-sets.
                    # 'lost' is a subset of these 35.
                    
                    # If 'lost' is empty, block i is redundant. Remove it!
                    if lost == 0:
                        # Remove block i
                        current_blocks.pop(i)
                        current_bitmasks.pop(i)
                        current_size -= 1
                        improved = True
                        # Recalculate union_mask
                        union_mask = 0
                        for bm in current_bitmasks:
                            union_mask |= bm
                        break  # Restart loop with new state
                    
                    # If 'lost' is not empty, we need to find a replacement.
                    # This is hard. Let's skip complex replacements for r=1 and focus on pruning.
                    # But pruning is already done by the 'lost == 0' check.
                    
                    # Let's try a different LS move:
                    # Remove block i. If coverage is broken, try to fix it by adding a new block?
                    # That increases size. Not helpful for minimization.
                    
                    # So for r=1, only pruning is possible.
                    
                if not improved:
                    no_improve += 1
                    # If no pruning possible, try r=2?
                    # Or random restart?
                    # For now, if r=1 fails, we might be stuck.
                    # Let's try a random perturbation: remove a random block, add a random block that covers the lost t-sets.
                    # This is expensive to find.
                    
                    # Alternative: Try to find a block that covers 'lost' for a random i.
                    # Pick a random i.
                    i = random.randint(0, len(current_blocks) - 1)
                    union_others = 0
                    for j in range(len(current_blocks)):
                        if j != i:
                            union_others |= current_bitmasks[j]
                    lost = current_bitmasks[i] & ~union_others
                    
                    if lost != 0:
                        # Find a block covering 'lost'.
                        # 'lost' has at least 1 t-set.
                        # Let's pick one t-set from 'lost' to anchor the new block.
                        # Pick the first set bit in 'lost'.
                        tmp = lost
                        anchor_idx = (tmp & -tmp).bit_length() - 1
                        anchor_tset = all_tsets[anchor_idx]
                        
                        # Generate candidates containing anchor_tset
                        # But we need to cover ALL of 'lost'.
                        # This is a constraint satisfaction problem.
                        # Heuristic: Try a few random blocks containing anchor_tset and check if they cover 'lost'.
                        
                        remaining_points = [x for x in range(v) if x not in anchor_tset]
                        found_replacement = False
                        
                        for _ in range(50):  # Try 50 random candidates
                            combo = random.sample(remaining_points, k - t)
                            block = list(anchor_tset) + list(combo)
                            bm = block_to_bitmask(block)
                            if (lost & ~bm) == 0:  # bm covers all of lost
                                # Check if this replacement is valid (doesn't break other coverage)
                                # Since 'lost' was the only thing i covered uniquely, and bm covers 'lost',
                                # the new union is (Union_others | bm).
                                # Since Union_others covered everything else, and bm covers 'lost',
                                # the new union is full_mask. So it's valid.
                                
                                # Does it allow further pruning?
                                # Maybe. Let's do the swap.
                                current_blocks[i] = sorted(block)
                                current_bitmasks[i] = bm
                                improved = True
                                found_replacement = True
                                break
                        
                        if found_replacement:
                            # Recalculate union_mask
                            union_mask = 0
                            for bm in current_bitmasks:
                                union_mask |= bm
                            break  # Restart loop
            
            # If no improvement in this neighborhood, increase r or break
            if not improved:
                r += 1
                if r > 2:  # Don't go too deep
                    break
            else:
                no_improve = 0
                r = 1  # Reset to smallest neighborhood
        
        return current_blocks, current_bitmasks, current_uncovered, union_mask

    best_blocks = None
    best_count = float('inf')
    
    # Multi-start
    while time.time() < deadline - 5:
        # 1. Greedy
        blocks, bitmasks, uncovered = greedy_construct()
        
        if uncovered != 0:
            # Greedy failed to cover everything (shouldn't happen if logic is correct)
            # Force cover remaining
            while uncovered != 0 and time.time() < deadline - 2:
                # Pick a random uncovered t-set
                idx = random.randint(0, n_tsets - 1)
                if (uncovered >> idx) & 1:
                    anchor_tset = all_tsets[idx]
                    remaining_points = [x for x in range(v) if x not in anchor_tset]
                    combo = random.sample(remaining_points, k - t)
                    block = sorted(list(anchor_tset) + list(combo))
                    bm = block_to_bitmask(block)
                    blocks.append(block)
                    bitmasks.append(bm)
                    uncovered &= ~bm
        
        if len(blocks) >= best_count:
            continue
            
        # 2. Local Search
        ls_blocks, ls_bitmasks, ls_uncovered, ls_union = local_search(
            blocks, bitmasks, uncovered, deadline - 2
        )
        
        if ls_uncovered == 0 and len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Fallback if no valid solution found (should not happen)
    if best_blocks is None:
        # Simple greedy without LS
        blocks, bitmasks, uncovered = greedy_construct()
        while uncovered != 0 and time.time() < deadline - 1:
            idx = random.randint(0, n_tsets - 1)
            if (uncovered >> idx) & 1:
                anchor_tset = all_tsets[idx]
                remaining_points = [x for x in range(v) if x not in anchor_tset]
                combo = random.sample(remaining_points, k - t)
                block = sorted(list(anchor_tset) + list(combo))
                bm = block_to_bitmask(block)
                blocks.append(block)
                bitmasks.append(bm)
                uncovered &= ~bm
        best_blocks = blocks
    
    return best_blocks
