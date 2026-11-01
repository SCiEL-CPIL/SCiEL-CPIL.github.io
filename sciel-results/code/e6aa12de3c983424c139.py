import itertools
import time
import random

# Precomputation
v, k, t = 19, 7, 4
# Map 4-subsets to indices
subsets_4 = list(itertools.combinations(range(v), t))
subset_to_idx = {s: i for i, s in enumerate(subsets_4)}
N_subsets = len(subsets_4)

# Generate all 7-subsets and their masks
blocks_7 = list(itertools.combinations(range(v), k))
block_masks = []
block_subsets = [] # Store the actual 4-subsets indices for fast count update
for b in blocks_7:
    mask = 0
    idxs = []
    for s in itertools.combinations(b, t):
        idx = subset_to_idx[s]
        mask |= (1 << idx)
        idxs.append(idx)
    block_masks.append(mask)
    block_subsets.append(idxs)

N_blocks = len(blocks_7)

# Greedy with Randomization
def solve():
    deadline = time.time() + 115
    best_blocks = None
    best_len = float('inf')
    
    # Initial full mask
    full_mask = (1 << N_subsets) - 1
    
    while time.time() < deadline:
        uncovered = full_mask
        used_blocks = []
        cover_counts = [0] * N_subsets
        
        # 1. Greedy Construction
        while uncovered:
            best_gain = -1
            candidates = []
            # Iterate all blocks
            for i, mask in enumerate(block_masks):
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Pick randomly from candidates to diversify
            idx = random.choice(candidates)
            used_blocks.append(idx)
            
            # Update uncovered
            uncovered &= ~block_masks[idx]
            
            # Update cover counts
            for s_idx in block_subsets[idx]:
                cover_counts[s_idx] += 1
        
        # 2. Local Optimization
        # We perform redundancy removal and swaps.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 2a. Redundancy Removal
            # We need to check if a block is redundant.
            # A block is redundant if all its subsets have count > 1.
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for idx in used_blocks:
                    is_redundant = True
                    for s_idx in block_subsets[idx]:
                        if cover_counts[s_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        # Remove it
                        for s_idx in block_subsets[idx]:
                            cover_counts[s_idx] -= 1
                        changed = True
                        improved = True
                    else:
                        new_blocks.append(idx)
                used_blocks = new_blocks

            if not used_blocks:
                break

            # 2b. 1-for-1 Swaps
            # For each block in the cover, try to replace it with another block
            # such that coverage is maintained.
            # Condition: All t-subsets of B_i that are uniquely covered by B_i 
            # must be covered by the new block B_j.
            
            # To make this efficient, we iterate through blocks in used_blocks.
            # We need to be careful with indices if we modify the list.
            # We'll iterate over a copy of the indices or just mark swaps.
            
            # Optimization: Pre-compute the "unique" subsets for each block in the current cover?
            # That's O(N_cover * 35). N_cover ~ 150. 150*35 = 5250. Fast.
            
            unique_subsets = {}
            for idx in used_blocks:
                u_list = []
                for s_idx in block_subsets[idx]:
                    if cover_counts[s_idx] == 1:
                        u_list.append(s_idx)
                if u_list:
                    unique_subsets[idx] = u_list
                else:
                    # Should have been removed by redundancy removal, but just in case
                    pass

            # Now try swaps
            # We iterate over blocks. If we find a swap, we apply it and restart the local optimization loop.
            # To avoid O(N_blocks) check for every block, we can just do it.
            # N_blocks = 50k. N_cover = 150. 150 * 50k = 7.5M checks.
            # Each check: verify if unique_subsets[idx] are subset of block_subsets[j].
            # If unique_subsets[idx] is small (size 1-2), this is very fast.
            
            # However, doing this for EVERY restart might be slow.
            # We can limit the number of swap attempts or randomize the order.
            
            found_swap = False
            # Randomize order of blocks to try to break symmetry
            block_order = list(used_blocks)
            random.shuffle(block_order)
            
            for i_idx in block_order:
                if i_idx not in unique_subsets:
                    continue
                u_list = unique_subsets[i_idx]
                u_set = set(u_list)
                
                # Find a replacement block j
                # j must not be in used_blocks
                # j must cover all subsets in u_set
                
                # Optimization: We can check if any block in used_blocks (other than i_idx) already covers u_set?
                # No, we want to ADD a new block or SWAP with an existing one?
                # If we swap with an existing block j, we remove i_idx and keep j.
                # But j is already there. So we are effectively just removing i_idx if j covers u_set?
                # No, if j covers u_set, and j is already in the cover, then u_set is covered by j.
                # But u_set consists of subsets with count 1. If j covers them, their count would be at least 2 (from i_idx and j).
                # Contradiction. So no existing block covers u_set.
                # Therefore, we MUST add a NEW block j not in used_blocks.
                
                # So we iterate over all blocks not in used_blocks.
                used_set = set(used_blocks)
                
                for j_idx in range(N_blocks):
                    if j_idx in used_set:
                        continue
                    
                    # Check if block j covers all subsets in u_list
                    # u_list is a list of indices.
                    # We can check if (mask_j & mask_u) == mask_u
                    # But we don't have mask_u precomputed.
                    # u_list is small. We can just check if all s in u_list are in block_subsets[j_idx].
                    
                    # Optimization: If u_list has 1 element, check if that element is in block_subsets[j_idx].
                    # If u_list has 2, check both.
                    
                    covers = True
                    for s_idx in u_list:
                        # Check if s_idx is in block_subsets[j_idx]
                        # block_subsets[j_idx] is a list.
                        # We can pre-compute a set for each block? Or just use 'in'.
                        # 'in' on a list of 35 items is fast.
                        if s_idx not in block_subsets[j_idx]:
                            covers = False
                            break
                    
                    if covers:
                        # Perform swap: Remove i_idx, Add j_idx
                        # Remove i_idx
                        used_blocks.remove(i_idx)
                        for s_idx in block_subsets[i_idx]:
                            cover_counts[s_idx] -= 1
                        
                        # Add j_idx
                        used_blocks.append(j_idx)
                        for s_idx in block_subsets[j_idx]:
                            cover_counts[s_idx] += 1
                        
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    break
            
            if found_swap:
                continue # Restart local optimization loop

            # 2c. 2-for-1 Replacement (Optional, if time permits)
            # Try to remove 2 blocks and add 1.
            # This is much more expensive. O(N_cover^2 * N_blocks).
            # 150^2 * 50k = 1.1B. Too slow.
            # We can restrict to pairs that are "close" or just sample pairs.
            # Given the 1-for-1 improvement, we might not need this.
            # Let's stick to 1-for-1 for now to ensure speed.

        if len(used_blocks) < best_len:
            best_len = len(used_blocks)
            best_blocks = [blocks_7[i] for i in used_blocks]
            if best_len <= 131:
                break

    return best_blocks

def construct(v, k, t):
    if v == 19 and k == 7 and t == 4:
        return solve()
    else:
        # Fallback to simple greedy for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            max_covered = 0
            for b in itertools.combinations(range(v), k):
                covered = 0
                for s in itertools.combinations(b, t):
                    if s in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block = b
            if not best_block:
                break
            blocks.append(list(best_block))
            for s in itertools.combinations(best_block, t):
                uncovered.discard(s)
        return blocks
