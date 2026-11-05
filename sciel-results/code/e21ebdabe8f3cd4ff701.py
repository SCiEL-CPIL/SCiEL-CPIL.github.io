"""Covering design C(24,7,3) using Bitset Greedy + Local Search with Swaps."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0

    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    full_mask = (1 << num_triples) - 1

    def get_block_mask(block):
        """Compute the bitset of triples covered by a block."""
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(sorted(triple))]
            mask |= (1 << idx)
        return mask

    # 1. Greedy Construction
    uncovered = full_mask
    blocks = []
    block_masks = [] # List of masks corresponding to blocks

    while uncovered:
        if time.time() - start_time > timeout:
            break
        
        # Pick a random uncovered triple
        anchor_idx = -1
        for _ in range(100):
            idx = random.randint(0, num_triples - 1)
            if (uncovered >> idx) & 1:
                anchor_idx = idx
                break
        if anchor_idx == -1:
            # Fallback: lowest set bit
            temp = uncovered
            if temp:
                lsb = temp & -temp
                anchor_idx = lsb.bit_length() - 1
            else:
                break

        anchor_triple = all_triples[anchor_idx]
        anchor_set = set(anchor_triple)
        available_points = [p for p in range(v) if p not in anchor_set]
        
        best_block = None
        best_gain = -1
        best_mask = 0
        
        # To speed up, we can sample a subset of candidates if the full set is too large?
        # C(21, 4) = 5985. This is small enough to iterate fully.
        for comb in itertools.combinations(available_points, k - t):
            candidate_block = tuple(sorted(anchor_triple + comb))
            mask = get_block_mask(candidate_block)
            gain = (mask & uncovered).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
                best_mask = mask

        if best_block is None:
            break
            
        blocks.append(list(best_block))
        block_masks.append(best_mask)
        uncovered &= ~best_mask

    if not blocks:
        return []

    # 2. Initial Cleanup: Remove redundant blocks
    def remove_redundant_blocks(blocks, block_masks):
        changed = True
        while changed:
            changed = False
            total_mask = 0
            for m in block_masks:
                total_mask |= m
            
            for i in range(len(blocks)):
                if time.time() - start_time > timeout:
                    return blocks, block_masks
                
                # Check if block i is redundant
                # If total_mask without mask[i] is still full_mask, it's redundant
                mask_wo_i = total_mask & ~block_masks[i]
                if mask_wo_i == full_mask:
                    # Remove block i
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break # Restart loop
        
        return blocks, block_masks

    blocks, block_masks = remove_redundant_blocks(blocks, block_masks)
    
    # 3. Local Search: Swap and Remove
    # We try to swap a block with another block (or a mutated block) 
    # such that the new set is valid and allows removing a redundant block.
    
    best_blocks = blocks
    best_masks = block_masks
    best_size = len(blocks)
    
    # Helper to check if a set of masks covers everything
    def is_valid(masks):
        total = 0
        for m in masks:
            total |= m
            if total == full_mask:
                return True
        return total == full_mask

    # We will maintain the current state and try to improve it.
    current_blocks = list(best_blocks)
    current_masks = list(best_masks)
    
    # Precompute a set of "good" candidate blocks?
    # Or just mutate existing blocks.
    
    iterations = 0
    max_iterations = 1000 # Safety cap
    
    while time.time() - start_time < timeout and iterations < max_iterations:
        iterations += 1
        
        if len(current_blocks) == best_size:
            # Try to find a swap that enables a removal
            if not current_blocks: break
            
        # Pick a random block to swap out
        i = random.randint(0, len(current_blocks) - 1)
        old_block = current_blocks[i]
        old_mask = current_masks[i]
        
        # Generate a candidate replacement
        # Strategy: Mutate the block by changing 1 or 2 points
        # Or pick a random block from the current set (swap with another)
        # Let's try mutation first as it explores new space.
        
        candidate_block = None
        if random.random() < 0.5:
            # Mutate: change 1 point
            points_to_change = random.randint(1, 2)
            new_block_set = set(old_block)
            for _ in range(points_to_change):
                if not new_block_set: break
                p_to_remove = random.choice(list(new_block_set))
                new_block_set.remove(p_to_remove)
                available = [p for p in range(v) if p not in new_block_set]
                if not available: break
                p_to_add = random.choice(available)
                new_block_set.add(p_to_add)
            
            if len(new_block_set) == k:
                candidate_block = tuple(sorted(new_block_set))
        else:
            # Swap with another block in the set
            j = random.randint(0, len(current_blocks) - 1)
            if j != i:
                candidate_block = tuple(current_blocks[j])
        
        if candidate_block is None:
            continue
            
        # Check if swapping block i with candidate_block is valid
        # We need to compute the mask of candidate_block
        # To avoid recomputing, we could cache, but let's just compute.
        # Optimization: If candidate_block is in current_blocks, we know its mask.
        if candidate_block in current_blocks:
            # Find its index
            for j, b in enumerate(current_blocks):
                if tuple(b) == candidate_block:
                    cand_mask = current_masks[j]
                    break
            else:
                cand_mask = get_block_mask(candidate_block)
        else:
            cand_mask = get_block_mask(candidate_block)
        
        # Check validity of the new set
        # New set masks = current_masks without i, plus cand_mask
        # If this set covers full_mask, it's valid.
        
        # Efficient check:
        # total_mask_current = OR of all current_masks
        # We can maintain total_mask? 
        # Let's just compute OR of the new set.
        # It's O(N) bitwise ORs. N is ~76. Very fast.
        
        new_masks = [m for j, m in enumerate(current_masks) if j != i]
        new_masks.append(cand_mask)
        
        # Check if new_masks cover full_mask
        total = 0
        valid = False
        for m in new_masks:
            total |= m
            if total == full_mask:
                valid = True
                break
        
        if not valid:
            continue
            
        # The swap is valid. Now check if we can remove any block from this new set.
        # We can just run the redundancy removal on this new set.
        # But we want to keep the state consistent.
        
        # Let's tentatively apply the swap
        new_blocks = [b for j, b in enumerate(current_blocks) if j != i]
        new_blocks.append(list(candidate_block))
        
        # Run redundancy removal on the new set
        # We need to pass masks too.
        # We have new_masks.
        
        # We need a helper that takes blocks and masks and removes redundant ones.
        # Let's inline a fast version.
        
        changed = True
        while changed:
            changed = False
            total_m = 0
            for m in new_masks:
                total_m |= m
            
            for j in range(len(new_blocks)):
                mask_wo_j = total_m & ~new_masks[j]
                if mask_wo_j == full_mask:
                    # Remove j
                    new_blocks.pop(j)
                    new_masks.pop(j)
                    changed = True
                    break
        
        # If we ended up with fewer blocks, accept the move
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            current_masks = new_masks
            best_blocks = new_blocks
            best_masks = new_masks
            best_size = len(new_blocks)
            
    return best_blocks

# Test locally
if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    
    # Verify
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, 3):
            covered.add(tuple(sorted(c)))
    
    all_subsets = set(itertools.combinations(range(24), 3))
    if covered == all_subsets:
        print("Valid covering")
    else:
        print("Invalid covering")
        print(f"Missing: {len(all_subsets - covered)}")
