import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_masks[i] is a bitset of t-subsets covered by block i
    block_masks = [0] * num_blocks
    # block_subset_indices[i] is a list of t-subset indices covered by block i
    block_subset_indices = [[] for _ in range(num_blocks)]
    
    # block_cover_masks[j] is a bitset of blocks covering t-subset j
    # This is a list of big integers, one for each t-subset
    # We need to build this efficiently.
    # Instead of iterating blocks then subsets, we can iterate subsets then blocks?
    # No, iterating blocks is better for generating subsets.
    
    # Initialize block_cover_masks with 0
    block_cover_masks = [0] * num_t_subsets
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs_idx = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            subs_idx.append(idx)
            # Set bit i in block_cover_masks[idx]
            block_cover_masks[idx] |= (1 << i)
        block_masks[i] = mask
        block_subset_indices[i] = subs_idx

    full_mask = (1 << num_t_subsets) - 1
    
    # Helper: Reduce redundant blocks
    def reduce_blocks(block_indices):
        if not block_indices:
            return []
        
        # Count coverage for each t-subset
        cover_count = [0] * num_t_subsets
        
        for idx in block_indices:
            for s_idx in block_subset_indices[idx]:
                cover_count[s_idx] += 1
                
        new_blocks = []
        for idx in block_indices:
            subs = block_subset_indices[idx]
            # Check if any subset is covered only by this block
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # It is redundant, do not add to new_blocks
                pass
            else:
                new_blocks.append(idx)
                
        return new_blocks

    # Helper: Greedy Set Cover
    def greedy_cover():
        uncovered = full_mask
        selected = []
        
        while uncovered:
            best_gain = 0
            best_idx = -1
            
            # Find block with max gain
            # Optimization: We can break early if gain is max possible?
            # Max possible is popcount(uncovered).
            # But we just need the best.
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    # If best_gain is 0, we are stuck (should not happen)
                    # If best_gain is small, we still continue to find max.
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected

    # Helper: Find candidates for a set of t-subsets
    # Returns a list of block indices that cover all subsets in E
    def find_candidates(E):
        if not E:
            return list(range(num_blocks))
        
        # Sort E by the number of blocks covering them (smallest first) to minimize intersection
        # We can approximate size by bit_count of block_cover_masks
        # But calculating bit_count for 3876 masks is expensive if done every time.
        # Let's just intersect in the given order, or random order.
        # For 2-exchange, E is small.
        
        # Start with the first element's mask
        current_mask = block_cover_masks[E[0]]
        
        for e in E[1:]:
            current_mask &= block_cover_masks[e]
            if current_mask == 0:
                return []
                
        # Extract indices from current_mask
        # This is slow if the mask is dense.
        # But if it's dense, we have many candidates.
        # We just need one.
        # If we need to pick randomly, we need to know the count or list.
        # For now, let's return a list of indices.
        # To get indices from a bitmask:
        indices = []
        m = current_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            # Optimization: If we only need a few, we can stop?
            # But for random selection, we need the list or a way to pick randomly.
            # If the list is huge, picking randomly from the bitmask is hard.
            # However, if the intersection is large, any block is fine.
            # If it's small, the list is small.
            # Let's assume the list is manageable.
            
        return indices

    # Initial Solution
    current_blocks = greedy_cover()
    current_blocks = reduce_blocks(current_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # Local Search
    # We maintain cover_count to quickly identify unique elements
    # But reduce_blocks recalculates it.
    # For local search, we need to know unique elements of a block in the current cover.
    
    def get_unique_elements(block_idx, current_cover):
        # Returns list of t-subset indices covered ONLY by block_idx in current_cover
        # We need to know if any other block in current_cover covers them.
        # This is O(|current_cover| * 35) if done naively.
        # We can maintain a global cover_count array.
        pass

    # Let's maintain cover_count dynamically.
    cover_count = [0] * num_t_subsets
    for idx in current_blocks:
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += 1

    def update_cover_count(idx, delta):
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += delta

    def is_redundant(idx):
        for s_idx in block_subset_indices[idx]:
            if cover_count[s_idx] == 1:
                return False
        return True

    def try_reduce():
        # Remove redundant blocks
        new_blocks = []
        for idx in current_blocks:
            if is_redundant(idx):
                update_cover_count(idx, -1)
            else:
                new_blocks.append(idx)
        return new_blocks

    # Reset cover_count for current_blocks
    cover_count = [0] * num_t_subsets
    for idx in current_blocks:
        for s_idx in block_subset_indices[idx]:
            cover_count[s_idx] += 1

    # Local Search Loop
    no_improvement_count = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Delete redundant blocks
        new_blocks = try_reduce()
        if len(new_blocks) < len(current_blocks):
            current_blocks = new_blocks
            improved = True
            if len(current_blocks) < best_size:
                best_blocks = list(current_blocks)
                best_size = len(best_blocks)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            continue # Restart loop to re-evaluate
        
        # 2. 1-Swap: Replace a block with another that covers its unique elements
        # We iterate through blocks and try to swap
        # To avoid O(N^2) swaps in one pass, we pick random blocks or iterate
        # Let's iterate through blocks in random order
        
        indices_to_try = list(range(len(current_blocks)))
        random.shuffle(indices_to_try)
        
        for pos in indices_to_try:
            if time.time() > deadline:
                break
                
            b_idx = current_blocks[pos]
            
            # Find unique elements of b_idx
            unique_e = []
            for s_idx in block_subset_indices[b_idx]:
                if cover_count[s_idx] == 1:
                    unique_e.append(s_idx)
                    
            if not unique_e:
                # Redundant, remove it
                update_cover_count(b_idx, -1)
                current_blocks.pop(pos)
                improved = True
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                break # Restart loop as indices changed
            
            # Find candidates that cover unique_e
            candidates = find_candidates(unique_e)
            
            # Filter out b_idx itself
            if b_idx in candidates:
                candidates.remove(b_idx)
                
            if candidates:
                # Pick a random candidate
                new_idx = random.choice(candidates)
                
                # Perform swap
                # Remove b_idx
                update_cover_count(b_idx, -1)
                # Add new_idx
                update_cover_count(new_idx, 1)
                current_blocks[pos] = new_idx
                
                # Check if this swap allows reduction (i.e., if new_idx is redundant? No, it covers unique_e so it's not redundant immediately.
                # But it might make other blocks redundant?
                # Let's just accept the swap and let the next iteration's reduction handle it.
                # Or check if size can be reduced.
                # Actually, 1-swap keeps size constant. It helps escape local minima.
                improved = True
                no_improvement_count += 1 # Swapping doesn't reduce size, so count as no improvement for "best"
                # But it might lead to a better state.
                # Let's not update best_blocks unless size decreases.
                
        # 3. 2-Exchange: Remove 2, Add 1
        # Pick random pairs
        # Limit number of attempts per iteration to avoid timeout
        max_2ex_attempts = 50
        attempts = 0
        while attempts < max_2ex_attempts and time.time() < deadline:
            attempts += 1
            if len(current_blocks) < 2:
                break
                
            # Pick two distinct blocks
            i1, i2 = random.sample(range(len(current_blocks)), 2)
            b1 = current_blocks[i1]
            b2 = current_blocks[i2]
            
            # Find unique elements of b1 U b2
            # An element is unique to {b1, b2} if cover_count[e] <= 2 and (b1 covers e or b2 covers e)
            # Actually, if cover_count[e] == 1, it's unique to one of them.
            # If cover_count[e] == 2, it's covered by exactly two blocks.
            # If we remove b1 and b2, e becomes uncovered if cover_count[e] <= 2 and e is in b1 or b2.
            
            unique_e_12 = []
            # We need to know which elements are covered by b1 or b2 and have count <= 2
            # We can iterate through subsets of b1 and b2
            
            # To avoid duplicates, use a set
            e_set = set()
            for s_idx in block_subset_indices[b1]:
                if cover_count[s_idx] <= 2:
                    e_set.add(s_idx)
            for s_idx in block_subset_indices[b2]:
                if cover_count[s_idx] <= 2:
                    e_set.add(s_idx)
                    
            if not e_set:
                continue
                
            unique_e_12 = list(e_set)
            
            # Find candidates that cover unique_e_12
            candidates = find_candidates(unique_e_12)
            
            # Filter out b1 and b2
            candidates = [c for c in candidates if c != b1 and c != b2]
            
            if candidates:
                new_idx = random.choice(candidates)
                
                # Perform 2-Exchange
                # Remove b1, b2
                update_cover_count(b1, -1)
                update_cover_count(b2, -1)
                # Add new_idx
                update_cover_count(new_idx, 1)
                
                # Remove b1, b2 from current_blocks
                # Need to be careful with indices
                # Sort indices to remove larger first
                if i1 > i2:
                    i1, i2 = i2, i1
                current_blocks.pop(i2)
                current_blocks.pop(i1)
                current_blocks.append(new_idx)
                
                improved = True
                if len(current_blocks) < best_size:
                    best_blocks = list(current_blocks)
                    best_size = len(best_blocks)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                break # Restart loop after a successful 2-exchange
                
        if not improved:
            no_improvement_count += 1
            # If stuck for a long time, we could try random restart or perturbation
            # But for now, just continue.
            if no_improvement_count > 100:
                # Try a random perturbation: remove a random block and repair
                # This is expensive, so maybe skip.
                pass

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    return final_blocks
