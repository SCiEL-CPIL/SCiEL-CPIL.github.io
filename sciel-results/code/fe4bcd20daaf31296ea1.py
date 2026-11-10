import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Precomputation
    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # All k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmasks for each block
    # mask[i] is an integer where bit j is set if block i covers t-subset j
    block_masks = [0] * num_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << num_tsubs) - 1
    
    # 2. Initial Valid Solution
    # We need a valid covering to start. A simple greedy gives a valid one.
    # To make the SA start "random" enough, we can run a randomized greedy.
    
    def greedy_randomized():
        covered = 0
        selected = []
        remaining_blocks = list(range(num_blocks))
        random.shuffle(remaining_blocks)
        
        while covered != full_mask:
            best_gain = -1
            best_block = -1
            
            # Optimization: only check blocks that cover at least one uncovered triple
            # But checking all is safer for correctness in the first few steps.
            # For speed, we can sample or use the inverted index.
            # Given the time limit, let's use a simple pass over remaining blocks.
            # 346k blocks * ~70 steps = 24M operations. Might be slow in Python if done naively.
            
            # Better: Inverted index for greedy.
            # But for simplicity and to avoid complex data structures in the inner loop of SA,
            # let's just do a few greedy steps with random tie-breaking.
            
            # Actually, let's just pick a random block that covers the most uncovered triples.
            # To speed up, we can precompute for each triple which blocks cover it.
            
            # Let's build inverted index once.
            # triple_to_blocks = {triple_idx: [block_idx, ...]}
            # This is memory heavy but fast access.
            
            pass # Placeholder, we'll implement a faster greedy below
            
        return selected

    # Faster Greedy using Inverted Index
    triple_to_blocks = [[] for _ in range(num_tsubs)]
    for bi, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(bi)
            m ^= lsb

    def run_greedy():
        covered = 0
        selected = []
        used_blocks = set()
        
        while covered != full_mask and time.time() < deadline - 60:
            uncovered = full_mask ^ covered
            if uncovered == 0:
                break
                
            # Pick a random uncovered triple
            # Iterate bits to find one? Or just random index check.
            # Random index check is O(1) expected if density is high, but here we start with 0.
            # Better: Pick a random triple from the list of uncovered ones.
            # We don't have a list of uncovered ones maintained.
            # Let's just pick a random block index from all_blocks? No.
            
            # Let's pick a random triple index from 0..num_tsubs-1 and check if uncovered.
            # If not, retry.
            while True:
                t_idx = random.randint(0, num_tsubs - 1)
                if (uncovered >> t_idx) & 1:
                    break
            
            candidates = triple_to_blocks[t_idx]
            best_gain = -1
            best_block = -1
            
            for bi in candidates:
                if bi in used_blocks:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    
            if best_block == -1:
                # Fallback: any block covering t_idx
                best_block = candidates[0]
                
            selected.append(best_block)
            used_blocks.add(best_block)
            covered |= block_masks[best_block]
            
        return selected

    # Get an initial solution
    initial_solution = run_greedy()
    if not initial_solution:
        # Fallback to deterministic greedy if random failed or timed out
        # This shouldn't happen if we have time.
        return []
        
    current_blocks = set(initial_solution)
    current_mask = 0
    for bi in current_blocks:
        current_mask |= block_masks[bi]
        
    # Ensure validity
    if current_mask != full_mask:
        # Complete it
        while current_mask != full_mask:
            uncovered = full_mask ^ current_mask
            while True:
                t_idx = random.randint(0, num_tsubs - 1)
                if (uncovered >> t_idx) & 1:
                    break
            candidates = triple_to_blocks[t_idx]
            best_gain = -1
            best_block = -1
            for bi in candidates:
                if bi in current_blocks:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            if best_block == -1:
                best_block = candidates[0]
            current_blocks.add(best_block)
            current_mask |= block_masks[best_block]

    best_solution = list(current_blocks)
    best_count = len(best_solution)
    
    # 3. Simulated Annealing
    # Objective: Minimize number of blocks.
    # Moves:
    # 1. Remove a block (if valid, i.e., it doesn't uncover any triple).
    # 2. Swap a block with another (if valid).
    # 3. (Optional) Add a block (to enable future removals).
    
    # To check validity of removal:
    # A block B is redundant if for every triple T in B, there is another block B' in the set covering T.
    # This is equivalent to: mask(B) is a subset of the union of masks of all other blocks.
    # Or: (mask(B) & ~(union_others)) == 0.
    
    # We maintain the current set of blocks and their union mask.
    
    # To speed up removal checks, we can precompute for each triple in the current set,
    # the count of blocks covering it.
    # If count[T] == 1 for any T in B, then B is not redundant.
    
    def get_redundant_blocks(block_set):
        redundant = []
        # Count occurrences of each triple
        counts = [0] * num_tsubs
        for bi in block_set:
            m = block_masks[bi]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        # Check each block
        for bi in block_set:
            m = block_masks[bi]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    break
                m ^= lsb
            else:
                # No triple has count 1, so block is redundant
                redundant.append(bi)
                
        return redundant

    # Initial redundancy removal
    redundant = get_redundant_blocks(current_blocks)
    for bi in redundant:
        current_blocks.discard(bi)
    # Recalculate mask
    current_mask = 0
    for bi in current_blocks:
        current_mask |= block_masks[bi]
        
    if len(current_blocks) < best_count:
        best_count = len(current_blocks)
        best_solution = list(current_blocks)

    # SA Parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.99
    max_iters = 10000
    
    current_blocks_set = set(current_blocks)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to remove a block
            if not current_blocks_set:
                continue
            bi = random.choice(list(current_blocks_set))
            
            # Check if redundant
            # Quick check: does it contain any unique triple?
            # We can check counts. But counts are expensive to maintain.
            # Let's just try removing and checking mask.
            # New mask = current_mask & ~block_masks[bi]
            # If new_mask == current_mask, it was redundant? No.
            # If new_mask == current_mask, it means block_masks[bi] was subset of others.
            # Wait, if we remove bi, the new union is (old_union & ~mask_bi).
            # If (old_union & ~mask_bi) == old_union, then mask_bi was 0? No.
            # If mask_bi is subset of (old_union \ mask_bi), then removing it doesn't change the union.
            # So, if (current_mask & ~block_masks[bi]) == current_mask, it is redundant.
            
            new_mask = current_mask & ~block_masks[bi]
            if new_mask == current_mask:
                # Valid removal
                current_blocks_set.discard(bi)
                current_mask = new_mask
                if len(current_blocks_set) < best_count:
                    best_count = len(current_blocks_set)
                    best_solution = list(current_blocks_set)
            else:
                # Invalid, ignore or penalize?
                # In SA, we usually only accept valid moves or moves that improve objective.
                # Here, validity is a hard constraint. So we only accept valid moves.
                pass
                
        elif move_type < 0.8:
            # Try to swap a block
            if not current_blocks_set:
                continue
            bi_old = random.choice(list(current_blocks_set))
            
            # Pick a random new block
            bi_new = random.randint(0, num_blocks - 1)
            if bi_new == bi_old:
                continue
            
            # Check if swapping maintains validity
            # New mask = (current_mask & ~block_masks[bi_old]) | block_masks[bi_new]
            # We need new_mask == full_mask
            
            # Optimization:
            # Let M_old = block_masks[bi_old]
            # Let M_new = block_masks[bi_new]
            # Let M_rest = current_mask & ~M_old
            # We need M_rest | M_new == full_mask
            # This is equivalent to: (full_mask & ~M_rest) is a subset of M_new
            # i.e., all triples not covered by others must be covered by M_new.
            
            M_old = block_masks[bi_old]
            M_new = block_masks[bi_new]
            M_rest = current_mask & ~M_old
            
            # Uncovered by rest:
            missing = full_mask & ~M_rest
            if (missing & ~M_new) == 0:
                # Valid swap
                current_blocks_set.discard(bi_old)
                current_blocks_set.add(bi_new)
                current_mask = M_rest | M_new
                
                # Check for redundancy after swap?
                # Maybe do it occasionally.
                if random.random() < 0.05:
                    redundant = get_redundant_blocks(current_blocks_set)
                    for b in redundant:
                        current_blocks_set.discard(b)
                    current_mask = 0
                    for b in current_blocks_set:
                        current_mask |= block_masks[b]
                        
                if len(current_blocks_set) < best_count:
                    best_count = len(current_blocks_set)
                    best_solution = list(current_blocks_set)
                    
        else:
            # Try to add a block (to escape local minima where removals are blocked)
            # This increases size, so we only accept if it leads to a better state later?
            # In SA, we can accept worse states with probability exp(-delta/T).
            # But here, "worse" means larger size.
            # However, adding a block doesn't violate validity.
            # We add a random block.
            
            bi_new = random.randint(0, num_blocks - 1)
            if bi_new in current_blocks_set:
                continue
                
            # Calculate delta in cost (number of blocks)
            # Cost = -len(blocks) ? Or just len(blocks).
            # We want to minimize len(blocks).
            # Adding a block increases len by 1.
            # We accept with prob exp(-1/T).
            
            if random.random() < (1.0 / T): # Rough approximation of exp(-1/T) for small T?
                # Actually, standard SA: accept if exp(-delta/E) > random.
                # delta = 1 (increase in blocks).
                # Accept if exp(-1/T) > random.
                if random.random() < (2.718281828 ** (-1.0 / T)):
                    current_blocks_set.add(bi_new)
                    current_mask |= block_masks[bi_new]
                    
                    # Try to remove something?
                    # No, just keep it.
                    
        T *= cooling_rate

    # Final cleanup: remove any remaining redundant blocks
    redundant = get_redundant_blocks(current_blocks_set)
    for bi in redundant:
        current_blocks_set.discard(bi)
        
    if len(current_blocks_set) < best_count:
        best_count = len(current_blocks_set)
        best_solution = list(current_blocks_set)

    # Verify final solution
    final_mask = 0
    for bi in best_solution:
        final_mask |= block_masks[bi]
        
    if final_mask != full_mask:
        # Fallback to greedy if SA broke it (should not happen with valid moves)
        # But just in case, use the initial greedy solution
        return [list(all_blocks[bi]) for bi in initial_solution]

    return [list(all_blocks[bi]) for bi in best_solution]

if __name__ == "__main__":
    # Test with small case
    # v=7, k=4, t=3
    start = time.time()
    blocks = construct(7, 4, 3)
    end = time.time()
    print(f"Time: {end-start:.2f}s, Blocks: {len(blocks)}")
    
    # Verify
    all_tsubs = set(itertools.combinations(range(7), 3))
    covered = set()
    for b in blocks:
        for tsub in itertools.combinations(b, 3):
            covered.add(tsub)
    if all_tsubs == covered:
        print("Valid")
    else:
        print("Invalid")
        
    # Main task
    # start = time.time()
    # blocks = construct(24, 7, 3)
    # end = time.time()
    # print(f"Time: {end-start:.2f}s, Blocks: {len(blocks)}")
