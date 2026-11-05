import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[b_idx] = mask
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    def greedy_set_cover(seed=0, use_ratio=False):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_blocks = []
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_score = -1
            best_idx = -1
            candidates = []
            
            # Find block with best score (max gain or max gain/point)
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain == 0:
                    continue
                if use_ratio:
                    score = gain / k  # all blocks have same size, so this is just gain
                    # Actually for ratio, we should consider "efficiency" differently
                    # Let's use gain / (number of points in block that are "useful")
                    # But all blocks have k points. So ratio = gain/k is same as gain.
                    # A better ratio: gain / (overlap + 1) or similar
                    # Let's stick with pure gain but add randomization for ties
                    score = gain
                else:
                    score = gain
                    
                if score > best_score:
                    best_score = score
                    best_idx = b_idx
                    candidates = [b_idx]
                elif score == best_score and score > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_score == 0:
                break
                
            # Randomly pick from candidates to diversify
            chosen_idx = rng.choice(candidates)
            
            solution_blocks.append(all_blocks[chosen_idx])
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_blocks

    def greedy_ratio(seed=0):
        """Greedy with ratio-based selection: maximize gain per 'new' point."""
        rng = random.Random(seed)
        uncovered = all_mask
        solution_blocks = []
        solution_indices = []
        used_points = set()
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_score = -1
            best_idx = -1
            candidates = []
            
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain == 0:
                    continue
                # Count how many points in this block are already "used"
                block = all_blocks[b_idx]
                new_points = 0
                for p in block:
                    if p not in used_points:
                        new_points += 1
                # Ratio: gain per new point (with minimum of 1 to avoid div by zero)
                score = gain / max(1, new_points)
                
                if score > best_score:
                    best_score = score
                    best_idx = b_idx
                    candidates = [b_idx]
                elif abs(score - best_score) < 1e-9 and score > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_score == 0:
                break
                
            chosen_idx = rng.choice(candidates)
            solution_blocks.append(all_blocks[chosen_idx])
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            for p in all_blocks[chosen_idx]:
                used_points.add(p)
            
        return solution_blocks

    def refine_solution(blocks):
        # Convert to list of lists for mutability
        blocks = [list(b) for b in blocks]
        
        # Helper to calculate coverage counts
        def get_counts(blk_list):
            counts = np.zeros(n_triplets, dtype=np.int32)
            for b in blk_list:
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    counts[tid] += 1
            return counts

        # Helper to get exclusive triplets for a block given counts
        def get_exclusive(block, counts):
            exc = []
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if counts[tid] == 1:
                    exc.append(tid)
            return exc

        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            
            # 1. Remove redundant blocks
            counts = get_counts(blocks)
            new_blocks = []
            for b in blocks:
                redundant = True
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
                else:
                    changed = True
            blocks = new_blocks
            
            if not blocks:
                break

            # 2. Try merge 2-for-1
            counts = get_counts(blocks)
            exclusives = []
            for b in blocks:
                exclusives.append(get_exclusive(b, counts))
                
            merged = False
            n_sol = len(blocks)
            
            for i in range(n_sol):
                if time.time() > deadline - 5:
                    break
                if not exclusives[i]:
                    continue
                    
                for j in range(i + 1, n_sol):
                    if time.time() > deadline - 5:
                        break
                    if not exclusives[j]:
                        continue
                    
                    points_needed = set()
                    for tid in exclusives[i]:
                        points_needed.update(all_triplets[tid])
                    for tid in exclusives[j]:
                        points_needed.update(all_triplets[tid])
                        
                    if len(points_needed) > k:
                        continue
                        
                    cand_set = set(points_needed)
                    for p in range(v):
                        if len(cand_set) == k:
                            break
                        if p not in cand_set:
                            cand_set.add(p)
                            
                    if len(cand_set) != k:
                        continue
                        
                    new_block = tuple(sorted(cand_set))
                    
                    new_blocks_list = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks_list.append(b)
                    new_blocks_list.append(list(new_block))
                    
                    blocks = new_blocks_list
                    merged = True
                    changed = True
                    break
                if merged:
                    break
            if merged:
                continue

            # 3. Try swap: replace one block with another that covers more uncovered triplets
            counts = get_counts(blocks)
            n_sol = len(blocks)
            swapped = False
            for i in range(n_sol):
                if time.time() > deadline - 5:
                    break
                # What would be uncovered if we remove block i?
                excl_i = get_exclusive(blocks[i], counts)
                if not excl_i:
                    continue
                    
                # Find a block that covers all of excl_i and adds new coverage
                best_swap_idx = -1
                best_new_gain = 0
                for b_idx in range(n_blocks):
                    if b_idx in [blocks.index(b) for b in blocks if b == list(all_blocks[b_idx])]:
                        continue
                    # Check if this block covers all exclusive triplets of i
                    covers_all = True
                    for tid in excl_i:
                        if not (block_masks[b_idx] >> tid) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Calculate additional coverage from this block
                    add_gain = (block_masks[b_idx] & ~block_masks[blocks.index(blocks[i])]).bit_count()
                    # Actually we need to compute what this block covers that's not covered by others
                    # Simplified: just check if it covers excl_i and has some new triplets
                    new_gain = (block_masks[b_idx] & (all_mask & ~block_masks[blocks.index(blocks[i])])).bit_count()
                    if new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_swap_idx = b_idx
                        
                if best_swap_idx >= 0 and best_new_gain > 0:
                    blocks[i] = list(all_blocks[best_swap_idx])
                    swapped = True
                    changed = True
                    break
            if swapped:
                continue

        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds with both strategies
    num_seeds = 15
    for seed in range(num_seeds):
        if time.time() > deadline - 40:
            break
        # Alternate between standard greedy and ratio greedy
        if seed % 2 == 0:
            blocks = greedy_set_cover(seed, use_ratio=False)
        else:
            blocks = greedy_ratio(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
        
    # Local search polish
    best_blocks = refine_solution(best_blocks)
    
    return best_blocks

if __name__ == "__main__":
    pass
