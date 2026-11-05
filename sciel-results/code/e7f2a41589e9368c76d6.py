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
    # n_triplets is ~2024, which fits easily in a Python integer
    block_masks = [0] * n_blocks
    block_triple_ids = [None] * n_blocks  # list of triplet IDs for each block
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        ids = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            ids.append(tid)
        block_masks[b_idx] = mask
        block_triple_ids[b_idx] = ids
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1

    def greedy_set_cover(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_blocks = []
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_gain = -1
            best_idx = -1
            candidates = []
            
            # Find block with max gain
            # We scan all blocks. 346k iterations * bit_count is fast.
            for b_idx in range(n_blocks):
                # If block is already used, gain is 0 (since uncovered bits are 0 there)
                # But we can just calculate gain directly.
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_gain == 0:
                # Should not happen if uncovered != 0, unless we ran out of time or logic error
                break
                
            # Randomly pick from candidates to diversify
            # If many candidates, picking one randomly helps avoid local minima
            chosen_idx = rng.choice(candidates)
            
            solution_blocks.append(all_blocks[chosen_idx])
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
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
                # Check if block is redundant
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
            # Precompute exclusive triplets for each block
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
                    
                    # Union of exclusive triplets
                    # We need to check if there is a block that covers all these triplets.
                    # The points involved in these triplets must be contained in the new block.
                    points_needed = set()
                    # We can optimize by just collecting points from the triplets
                    # But we need to ensure the new block covers the triplets.
                    # If the new block contains all points of the triplets, it covers them.
                    
                    # Let's collect points from exclusive triplets of i and j
                    # Note: exclusives are triplet IDs. We need to map back to points.
                    # This is slightly expensive. Let's precompute triplet points?
                    # all_triplets[tid] is the tuple of points.
                    
                    for tid in exclusives[i]:
                        points_needed.update(all_triplets[tid])
                    for tid in exclusives[j]:
                        points_needed.update(all_triplets[tid])
                        
                    if len(points_needed) > k:
                        continue
                        
                    # If points_needed <= k, we can form a block.
                    # We need to find a block that contains points_needed.
                    # Actually, ANY block containing points_needed will cover the triplets in points_needed?
                    # Yes, because the triplets are subsets of points_needed.
                    # So we just need to check if such a block exists in our universe.
                    # Since we are allowed to add ANY block, we can just construct one.
                    # But we must ensure it's a valid k-subset.
                    
                    # Construct candidate block
                    cand_set = set(points_needed)
                    # Fill remaining spots with arbitrary points
                    # To be deterministic and simple, pick smallest available
                    for p in range(v):
                        if len(cand_set) == k:
                            break
                        if p not in cand_set:
                            cand_set.add(p)
                            
                    if len(cand_set) != k:
                        continue
                        
                    new_block = tuple(sorted(cand_set))
                    
                    # Check if this new block is already in the solution?
                    # If it is, we just remove i and j.
                    # If it's not, we add it and remove i and j.
                    
                    # We need to verify that new_block covers all triplets in exclusives[i] and exclusives[j].
                    # By construction, it contains the points, so it covers the triplets.
                    
                    # Perform the merge
                    # Remove i and j, add new_block
                    # Be careful with indices if we modify list in place.
                    # Let's build a new list.
                    
                    new_blocks_list = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks_list.append(b)
                    new_blocks_list.append(list(new_block))
                    
                    # Verify validity? The evaluator does this. 
                    # But we should ensure we didn't break coverage.
                    # The only triplets that were uniquely covered by i or j are in exclusives[i] and exclusives[j].
                    # new_block covers them.
                    # Triplets covered by both i and j (but not exclusively) are still covered by the other block (or others).
                    # So coverage is maintained.
                    
                    blocks = new_blocks_list
                    merged = True
                    changed = True
                    break
                if merged:
                    break
            if merged:
                continue

        return blocks

    def simulated_annealing(initial_blocks, seed=0):
        rng = random.Random(seed)
        
        # State: set of selected block indices
        # We work with indices for speed
        selected_indices = set()
        for b in initial_blocks:
            # Find index of this block
            # This is slow if done naively. Let's use a dict for block->index lookup
            pass
        
        # Build lookup for block tuple to index
        # We already have all_blocks. 
        # To make SA fast, let's assume initial_blocks are from all_blocks or we map them.
        # In our pipeline, initial_blocks come from greedy which uses all_blocks indices.
        # But refine_solution returns lists. We need to map back to indices or work with lists.
        # Working with indices is much faster for counts updates.
        
        # Let's reconstruct indices from the refined blocks
        # This is a bottleneck if we don't have a map.
        # Let's create a map of tuple(sorted(block)) -> index
        # But blocks in refined might be "new" blocks not in all_blocks?
        # In refine_solution, we add new_block which is a tuple of sorted points.
        # It might not be in all_blocks? No, all_blocks contains ALL k-subsets.
        # So any valid block is in all_blocks.
        
        # However, constructing this map for 346k blocks is expensive memory-wise but doable?
        # Or we can just use the indices from the greedy phase and pass them through?
        # The current code returns blocks (lists of points).
        # Let's change the flow: 
        # 1. Greedy returns indices.
        # 2. Refine works on indices? Or points?
        # Refine currently works on points.
        # Let's keep Refine on points for simplicity of "merge" logic (finding points_needed).
        # But for SA, we want indices.
        
        # Alternative: Run SA directly on the greedy output (indices) BEFORE refine?
        # Or run SA AFTER refine, but map back to indices.
        
        # Let's map back to indices for SA.
        # Since v=24, k=7, the number of blocks is 346,104.
        # A dictionary with 346k keys is fine.
        
        block_to_idx = {}
        for i, b in enumerate(all_blocks):
            block_to_idx[b] = i
            
        selected_indices = set()
        for b in initial_blocks:
            idx = block_to_idx.get(tuple(b), None)
            if idx is not None:
                selected_indices.add(idx)
            else:
                # Should not happen if refine is correct
                pass

        # Initialize counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        for idx in selected_indices:
            for tid in block_triple_ids[idx]:
                counts[tid] += 1
                
        # Energy: number of blocks + penalty * uncovered
        # We want to minimize blocks.
        # If we maintain full coverage, energy = len(selected_indices).
        # If we break coverage, energy goes up.
        
        def get_uncovered_count():
            return np.sum(counts == 0)
            
        current_energy = len(selected_indices) + 10 * get_uncovered_count()
        best_energy = current_energy
        best_indices = set(selected_indices)
        
        # SA Parameters
        T = 50.0
        cooling_rate = 0.999
        T_min = 0.5
        
        # Precompute block triplets for fast count updates (already have block_triple_ids)
        
        # Main SA Loop
        while T > T_min and time.time() < deadline - 5:
            
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.3: # Remove
                if not selected_indices:
                    T *= cooling_rate
                    continue
                idx = rng.choice(list(selected_indices))
                
                # Check if removal is valid (doesn't uncover any triple)
                can_remove = True
                for tid in block_triple_ids[idx]:
                    if counts[tid] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Apply move
                    selected_indices.remove(idx)
                    for tid in block_triple_ids[idx]:
                        counts[tid] -= 1
                    
                    new_uncovered = get_uncovered_count()
                    new_energy = len(selected_indices) + 10 * new_uncovered
                    delta_e = new_energy - current_energy
                    
                    if delta_e < 0 or rng.random() < pow(2.71828, -delta_e / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_indices = set(selected_indices)
                    else:
                        # Revert
                        selected_indices.add(idx)
                        for tid in block_triple_ids[idx]:
                            counts[tid] += 1
            
            elif move_type < 0.6: # Add
                # Pick a random block not in selected
                # To speed up, pick from a subset or just random index and check
                # If num_blocks is large, random selection is O(1) but might hit selected often
                # Let's pick a random index
                idx = rng.randint(0, n_blocks - 1)
                if idx in selected_indices:
                    T *= cooling_rate
                    continue
                
                # Apply move
                selected_indices.add(idx)
                for tid in block_triple_ids[idx]:
                    counts[tid] += 1
                    
                new_uncovered = get_uncovered_count()
                new_energy = len(selected_indices) + 10 * new_uncovered
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < pow(2.71828, -delta_e / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_indices = set(selected_indices)
                else:
                    # Revert
                    selected_indices.remove(idx)
                    for tid in block_triple_ids[idx]:
                        counts[tid] -= 1

            else: # Swap
                if not selected_indices:
                    T *= cooling_rate
                    continue
                idx_out = rng.choice(list(selected_indices))
                idx_in = rng.randint(0, n_blocks - 1)
                if idx_in in selected_indices or idx_in == idx_out:
                    T *= cooling_rate
                    continue
                
                # Apply move
                # Remove idx_out
                for tid in block_triple_ids[idx_out]:
                    counts[tid] -= 1
                # Add idx_in
                for tid in block_triple_ids[idx_in]:
                    counts[tid] += 1
                    
                new_uncovered = get_uncovered_count()
                new_energy = len(selected_indices) + 10 * new_uncovered # Size same
                delta_e = new_energy - current_energy
                
                if delta_e < 0 or rng.random() < pow(2.71828, -delta_e / T):
                    current_energy = new_energy
                    # Commit swap
                    selected_indices.remove(idx_out)
                    selected_indices.add(idx_in)
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_indices = set(selected_indices)
                else:
                    # Revert
                    selected_indices.remove(idx_in)
                    selected_indices.add(idx_out)
                    for tid in block_triple_ids[idx_out]:
                        counts[tid] += 1
                    for tid in block_triple_ids[idx_in]:
                        counts[tid] -= 1
            
            T *= cooling_rate
            
        # Convert best_indices back to blocks
        best_blocks = [all_blocks[i] for i in best_indices]
        return best_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    # Bitmask greedy is fast. We can afford more seeds.
    num_seeds = 15
    for seed in range(num_seeds):
        if time.time() > deadline - 40:
            break
        blocks = greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
        
    # Local search polish (Refine)
    best_blocks = refine_solution(best_blocks)
    
    # Simulated Annealing Polish
    # Try a few seeds for SA as well
    best_sa_blocks = None
    best_sa_count = float('inf')
    
    # We can run SA on the refined best, and maybe on a few other good greedy results if time permits?
    # For now, just run on the best refined.
    
    if time.time() < deadline - 20:
        sa_blocks = simulated_annealing(best_blocks, seed=42)
        if len(sa_blocks) < best_sa_count:
            best_sa_count = len(sa_blocks)
            best_sa_blocks = sa_blocks
            
    # Final Refine on SA output?
    if best_sa_blocks and time.time() < deadline - 10:
        final_blocks = refine_solution(best_sa_blocks)
        if len(final_blocks) < best_sa_count:
            best_sa_blocks = final_blocks
            best_sa_count = len(final_blocks)
            
    # Choose the best between refined greedy and SA-refined
    if best_sa_blocks and len(best_sa_blocks) < len(best_blocks):
        best_blocks = best_sa_blocks
        
    return best_blocks

if __name__ == "__main__":
    pass
