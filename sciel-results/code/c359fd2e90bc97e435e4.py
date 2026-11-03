import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 135.0
    start_time = time.time()

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each point, the list of t-subsets it is part of? 
    # No, we need block coverage.
    # We will use a coverage array of size n_t.

    def get_block_indices(block):
        """Returns list of indices of t-subsets covered by this block."""
        idxs = []
        for ts in itertools.combinations(block, t):
            idxs.append(t_subset_to_idx[ts])
        return idxs

    def is_valid_covering(blocks):
        covered = np.zeros(n_t, dtype=np.bool_)
        for b in blocks:
            for idx in get_block_indices(b):
                covered[idx] = True
        return covered.all()

    def count_uncovered(blocks):
        cc = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for idx in get_block_indices(b):
                cc[idx] += 1
        return np.sum(cc == 0)

    def greedy_construct(uncovered_set, seed):
        """
        Greedily add blocks to cover 'uncovered_set'.
        uncovered_set is a set of indices of t-subsets that need covering.
        Returns a list of blocks added.
        """
        rng = random.Random(seed)
        blocks_added = []
        uncovered = set(uncovered_set)
        
        if not uncovered:
            return blocks_added

        # To speed up, we can precompute for each t-subset, which blocks cover it?
        # No, that's huge. 
        # Instead, for each uncovered t-subset, we can look for a block covering it.
        # Better: Standard greedy set cover.
        # Score of a block B = number of uncovered t-subsets it covers.
        
        # Optimization: We don't need to check all 346k blocks every step.
        # We can sample or use a heuristic.
        # For robustness within time limits, we will use a "randomized greedy" 
        # that picks a random uncovered triple, and then picks a random block containing it,
        # or the best block containing it among a sample.
        
        while uncovered:
            # Pick a random uncovered triple to focus on
            target_idx = rng.choice(list(uncovered))
            target_triple = all_t_subsets[target_idx]
            
            # Find blocks that cover this target triple
            # A block must contain all 3 points of target_triple.
            # The block has k=7 points. 3 are fixed. We need to choose 4 more from v-3 points.
            # Number of such blocks = C(v-3, k-3) = C(21, 4) = 5985.
            # This is small enough to check all of them for this target.
            
            fixed_points = set(target_triple)
            remaining_points = [p for p in range(v) if p not in fixed_points]
            
            best_block = None
            best_score = -1
            
            # Sample a subset of potential blocks if C(21,4) is too slow? 
            # 5985 blocks * 35 triples each = ~200k ops per step.
            # If we have ~100 steps, that's 20M ops. Feasible in Python if optimized? 
            # Maybe a bit slow. Let's sample 100 blocks from the candidates.
            
            candidate_blocks = []
            # Generate combinations of 4 points from remaining_points
            # C(21, 4) = 5985.
            for combo in itertools.combinations(remaining_points, k - t):
                block = sorted(fixed_points | set(combo))
                candidate_blocks.append(block)
                
            # If too many, sample
            if len(candidate_blocks) > 200:
                candidate_blocks = rng.sample(candidate_blocks, 200)
            
            for block in candidate_blocks:
                score = 0
                for idx in get_block_indices(block):
                    if idx in uncovered:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_block = block
            
            if best_block is None:
                # Fallback: just pick a random block containing the triple
                combo = rng.sample(remaining_points, k - t)
                best_block = sorted(fixed_points | set(combo))
            
            blocks_added.append(best_block)
            
            # Update uncovered
            for idx in get_block_indices(best_block):
                uncovered.discard(idx)
                
        return blocks_added

    def local_search(blocks, seed, time_budget):
        """
        Local search to minimize number of blocks.
        Moves: 
        1. Try to remove a block if it doesn't break coverage.
        2. Try to swap a point in a block to improve coverage efficiency (or enable removal of another block).
        """
        rng = random.Random(seed)
        blocks = list(blocks)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            for idx in get_block_indices(b):
                cover_count[idx] += 1
        
        current_size = len(blocks)
        best_blocks = list(blocks)
        best_size = current_size
        
        # Simple local search: try to remove redundant blocks
        # Then try to merge/swap to reduce count? 
        # Swapping doesn't reduce count. Only removal does.
        # To reduce count, we need to find a block that is "redundant" given the others.
        # Or, we need to modify blocks so that one becomes redundant.
        
        # Strategy: 
        # 1. Check for directly redundant blocks.
        # 2. If none, perform a "perturbation" (swap points) and check again.
        
        T = 1.0
        cooling = 0.95
        
        while time.time() < time_budget and T > 0.01:
            # Check for redundant blocks
            redundant_found = False
            for i in range(len(blocks)):
                # Temporarily remove block i
                for idx in get_block_indices(blocks[i]):
                    cover_count[idx] -= 1
                
                if np.all(cover_count > 0):
                    # Block i is redundant
                    blocks.pop(i)
                    # Re-add to cover_count? No, we removed it permanently.
                    # We need to update cover_count to reflect the removal.
                    # It is already updated (decremented).
                    # So we just continue.
                    redundant_found = True
                    break
                else:
                    # Revert
                    for idx in get_block_indices(blocks[i]):
                        cover_count[idx] += 1
            
            if redundant_found:
                best_blocks = list(blocks)
                best_size = len(blocks)
                T = 1.0 # Reset temperature on improvement
                continue
            
            # No redundant blocks found. Perform a move.
            # Move: Swap a point in block A with a point in block B (or outside).
            # We want to create a situation where a block becomes redundant.
            
            if len(blocks) < 2:
                break
                
            i = rng.randint(0, len(blocks) - 1)
            block_i = blocks[i]
            
            # Pick a point to remove from block_i
            x_idx = rng.randint(0, k - 1)
            x = block_i[x_idx]
            
            # Pick a point to add. 
            # Option 1: From another block.
            # Option 2: Random point.
            if rng.random() < 0.5 and len(blocks) > 1:
                j = rng.randint(0, len(blocks) - 1)
                if i == j:
                    j = (j + 1) % len(blocks)
                block_j = blocks[j]
                y_idx = rng.randint(0, k - 1)
                y = block_j[y_idx]
                if x == y:
                    continue
                if x in block_j or y in block_i:
                    continue
            else:
                y = rng.choice([p for p in range(v) if p not in block_i])
                # No second block affected for cover count update in this branch? 
                # Wait, if we pick from outside, only block_i changes.
                # If we pick from block_j, both change.
                # Let's stick to "Swap within a single block" (replace point) for simplicity and speed.
                # This changes the coverage of block_i only.
                pass
            
            # Let's simplify: Replace point x in block_i with y.
            # Calculate change in cover_count.
            rest_i = [p for p in block_i if p != x]
            
            lost = []
            gained = []
            for a, b in itertools.combinations(rest_i, t - 1):
                ts_l = tuple(sorted((x, a, b)))
                idx_l = t_subset_to_idx.get(ts_l)
                if idx_l is not None:
                    lost.append(idx_l)
                ts_g = tuple(sorted((y, a, b)))
                idx_g = t_subset_to_idx.get(ts_g)
                if idx_g is not None:
                    gained.append(idx_g)
            
            # Apply
            for idx in lost:
                cover_count[idx] -= 1
            for idx in gained:
                cover_count[idx] += 1
            
            new_block_i = list(block_i)
            new_block_i[x_idx] = y
            new_block_i.sort()
            blocks[i] = new_block_i
            
            # Acceptance: Always accept if it doesn't break coverage?
            # If it breaks coverage (cover_count drops to 0 somewhere), we might still accept if T is high?
            # No, we must maintain validity.
            # So, only accept if np.all(cover_count > 0).
            
            if np.all(cover_count > 0):
                # Valid move. Keep it.
                # Check for redundancy again immediately?
                # The loop will do it next iteration.
                pass
            else:
                # Invalid. Revert.
                for idx in gained:
                    cover_count[idx] -= 1
                for idx in lost:
                    cover_count[idx] += 1
                blocks[i] = block_i
            
            T *= cooling
            
        return best_blocks

    # Main Loop: Iterated Greedy
    best_blocks = None
    best_count = float('inf')
    
    # Initial Solution
    init_seed = 0
    initial_uncovered = set(range(n_t))
    init_blocks = greedy_construct(initial_uncovered, init_seed)
    
    # Optimize initial
    opt_blocks = local_search(init_blocks, init_seed, time_budget=10.0)
    if is_valid_covering(opt_blocks):
        best_blocks = opt_blocks
        best_count = len(opt_blocks)
    
    # Iterated Greedy Loop
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        seed = iteration + 100
        
        # Perturbation: Take current best, remove some random blocks
        if best_blocks is None:
            break
            
        perturbed_blocks = list(best_blocks)
        # Remove 5-10 random blocks
        num_remove = min(len(perturbed_blocks), rng.randint(5, 10)) if 'rng' in locals() else 5
        # We need a rng instance
        rng_iter = random.Random(seed)
        num_remove = min(len(perturbed_blocks), rng_iter.randint(5, 10))
        
        indices_to_remove = rng_iter.sample(range(len(perturbed_blocks)), num_remove)
        indices_to_remove.sort(reverse=True)
        
        # Determine which triples become uncovered
        # We need to calculate coverage of remaining blocks
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in perturbed_blocks:
            for idx in get_block_indices(b):
                cover_count[idx] += 1
        
        for idx in indices_to_remove:
            b = perturbed_blocks[idx]
            for t_idx in get_block_indices(b):
                cover_count[t_idx] -= 1
            perturbed_blocks.pop(idx)
            
        uncovered_set = set(np.where(cover_count == 0)[0])
        
        # Repair
        added_blocks = greedy_construct(uncovered_set, seed)
        repaired_blocks = perturbed_blocks + added_blocks
        
        # Optimize
        opt_repaired = local_search(repaired_blocks, seed, time_budget=5.0)
        
        if is_valid_covering(opt_repaired):
            if len(opt_repaired) < best_count:
                best_count = len(opt_repaired)
                best_blocks = opt_repaired
                # Optionally, restart with a higher time budget for local search if we improve
                # But we have a global deadline.

    # Final Polish
    if best_blocks is not None:
        final_opt = local_search(best_blocks, 999, time_budget=deadline - time.time() - 5.0)
        if is_valid_covering(final_opt):
            best_blocks = final_opt

    if best_blocks is None:
        # Fallback
        best_blocks = greedy_construct(set(range(n_t)), 0)
        
    return best_blocks
