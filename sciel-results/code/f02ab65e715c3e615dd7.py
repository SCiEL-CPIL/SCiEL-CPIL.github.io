"""Covering design C(24,7,3) - deletion-based local search with simulated annealing."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    def get_block_covered_indices(block):
        """Get the set of t-subset indices covered by a block."""
        return frozenset(t_subset_to_idx[ts] for ts in itertools.combinations(block, t))
    
    def greedy_cover(seed=0):
        """Fast greedy covering to get an initial valid solution."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random uncovered t-subset to start a block
            start_idx = rng.choice(list(uncovered))
            block = list(all_t_subsets[start_idx])
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if t_subset_to_idx.get(ts) in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                uncovered.discard(t_subset_to_idx[ts])
        
        return blocks
    
    def local_search(blocks, seed=0):
        """
        Deletion-based local search with simulated annealing.
        Moves:
        1. Remove a redundant block (if possible).
        2. Swap a block with a new block that covers more uncovered sets (if it reduces total count or improves structure).
        Actually, for minimization, we primarily want to remove blocks.
        If we can't remove, we might try to swap a block for another that allows subsequent removals?
        A simpler and effective move is:
        - Try to remove a block.
        - If stuck, try to replace a block with a different one that covers a superset or different sets, hoping to enable removal later.
        
        Let's stick to:
        1. Greedy removal of redundant blocks.
        2. Randomized removal attempts: pick a random block, try to remove it. If it fails, maybe try swapping it?
        
        Better strategy:
        - Maintain a list of blocks and their coverage counts.
        - Iteration:
          a. Try to remove a random block. If valid, accept (count decreases).
          b. If invalid, try to swap the block with a new random block (or a block constructed to cover the unique sets it had).
             This is complex.
             
        Let's use a standard "remove if possible" loop with randomization, and if stuck, perturb by replacing a block.
        """
        rng = random.Random(seed)
        blocks = list(blocks)
        
        # Precompute coverage
        block_covers = [get_block_covered_indices(b) for b in blocks]
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bc in block_covers:
            for idx in bc:
                cover_count[idx] += 1
        
        current_len = len(blocks)
        
        # Phase 1: Greedy removal of redundant blocks
        changed = True
        while changed:
            changed = False
            # Try to remove any redundant block
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                # Check if block i is redundant
                redundant = True
                for idx in block_covers[i]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    for idx in block_covers[i]:
                        cover_count[idx] -= 1
                    del blocks[i]
                    del block_covers[i]
                    current_len -= 1
                    changed = True
                    break # Restart loop as indices changed
        
        # Phase 2: Simulated Annealing / Randomized Search
        # We try to swap blocks to find a configuration where more blocks can be removed.
        # Move: Pick a random block B_i. Generate a new block B_new.
        # Replace B_i with B_new.
        # Then try to remove redundant blocks.
        # If the total count decreases, accept.
        # If it increases, accept with probability exp(-delta/T).
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        max_iterations = 5000
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            if T < min_T:
                break
                
            if not blocks:
                break
                
            # Pick a random block to replace
            i = rng.randint(0, len(blocks) - 1)
            
            # Generate a new block
            # Strategy: Pick a random t-subset from the ones covered by the old block, or random from all?
            # Better: Pick a random t-subset from the entire space, or from the uncovered/under-covered sets?
            # Since we want to maintain coverage, let's pick a random t-subset.
            # To be more effective, pick a t-subset that is currently covered by only one block (the one we are replacing)?
            # No, that would break coverage immediately.
            
            # Let's try: Pick a random 7-subset.
            new_block = sorted(rng.sample(range(v), k))
            new_block = tuple(new_block)
            
            if new_block == tuple(blocks[i]):
                continue
                
            new_cover = get_block_covered_indices(new_block)
            
            # Check validity of replacement:
            # For all t-subsets in old_cover, if cover_count[idx] == 1, then new_cover must contain idx.
            # This is a strict constraint. If violated, the move is invalid (coverage lost).
            # However, in SA, we might allow temporary invalidity?
            # No, the evaluator checks validity at the end. Intermediate states can be invalid if we track it,
            # but it's safer to only make valid moves or track the penalty.
            # Let's stick to valid moves only for simplicity and robustness, but use random restarts if stuck.
            # Actually, "valid moves only" local search gets stuck easily.
            
            # Let's implement a penalty-based SA where we allow invalid states but penalize them.
            # Objective: Minimize (num_blocks * 1000 + num_uncovered_t_subsets)
            
            # Current state:
            # We need to track uncovered sets if we allow invalid moves.
            # Let's refactor to track `cover_count` and `uncovered_count`.
            
            # For now, let's try a simpler "valid move" heuristic:
            # Only replace B_i with B_new if B_new covers all the unique t-subsets of B_i.
            # i.e., for all idx in old_cover, if cover_count[idx] == 1, then idx in new_cover.
            
            valid_swap = True
            old_cover = block_covers[i]
            for idx in old_cover:
                if cover_count[idx] == 1 and idx not in new_cover:
                    valid_swap = False
                    break
            
            if not valid_swap:
                continue
                
            # If valid swap, perform it
            # Update cover counts
            for idx in old_cover:
                cover_count[idx] -= 1
            for idx in new_cover:
                cover_count[idx] += 1
            
            blocks[i] = list(new_block)
            block_covers[i] = new_cover
            
            # Now try to remove redundant blocks
            removed_any = False
            for j in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    break
                redundant = True
                for idx in block_covers[j]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_covers[j]:
                        cover_count[idx] -= 1
                    del blocks[j]
                    del block_covers[j]
                    current_len -= 1
                    removed_any = True
                    break # Restart removal loop
            
            # If no block was removed, the swap didn't help immediately.
            # We could revert the swap to keep the search tight, or keep it to explore.
            # Let's keep it, as it might enable future removals.
            
            T *= cooling
            
        return blocks

    # Initial Solution
    best_blocks = greedy_cover(0)
    best_len = len(best_blocks)
    
    # Local Search
    best_blocks = local_search(best_blocks, seed=1)
    best_len = len(best_blocks)
    
    # Multiple restarts
    for seed in range(2, 100):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a new initial solution
        init_blocks = greedy_cover(seed)
        
        # Apply local search
        ls_blocks = local_search(init_blocks, seed=seed)
        
        if len(ls_blocks) < best_len:
            best_len = len(ls_blocks)
            best_blocks = ls_blocks
            
    # Final validation
    if best_blocks is None:
        return []
        
    covered = set()
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[ts])
            
    if covered != set(range(n_t)):
        # Fallback
        return greedy_cover(0)
        
    return best_blocks
