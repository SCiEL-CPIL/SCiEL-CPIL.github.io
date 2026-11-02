import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # Precompute block coverage as numpy arrays of indices
    # This is memory heavy (75582 * 56 * 4 bytes ~ 17MB) but fast to access
    # We will use lists of lists for block_cover to allow easy set operations if needed, 
    # but numpy for counting.
    # Actually, for the solver, we need to check "does block b cover all t in target_set?"
    # and "what new t's does block b cover?".
    
    # Let's use a list of frozensets for block_cover to allow fast intersection?
    # Or just lists and use numpy for masks.
    
    # For the solver, we maintain a boolean mask of uncovered t's.
    # We need to quickly find blocks that cover a specific hard t.
    # t_to_blocks[ti] = list of block indices.
    
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # 2. Greedy Initial Solution
    def fast_greedy(rng):
        """
        Fast greedy to get a valid starting point.
        """
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []
        
        while num_uncovered > 0:
            # Pick random uncovered
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            ti_target = rng.choice(uncovered_indices.tolist())
            
            # Candidates
            candidates = t_to_blocks[ti_target]
            
            best_block = -1
            best_gain = -1
            
            # Evaluate candidates
            for bi in candidates:
                gain = 0
                # Optimization: precomputed sum? No, mask changes.
                # Use numpy sum on masked view? 
                # block_cover[bi] is a list. Convert to array? 
                # Let's just loop, it's small (56).
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            selected.append(best_block)
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                    
        return selected

    def prune(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return []
            
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    # 3. New Strategy: t-subset-centric Solver
    # State: 
    #   uncovered_mask: np.array of bools (True if uncovered)
    #   current_blocks: list of block indices
    #   
    # Move:
    #   1. Identify "hard" uncovered t-subsets (those with fewest available blocks? 
    #      Or just pick one).
    #   2. We want to REDUCE the number of blocks.
    #      This is hard to do directly with "add" moves.
    #      
    # Alternative: Local Search on Block Count.
    #   Start with greedy solution.
    #   Try to remove a block. If it creates uncovered t's, we MUST add a block 
    #   that covers them. 
    #   If the new block covers MORE uncovered t's than the removed one did 
    #   (i.e., net reduction in uncovered count is 0, but we removed 1 and added 1? No.)
    #   
    # Let's think about "Swap" moves more effectively.
    #   Current set S.
    #   Pick a block b in S.
    #   Let U_b be the set of t-subsets covered ONLY by b in S.
    #   If we remove b, U_b becomes uncovered.
    #   We need to add a block c (not in S) such that c covers all t in U_b.
    #   If such a c exists, we can swap b -> c.
    #   Size remains same.
    #   BUT, if c also covers other t's that were previously covered only by other blocks,
    #   those other blocks might become redundant and can be removed.
    #   
    # So the move is:
    #   1. Pick b in S.
    #   2. Compute U_b (unique coverage of b).
    #   3. Find candidates c such that U_b subset of cover(c).
    #   4. For each candidate c, simulate swap S' = S - {b} + {c}.
    #   5. Check if any blocks in S' are redundant.
    #   6. If redundant blocks exist, remove them.
    #   7. If len(S') < len(S), accept.
    
    def local_search_reduce(blocks, rng, time_limit):
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        # Precompute counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        # Helper to get unique coverage of a block
        def get_unique(bi):
            return [ti for ti in block_cover[bi] if counts[ti] == 1]
            
        # Helper to check if a block is redundant
        def is_redundant(bi):
            for ti in block_cover[bi]:
                if counts[ti] == 1:
                    return False
            return True
            
        # Helper to apply swap and prune
        def try_swap(bi, c):
            # Remove bi
            for ti in block_cover[bi]:
                counts[ti] -= 1
            # Add c
            for ti in block_cover[c]:
                counts[ti] += 1
                
            # Check for redundant blocks in current_blocks
            # Note: current_blocks still contains bi at its old position? 
            # We are simulating. 
            # Let's work on a copy of the list? No, too slow.
            # We know bi is at some index. 
            # Let's just check all blocks in current_blocks.
            
            new_blocks = []
            removed_count = 0
            for b in current_blocks:
                if b == bi:
                    # We are replacing bi with c. 
                    # If bi is redundant in the NEW state (which it is, because we're removing it), 
                    # we don't add it back.
                    # Wait, the logic is: 
                    # State S' = (S \ {bi}) U {c}.
                    # We need to check if any b in S \ {bi} is redundant in S'.
                    # And if c is redundant in S' (unlikely if it covers U_b).
                    
                    # Let's re-evaluate redundancy for all b in S \ {bi} and c.
                    pass
            
            # This simulation is getting complex with shared state.
            # Let's use a cleaner approach:
            # 1. Identify candidates c for a given bi.
            # 2. For a candidate c, calculate the "gain" in terms of potential removals.
            
            pass # Placeholder, will implement below
            
        while time.time() < time_limit:
            # Pick a random block to try to replace
            idx = rng.randrange(len(current_blocks))
            bi = current_blocks[idx]
            
            unique_ts = get_unique(bi)
            
            if not unique_ts:
                # bi is redundant!
                current_blocks.pop(idx)
                for ti in block_cover[bi]:
                    counts[ti] -= 1
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
                continue
            
            # Find candidates c that cover all unique_ts
            # Start with candidates for the first unique_t
            candidates = set(t_to_blocks[unique_ts[0]])
            for ti in unique_ts[1:]:
                candidates &= set(t_to_blocks[ti])
                
            candidates.discard(bi)
            
            if not candidates:
                continue
            
            # Score candidates
            # We want a candidate c that maximizes the number of blocks that become redundant.
            # A block b in S (b != bi) becomes redundant if all its unique t's are covered by c 
            # (and thus counts[t] increases, removing uniqueness from b).
            
            # Precompute unique t's for all blocks? 
            # unique_map: dict block_idx -> list of unique t's
            # This is expensive to update.
            
            # Approximation: 
            # Score(c) = sum(1 for b in S if b != bi and all(t in cover(c) for t in unique_t's of b))
            
            # Let's just try a few random candidates or the best one based on a simple heuristic.
            # Heuristic: c should cover many t's with count == 1 that are NOT in bi.
            
            best_cand = -1
            best_score = -1
            
            # Limit candidates to speed up
            cand_list = list(candidates)
            if len(cand_list) > 50:
                cand_list = rng.sample(cand_list, 50)
                
            for c in cand_list:
                score = 0
                # Count how many unique t's of OTHER blocks are covered by c
                # This is a proxy for "how many blocks might become redundant"
                for ti in block_cover[c]:
                    if counts[ti] == 1:
                        # Check if this ti is unique to bi
                        # If ti is in unique_ts of bi, it's not a "new" gain for other blocks
                        # If ti is NOT in unique_ts of bi, it means some other block b has ti as unique.
                        # If c covers ti, b loses its uniqueness on ti.
                        # If b loses ALL its unique t's, b becomes redundant.
                        
                        # To know if b becomes redundant, we need to know if c covers ALL unique t's of b.
                        # This is expensive.
                        
                        # Simpler score: just count unique t's covered by c that are not in bi.
                        # This encourages consolidation.
                        if ti not in unique_ts: # unique_ts is a list, O(n) check. 
                            score += 1
                            
                if score > best_score:
                    best_score = score
                    best_cand = c
                    
            if best_cand == -1:
                continue
                
            # Perform Swap
            # Remove bi
            for ti in block_cover[bi]:
                counts[ti] -= 1
            # Add c
            for ti in block_cover[best_cand]:
                counts[ti] += 1
            current_blocks[idx] = best_cand
            
            # Prune
            new_blocks = []
            drop_occurred = False
            for b in current_blocks:
                if is_redundant(b):
                    for ti in block_cover[b]:
                        counts[ti] -= 1
                    drop_occurred = True
                else:
                    new_blocks.append(b)
                    
            if drop_occurred:
                current_blocks = new_blocks
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                current_blocks = new_blocks
                
        return best_blocks

    # Main Execution
    best_solution = None
    best_len = float('inf')
    
    # Phase 1: Multi-start Greedy
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 20.0:
            break
        rng = random.Random(123 + i)
        blocks = fast_greedy(rng)
        blocks = prune(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
    # Phase 2: Local Search Reduce
    if best_solution:
        rng = random.Random(999)
        optimized = local_search_reduce(best_solution, rng, deadline - 5.0)
        if len(optimized) < best_len:
            best_len = len(optimized)
            best_solution = optimized

    if not best_solution:
        rng = random.Random(1)
        best_solution = fast_greedy(rng)
        best_solution = prune(best_solution)

    return [list(all_blocks[bi]) for bi in best_solution]
