"""
Covering design C(19,8,5) using Lagrangian-guided greedy with NumPy acceleration.
Combined with beam search local improvement and redundancy removal.
"""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0

    # Precompute all t-subsets
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build block_cover_lists as NumPy arrays of t-set indices
    # This avoids the memory overhead of the dense matrix and allows fast sparse sum
    block_cover_lists = []
    for block in all_blocks:
        indices = [t_set_index[ts] for ts in itertools.combinations(block, t)]
        block_cover_lists.append(np.array(indices, dtype=np.int32))

    # Build t_set_to_blocks
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx, cover_list in enumerate(block_cover_lists):
        for ts_idx in cover_list:
            t_set_to_blocks[ts_idx].append(b_idx)

    best_solution = None
    best_size = float('inf')

    # Multiple strategies with different parameters
    # Mix of Lagrangian-guided and standard greedy-like behaviors
    strategies = [
        # (beam_width, seed, lambda_init, lambda_step)
        (1, 42, 1.0, 0.1),
        (5, 101, 1.0, 0.1),
        (10, 102, 2.0, 0.2),
        (20, 103, 1.5, 0.15),
        (50, 104, 1.0, 0.05),
        (10, 201, 3.0, 0.3),
        (20, 202, 2.0, 0.2),
        (50, 203, 1.5, 0.1),
        (1, 301, 5.0, 0.5),
        (5, 302, 4.0, 0.4),
        (10, 303, 3.0, 0.3),
        (20, 304, 2.5, 0.25),
        (5, 401, 1.0, 0.05),
        (10, 402, 1.5, 0.1),
        (20, 403, 2.0, 0.15),
        (50, 404, 1.0, 0.08),
    ]

    for beam_width, seed, lambda_init, lambda_step in strategies:
        if time.time() > deadline - 15.0:
            break

        sol = _lagrangian_greedy_cover(
            v, k, t, t_set_index, all_blocks, block_cover_lists,
            num_t_sets, t_set_to_blocks, deadline,
            rng_seed=seed, beam_width=beam_width,
            lambda_init=lambda_init, lambda_step=lambda_step
        )
        if sol:
            # Quick redundancy removal before comparison to save time? 
            # Actually, let's just compare raw size first, then polish the best.
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol

    if best_solution is None:
        return []

    # Post-processing: remove redundant blocks and local improvement
    best_solution = _remove_redundant_blocks(best_solution, block_cover_lists, all_blocks, deadline)
    best_solution = _local_improve(best_solution, block_cover_lists, all_blocks, t_set_index, v, k, t, deadline)

    # Ensure blocks are lists
    return [list(b) for b in best_solution]


def _lagrangian_greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_lists,
                              num_t_sets, t_set_to_blocks, deadline,
                              rng_seed=42, beam_width=1, lambda_init=1.0, lambda_step=0.1):
    """
    Lagrangian-relaxation-guided greedy set cover with NumPy-accelerated scoring.
    """
    rng = random.Random(rng_seed)
    uncovered = set(range(num_t_sets))
    selected_blocks = []

    # Penalty weights
    w = np.full(num_t_sets, lambda_init, dtype=np.float64)

    while uncovered:
        if time.time() > deadline - 3.0:
            return None

        # Pick target t-sets to guide search
        if len(uncovered) <= beam_width:
            targets = list(uncovered)
        else:
            targets = rng.sample(list(uncovered), beam_width)

        # Find candidate blocks
        candidate_blocks = set()
        for target_tset_idx in targets:
            best_block_idx = -1
            max_score = -1.0

            for b_idx in t_set_to_blocks[target_tset_idx]:
                # Sparse score computation: sum of weights of covered t-sets
                indices = block_cover_lists[b_idx]
                score = w[indices].sum()
                if score > max_score:
                    max_score = score
                    best_block_idx = b_idx

            if best_block_idx != -1:
                candidate_blocks.add(best_block_idx)

        # Pick best block among candidates
        final_best_block_idx = -1
        final_max_score = -1.0

        for b_idx in candidate_blocks:
            indices = block_cover_lists[b_idx]
            score = w[indices].sum()
            if score > final_max_score:
                final_max_score = score
                final_best_block_idx = b_idx

        if final_best_block_idx == -1:
            break

        selected_blocks.append(all_blocks[final_best_block_idx])

        # Update uncovered
        covered_indices = block_cover_lists[final_best_block_idx]
        for idx in covered_indices:
            uncovered.discard(idx)

        # Update Lagrangian multipliers
        if uncovered:
            uncovered_list = list(uncovered)
            w[uncovered_list] *= (1.0 + lambda_step)
        
        covered_list = covered_indices.tolist()
        w[covered_list] *= (1.0 - lambda_step * 0.5)
        w = np.maximum(w, 0.1)

    return selected_blocks


def _remove_redundant_blocks(blocks, block_cover_lists, all_blocks, deadline):
    """Remove blocks that don't cover any t-set uniquely."""
    if not blocks:
        return blocks

    all_blocks_set = {frozenset(b): i for i, b in enumerate(all_blocks)}
    current_block_indices = []
    for block in blocks:
        block_set = frozenset(block)
        if block_set in all_blocks_set:
            current_block_indices.append(all_blocks_set[block_set])

    if len(current_block_indices) != len(blocks):
        return blocks

    # Determine num_t_sets from the maximum index in block_cover_lists
    max_idx = 0
    for bl in block_cover_lists:
        if len(bl) > 0:
            max_idx = max(max_idx, np.max(bl))
    num_t_sets = max_idx + 1
    
    if num_t_sets == 0:
        return blocks

    # Count coverage
    coverage = np.zeros(num_t_sets, dtype=np.int32)
    for b_idx in current_block_indices:
        for idx in block_cover_lists[b_idx]:
            coverage[idx] += 1

    # Iteratively remove redundant blocks
    improved = True
    while improved:
        if time.time() > deadline - 10.0:
            break
        improved = False
        
        # Iterate in reverse to handle indices safely if popping
        # We check if any t-set has coverage 1 and is covered by this block.
        # If so, the block is critical and cannot be removed.
        for i in range(len(current_block_indices) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            
            b_idx = current_block_indices[i]
            can_remove = True
            for idx in block_cover_lists[b_idx]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for idx in block_cover_lists[b_idx]:
                    coverage[idx] -= 1
                current_block_indices.pop(i)
                improved = True
                break # Restart loop to re-evaluate others

    if not current_block_indices:
        return blocks

    return [all_blocks[b_idx] for b_idx in current_block_indices]

def _local_improve(blocks, block_cover_lists, all_blocks, t_set_index, v, k, t, deadline):
    """
    Try to replace pairs of blocks with a single block that covers all unique t-sets.
    This is a 1-exchange heuristic.
    """
    if not blocks:
        return blocks
    
    # Map block tuple to index in all_blocks for fast lookup
    block_to_idx = {frozenset(b): i for i, b in enumerate(all_blocks)}
    
    # Current block indices
    current_indices = [block_to_idx[frozenset(b)] for b in blocks]
    
    # Precompute coverage counts for all t-sets
    num_t_sets = len(t_set_index)
    coverage = np.zeros(num_t_sets, dtype=np.int32)
    for b_idx in current_indices:
        for idx in block_cover_lists[b_idx]:
            coverage[idx] += 1
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        n = len(current_indices)
        
        # Try removing one block
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            
            b_idx = current_indices[i]
            # Check if any t-set becomes uncovered
            can_remove = True
            for idx in block_cover_lists[b_idx]:
                if coverage[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                for idx in block_cover_lists[b_idx]:
                    coverage[idx] -= 1
                current_indices.pop(i)
                improved = True
                break # Restart loop
        
        if improved:
            continue
            
        # Try 1-exchange: remove block i, add block j
        # We iterate through all pairs. This is O(N^2 * C) which might be slow if N is large.
        # N is ~300. 300^2 = 90,000. C is ~120. 10^7 operations. Feasible in Python with NumPy?
        # Maybe too slow in pure Python loops. Let's be careful.
        
        # Optimization: Only try exchanges if we have enough time.
        if time.time() > deadline - 10.0:
            break
            
        # For each block i to remove
        for i in range(len(current_indices)):
            if time.time() > deadline - 5.0:
                break
            
            b_idx_i = current_indices[i]
            
            # Temporarily remove i
            for idx in block_cover_lists[b_idx_i]:
                coverage[idx] -= 1
                
            # Find uncovered t-sets (coverage == 0)
            # We need to find a block j that covers ALL uncovered t-sets caused by removing i
            # Actually, removing i might uncover multiple t-sets.
            # Let's identify the set of t-sets that are now uncovered.
            # These are the t-sets in block_cover_lists[b_idx_i] that had coverage 1.
            
            # Let's just check if coverage has any 0s.
            # If it does, we need to find a block that covers all of them.
            
            # To speed this up, we can just check if any t-set has coverage 0.
            # If yes, we need to find a candidate block.
            
            # Efficient check:
            # If min(coverage) > 0, then we can just remove i (which is handled in the previous block logic if we re-ran it, but we broke out of that).
            # Wait, the previous block logic only removes if it's safe to remove *without* adding.
            # Here we are doing exchange.
            
            # Let's check if coverage has any 0s.
            if np.any(coverage == 0):
                # We need to find a block that covers all indices where coverage is 0.
                # Let's get the list of uncovered indices.
                uncovered_indices = np.where(coverage == 0)[0]
                
                # Find candidate blocks that cover all these indices.
                # A block j is a candidate if set(uncovered_indices) is a subset of set(block_cover_lists[j]).
                
                # To do this efficiently, we can check the intersection size.
                # len(uncovered_indices) must be equal to len(set(uncovered_indices) & set(block_cover_lists[j]))
                
                # Since uncovered_indices comes from removing one block, its size is at most C(8,5)=56.
                # We can iterate through all blocks? 75582 blocks.
                # For each block, check if it covers all uncovered indices.
                
                # Let's try this only if the number of uncovered indices is small.
                if len(uncovered_indices) < 10:
                    for j in range(len(all_blocks)):
                        if j == b_idx_i:
                            continue
                        # Check if block j covers all uncovered indices
                        # We can use a set intersection or a loop
                        # Since uncovered_indices is small, a loop is fine.
                        is_candidate = True
                        for u_idx in uncovered_indices:
                            if u_idx not in block_cover_lists[j]:
                                is_candidate = False
                                break
                        if is_candidate:
                            # Add block j
                            for idx in block_cover_lists[j]:
                                coverage[idx] += 1
                            
                            # Replace i with j
                            current_indices.pop(i)
                            current_indices.append(j)
                            improved = True
                            break # Restart outer loop
                            
                    if improved:
                        break # Restart outer loop
            else:
                # No uncovered indices, meaning we could have removed i without adding.
                # This case should have been caught by the previous "remove one block" logic.
                # But maybe the order matters? No, if coverage is all > 0, it was safe to remove.
                # Why didn't we remove it?
                # Ah, the previous loop breaks immediately upon finding a removal.
                # If we didn't find a removal, it means for every i, there was a t-set with coverage 1.
                # So this branch (no uncovered indices after removal) should never happen if the previous loop ran correctly.
                # Wait, if we remove i, and coverage becomes all > 0, then it was safe to remove.
                # So the previous loop would have caught it.
                # So this branch is likely dead code or only happens due to race conditions in logic?
                # Let's just handle it by removing i.
                current_indices.pop(i)
                improved = True
                break # Restart outer loop
            
            # Restore coverage if we didn't improve
            if not improved:
                for idx in block_cover_lists[b_idx_i]:
                    coverage[idx] += 1
            
            if improved:
                break # Restart outer loop

    if not current_indices:
        return blocks

    return [all_blocks[b_idx] for b_idx in current_indices]
