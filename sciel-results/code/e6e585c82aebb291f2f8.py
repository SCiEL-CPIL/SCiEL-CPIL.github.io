import time
import random
import itertools
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and the t-subsets they cover
    # This allows O(1) lookup of what a block covers
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # block_to_triples[i] = list of triple indices covered by all_k_subsets[i]
    block_to_triples = [None] * n_k_subsets
    # triple_to_blocks[j] = list of block indices covering all_t_subsets[j]
    triple_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(all_k_subsets):
        triples = []
        for triple in itertools.combinations(block, t):
            idx = t_subset_to_idx[triple]
            triples.append(idx)
            triple_to_blocks[idx].append(i)
        block_to_triples[i] = triples

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_indices = set(range(n_t_subsets))
        
        blocks = []
        
        while uncovered_indices:
            if time.time() > deadline - 10:
                return None
            
            # Pick a random uncovered triple
            start_idx = next(iter(uncovered_indices))
            
            # Find a block covering start_idx that covers the most other uncovered triples
            best_block_id = -1
            best_gain = -1
            
            candidates = triple_to_blocks[start_idx]
            # Shuffle to break ties randomly
            rng.shuffle(candidates)
            
            for b_id in candidates:
                gain = 0
                for t_idx in block_to_triples[b_id]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_id = b_id
            
            if best_block_id == -1:
                return None
                
            blocks.append(best_block_id)
            
            # Update uncovered
            for t_idx in block_to_triples[best_block_id]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    uncovered_indices.remove(t_idx)
                    
        return blocks

    def reduce_and_optimize(blocks, seed):
        if not blocks:
            return blocks
            
        rng = random.Random(seed)
        
        # Current state
        current_blocks = list(blocks)
        
        # Compute coverage counts
        cov_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_id in current_blocks:
            for t_idx in block_to_triples[b_id]:
                cov_counts[t_idx] += 1
                
        # Phase 1: Simple Reduction
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Try to remove each block
            for i in range(len(current_blocks) - 1, -1, -1):
                b_id = current_blocks[i]
                can_remove = True
                for t_idx in block_to_triples[b_id]:
                    if cov_counts[t_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    for t_idx in block_to_triples[b_id]:
                        cov_counts[t_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                    
        # Phase 2: Local Search (Swap)
        # Try to swap blocks to enable further reductions
        # We look for a block B that has "waste" (covers triples with high count)
        # and replace it with B' that covers "critical" triples (count 1)
        
        swap_attempts = 0
        while time.time() < deadline - 5:
            if not current_blocks:
                break
                
            # Pick a random block to potentially swap
            b_idx = rng.randint(0, len(current_blocks) - 1)
            old_b_id = current_blocks[b_idx]
            
            # Identify critical triples in this block (count == 1)
            critical_triples = []
            for t_idx in block_to_triples[old_b_id]:
                if cov_counts[t_idx] == 1:
                    critical_triples.append(t_idx)
            
            # If no critical triples, we can just remove it (handled in Phase 1, but maybe missed?)
            if not critical_triples:
                # Try to remove
                for t_idx in block_to_triples[old_b_id]:
                    cov_counts[t_idx] -= 1
                current_blocks.pop(b_idx)
                continue
            
            # We need to find a new block B' that covers all critical_triples
            # and is "better" (covers more low-count triples)
            
            # Find intersection of blocks covering each critical triple
            # If there are many critical triples, the candidate set is small
            # If there is 1, it's large.
            
            # Strategy: Pick one critical triple, get its candidate blocks,
            # filter by other critical triples, then score them.
            
            # To avoid O(n) check for every candidate, we sample or use heuristics.
            # Since n_k_subsets is ~346k, and candidates for one triple is ~ (24-3 choose 7-3) = 21C4 = 5985?
            # Wait, block size 7. Triple size 3.
            # Remaining points: 24-3=21. Need 7-3=4 more points.
            # Candidates for one triple: C(21, 4) = 5985.
            # This is small enough to iterate!
            
            target_triple = critical_triples[0]
            candidates = triple_to_blocks[target_triple]
            
            best_new_b_id = -1
            best_score = -1
            
            for cand_id in candidates:
                # Check if cand covers all other critical triples
                valid = True
                for t_idx in critical_triples[1:]:
                    # Check if t_idx is in block_to_triples[cand_id]
                    # This is O(35) if we do linear search, or we can precompute a set
                    # For speed, we assume the list is small enough or we use a set for the block's triples
                    # Let's just check membership.
                    if t_idx not in block_to_triples[cand_id]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Score the candidate: Sum of (1 / cov_counts[t]) for triples in cand?
                # Or simply: Number of triples with count 1 or 2 covered?
                # We want to cover "rare" triples.
                score = 0
                for t_idx in block_to_triples[cand_id]:
                    if cov_counts[t_idx] <= 2:
                        score += 10
                    elif cov_counts[t_idx] <= 5:
                        score += 1
                    else:
                        score -= 1 # Penalty for covering very common triples?
                    
                if score > best_score:
                    best_score = score
                    best_new_b_id = cand_id
                    
            if best_new_b_id != -1 and best_new_b_id != old_b_id:
                # Perform swap
                # Decrement counts for old
                for t_idx in block_to_triples[old_b_id]:
                    cov_counts[t_idx] -= 1
                # Increment counts for new
                for t_idx in block_to_triples[best_new_b_id]:
                    cov_counts[t_idx] += 1
                # Update block list
                current_blocks[b_idx] = best_new_b_id
                
                # After swap, try to remove some blocks again
                # Simple pass
                for i in range(len(current_blocks) - 1, -1, -1):
                    b_id = current_blocks[i]
                    can_remove = True
                    for t_idx in block_to_triples[b_id]:
                        if cov_counts[t_idx] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for t_idx in block_to_triples[b_id]:
                            cov_counts[t_idx] -= 1
                        current_blocks.pop(i)
                        
        # Convert block IDs back to lists of points
        return [list(all_k_subsets[b_id]) for b_id in current_blocks]

    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds
    seed = 0
    while time.time() < deadline - 15:
        greedy_result = greedy_cover(seed)
        if greedy_result is not None:
            optimized = reduce_and_optimize(greedy_result, seed)
            if len(optimized) < best_count:
                best_count = len(optimized)
                best_blocks = optimized
        seed += 1
        
    if best_blocks is None:
        # Fallback
        return []
        
    return best_blocks

# EVOLVE-BLOCK-END
