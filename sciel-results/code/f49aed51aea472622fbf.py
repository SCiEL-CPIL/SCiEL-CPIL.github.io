import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    if v == 24 and k == 7 and t == 3:
        return construct_24_7_3(deadline)
    else:
        return construct_generic(v, k, t, deadline)

def construct_24_7_3(deadline):
    v, k, t = 24, 7, 3
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Precompute blocks and their masks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Masks for triples covered by each block
    block_masks = [0] * num_blocks
    block_triple_lists = [[] for _ in range(num_blocks)]
    
    # Precomputation
    for i, block in enumerate(all_blocks):
        mask = 0
        t_list = []
        for triple in itertools.combinations(block, t):
            idx = tsub_to_idx[triple]
            mask |= (1 << idx)
            t_list.append(idx)
        block_masks[i] = mask
        block_triple_lists[i] = t_list

    def greedy_construct(randomize=False):
        uncovered_mask = (1 << num_tsubs) - 1
        blocks = []
        
        while uncovered_mask:
            best_idx = -1
            best_cov = -1
            
            for i in range(num_blocks):
                m = block_masks[i]
                cov = (m & uncovered_mask).bit_count()
                
                if randomize:
                    cov += random.random() * 0.1
                
                if cov > best_cov:
                    best_cov = cov
                    best_idx = i
            
            if best_idx == -1:
                break
                
            blocks.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
            if time.time() > deadline - 10:
                break
                
        return blocks

    def refine(blocks):
        if not blocks:
            return []
            
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts[t_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                redundant = True
                for t_idx in block_triple_lists[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_triple_lists[b_idx]:
                        counts[t_idx] -= 1
                    blocks.pop(i)
                    changed = True
                    
                    if time.time() > deadline - 5:
                        return blocks
        return blocks

    def local_search(blocks, max_iters=5000):
        """
        Local search: try swapping a block with another that covers more of the 
        unique triples of the removed block, or removing a block if it's redundant.
        Objective: minimize number of blocks while maintaining coverage.
        """
        if not blocks:
            return blocks
        
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in block_triple_lists[b_idx]:
                counts[t_idx] += 1
        
        # Precompute: for each block, which other blocks cover each of its triples
        # This is expensive, so we compute on the fly but cache the triple lists
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            if time.time() > deadline - 15:
                break
            
            # Try removing each block
            for i in range(len(blocks)):
                b_idx = blocks[i]
                # Check if all triples in this block are covered by other blocks
                all_covered = True
                for t_idx in block_triple_lists[b_idx]:
                    if counts[t_idx] <= 1:
                        all_covered = False
                        break
                if all_covered:
                    # Remove this block
                    for t_idx in block_triple_lists[b_idx]:
                        counts[t_idx] -= 1
                    blocks.pop(i)
                    improved = True
                    break  # Restart from beginning
            
            if improved or iters >= max_iters or time.time() > deadline - 15:
                continue
            
            # Try swapping: replace block i with block j that covers more unique triples
            # But only if j covers all triples that i uniquely covers
            for i in range(len(blocks)):
                if time.time() > deadline - 15:
                    break
                b_idx_i = blocks[i]
                # Find triples uniquely covered by block i
                unique_triples = []
                for t_idx in block_triple_lists[b_idx_i]:
                    if counts[t_idx] == 1:
                        unique_triples.append(t_idx)
                
                if not unique_triples:
                    continue
                
                # Try to find a replacement block that covers all unique_triples
                # and has lower marginal cost (fewer additional triples)
                best_j = -1
                best_gain = 0
                best_cost = 0
                
                for j in range(num_blocks):
                    if j == b_idx_i:
                        continue
                    # Check if block j covers all unique triples
                    covers_all = True
                    for t_idx in unique_triples:
                        if not (block_masks[j] >> t_idx) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate gain: how many new triples does j cover that i doesn't?
                    # Actually, we want to minimize total blocks, so any valid swap that
                    # allows removal of i is good. But we should prefer j that covers
                    # many triples already covered, to make future removals easier.
                    
                    # Count how many of j's triples are already covered (by others)
                    overlap = 0
                    for t_idx in block_triple_lists[j]:
                        if counts[t_idx] >= 2:  # Covered by at least one other block (including i)
                            overlap += 1
                    
                    # We want high overlap (j's triples are redundant)
                    if overlap > best_gain:
                        best_gain = overlap
                        best_j = j
                        best_cost = 0
                
                if best_j != -1:
                    # Perform swap: remove i, add j
                    for t_idx in block_triple_lists[b_idx_i]:
                        counts[t_idx] -= 1
                    for t_idx in block_triple_lists[best_j]:
                        counts[t_idx] += 1
                    blocks[i] = best_j
                    improved = True
                    break  # Restart from beginning
        
        return blocks

    # Run 1: Deterministic greedy + refine + local search
    blocks_idx = greedy_construct(randomize=False)
    blocks_idx = refine(blocks_idx)
    blocks_idx = local_search(blocks_idx)
    
    best_blocks = [list(all_blocks[i]) for i in blocks_idx]
    best_len = len(blocks_idx)
    
    # Run 2: Multiple randomized restarts with local search
    if time.time() < deadline - 20:
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            blocks_idx_r = greedy_construct(randomize=True)
            blocks_idx_r = refine(blocks_idx_r)
            blocks_idx_r = local_search(blocks_idx_r)
            if len(blocks_idx_r) < best_len:
                best_len = len(blocks_idx_r)
                blocks_idx = blocks_idx_r
                best_blocks = [list(all_blocks[i]) for i in blocks_idx]
    
    return best_blocks

def construct_generic(v, k, t, deadline):
    # Fallback for other parameters
    # Fixed version of the original logic
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    def greedy_construct():
        uncovered = set(range(num_tsubs))
        blocks = []
        
        while uncovered:
            best_block = None
            best_count = -1
            candidates = set()
            # Sample a subset of uncovered triples to generate candidates
            sample_size = min(20, len(uncovered))
            for idx in random.sample(list(uncovered), sample_size):
                triple = all_tsubs[idx]
                remaining = [x for x in range(v) if x not in triple]
                for combo in itertools.combinations(remaining, k - t):
                    block = tuple(sorted(triple + combo))
                    candidates.add(block)
            
            if not candidates:
                break
                
            for block in candidates:
                count = 0
                for triple in itertools.combinations(block, t):
                    idx = tsub_to_idx[triple]
                    if idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            for triple in itertools.combinations(best_block, t):
                idx = tsub_to_idx[triple]
                uncovered.discard(idx)
            
            if time.time() > deadline - 5:
                break
        
        return blocks

    blocks = greedy_construct()
    
    # Simple refinement: remove redundant blocks
    # Using sets for speed in generic case (might be slow if v is large, but fallback)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            block_set = set(blocks[i])
            redundant = True
            for triple in itertools.combinations(blocks[i], t):
                triple_set = set(triple)
                covered = False
                for j, other_block in enumerate(blocks):
                    if j == i:
                        continue
                    if triple_set.issubset(set(other_block)):
                        covered = True
                        break
                if not covered:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                
    return blocks
