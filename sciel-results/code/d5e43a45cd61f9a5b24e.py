import itertools
import time
import random
import numpy as np
from collections import defaultdict

def _precompute(v, k, t):
    """
    Precompute mappings for efficient set cover operations.
    Returns:
    - tsubs: list of all t-subsets as tuples
    - tsub_id: dict mapping t-subset tuple to index
    - block_tsubs: dict mapping k-block tuple to numpy array of t-subset indices
    - tsub_to_blocks: dict mapping t-subset index to list of blocks
    """
    tsubs = list(itertools.combinations(range(v), t))
    tsub_id = {ts: i for i, ts in enumerate(tsubs)}
    
    block_tsubs = {}
    tsub_to_blocks = defaultdict(list)
    for block in itertools.combinations(range(v), k):
        covered = []
        for ts in itertools.combinations(block, t):
            idx = tsub_id[ts]
            covered.append(idx)
            tsub_to_blocks[idx].append(block)
        block_tsubs[block] = np.array(covered, dtype=np.int32)
        
    return tsubs, tsub_id, block_tsubs, tsub_to_blocks

def _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs, tsub_to_blocks):
    """
    Greedy set cover using random selection of uncovered t-subsets.
    """
    n_tsubs = len(tsubs)
    uncovered_mask = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    while uncovered_mask.any():
        # Pick a random uncovered t-subset
        uncovered_indices = np.where(uncovered_mask)[0]
        ts_id = int(uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)])
        
        candidates = tsub_to_blocks[ts_id]
        
        best_block = None
        best_gain = -1
        
        for block in candidates:
            covered = block_tsubs[block]
            gain = int(np.sum(uncovered_mask[covered]))
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            break
            
        blocks.append(best_block)
        uncovered_mask[block_tsubs[best_block]] = False
        
    return blocks

def _local_search(blocks, v, k, t, deadline, tsubs, tsub_id, block_tsubs):
    """
    Local search: Removal and 2-Block Merge.
    """
    blocks = [list(b) for b in blocks]
    
    def build_coverage(blocks):
        coverage = defaultdict(set)
        for bi, block in enumerate(blocks):
            b_tuple = tuple(block)
            if b_tuple in block_tsubs:
                for ts_id in block_tsubs[b_tuple]:
                    coverage[int(ts_id)].add(bi)
        return coverage

    def get_unique_sets(bi, blocks, coverage):
        b_tuple = tuple(blocks[bi])
        unique = []
        if b_tuple not in block_tsubs:
            return unique
        for ts_id in block_tsubs[b_tuple]:
            ts_id = int(ts_id)
            cov = coverage.get(ts_id)
            if cov is not None and len(cov) == 1 and bi in cov:
                unique.append(ts_id)
        return unique

    def rebuild_coverage(blocks):
        coverage = defaultdict(set)
        for bi, block in enumerate(blocks):
            b_tuple = tuple(block)
            if b_tuple in block_tsubs:
                for ts_id in block_tsubs[b_tuple]:
                    coverage[int(ts_id)].add(bi)
        return coverage

    coverage = build_coverage(blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_blocks = len(blocks)
        
        # Pass 1: Removal
        for bi in range(n_blocks):
            if time.time() >= deadline:
                break
            unique = get_unique_sets(bi, blocks, coverage)
            if not unique:
                blocks.pop(bi)
                coverage = rebuild_coverage(blocks)
                improved = True
                break
        
        if improved:
            continue

        # Pass 2: Merging
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            for bj in range(bi + 1, len(blocks)):
                if time.time() >= deadline:
                    break
                unique_i = get_unique_sets(bi, blocks, coverage)
                unique_j = get_unique_sets(bj, blocks, coverage)
                
                if not unique_i and not unique_j:
                    continue
                    
                union_elems = set()
                for ts_id in unique_i:
                    union_elems.update(tsubs[ts_id])
                for ts_id in unique_j:
                    union_elems.update(tsubs[ts_id])
                    
                if len(union_elems) > k:
                    continue
                    
                remaining = [x for x in range(v) if x not in union_elems]
                needed = k - len(union_elems)
                
                if needed == 0:
                    candidate = tuple(sorted(union_elems))
                    if candidate in block_tsubs:
                        blocks.pop(bj)
                        blocks.pop(bi)
                        blocks.insert(bi, list(candidate))
                        coverage = rebuild_coverage(blocks)
                        improved = True
                        break
                elif needed <= 2:
                    for ext in itertools.combinations(remaining, needed):
                        candidate_elems = union_elems.union(ext)
                        candidate = tuple(sorted(candidate_elems))
                        if candidate in block_tsubs:
                            blocks.pop(bj)
                            blocks.pop(bi)
                            blocks.insert(bi, list(candidate))
                            coverage = rebuild_coverage(blocks)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
                
    return blocks

def construct(v, k, t):
    deadline = time.time() + 145
    rng = random.Random(42)
    
    tsubs, tsub_id, block_tsubs, tsub_to_blocks = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Greedy restarts (fast, ~1-2s each)
    greedy_deadline = deadline - 80
    while time.time() < greedy_deadline:
        blocks = _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs, tsub_to_blocks)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
    
    # Phase 2: Local search with remaining time
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, deadline - 2, tsubs, tsub_id, block_tsubs)
        
    return best_blocks
