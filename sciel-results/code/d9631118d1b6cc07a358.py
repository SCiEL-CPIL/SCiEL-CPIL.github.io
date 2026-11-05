"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def _greedy_cover_ratio(v, k, t, deadline):
    """
    Greedy set cover using ratio-based selection: maximize newly_covered / penalty.
    The penalty encourages selecting blocks that cover many NEW triples relative to
    their size/overlap with already-selected blocks.
    Returns list of blocks.
    """
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, the triple indices it covers
    block_triple_indices = []
    for block in all_blocks:
        indices = []
        for triple in itertools.combinations(block, t):
            indices.append(triple_to_idx[triple])
        block_triple_indices.append(indices)
    
    # triple_to_blocks: for each triple, list of block indices covering it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi in range(num_blocks):
        for ti in block_triple_indices[bi]:
            triple_to_blocks[ti].append(bi)
    
    # block_uncovered_count[bi] = number of currently uncovered triples in block bi
    block_uncovered_count = np.array([len(bti) for bti in block_triple_indices], dtype=np.int32)
    
    # uncovered: boolean array
    uncovered = np.ones(num_triples, dtype=bool)
    num_uncovered = num_triples
    
    # selected_blocks: set of block indices already selected
    selected = set()
    
    blocks = []
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Ratio-based selection: maximize newly_covered / (1 + overlap_penalty)
        # overlap_penalty = number of triples in this block that are already covered
        best_block = -1
        best_ratio = -1
        
        for bi in range(num_blocks):
            if bi in selected:
                continue
            new_count = block_uncovered_count[bi]
            if new_count <= 0:
                continue
            total_in_block = len(block_triple_indices[bi])
            # Penalty: how much of this block is already covered by other selected blocks
            # We approximate by looking at how many triples in this block have been covered
            # by previously selected blocks. We track this via triple_cover_count.
            # Actually, let's use a simpler heuristic: ratio = new_count / sqrt(total_in_block)
            # This penalizes larger blocks slightly, encouraging more focused coverage.
            ratio = new_count / (1.0 + 0.1 * total_in_block)
            if ratio > best_ratio:
                best_ratio = ratio
                best_block = bi
        
        if best_block < 0:
            break
        
        blocks.append(list(all_blocks[best_block]))
        selected.add(best_block)
        
        # Update coverage
        for ti in block_triple_indices[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                num_uncovered -= 1
                for bi2 in triple_to_blocks[ti]:
                    block_uncovered_count[bi2] -= 1
    
    return blocks


def _greedy_cover_with_tracking(v, k, t, deadline, use_ratio=False):
    """
    Greedy set cover with full tracking for post-processing.
    If use_ratio is True, uses ratio-based selection instead of pure max-gain.
    Returns (blocks, triple_to_block_ids) where triple_to_block_ids maps
    each triple to the list of block indices that cover it.
    """
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, the triple indices it covers
    block_triple_indices = []
    for block in all_blocks:
        indices = []
        for triple in itertools.combinations(block, t):
            indices.append(triple_to_idx[triple])
        block_triple_indices.append(indices)
    
    # triple_to_blocks: for each triple, list of block indices
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi in range(num_blocks):
        for ti in block_triple_indices[bi]:
            triple_to_blocks[ti].append(bi)
    
    # block_uncovered_count[bi] = number of currently uncovered triples in block bi
    block_uncovered_count = np.array([len(bti) for bti in block_triple_indices], dtype=np.int32)
    
    # uncovered: boolean array
    uncovered = np.ones(num_triples, dtype=bool)
    num_uncovered = num_triples
    
    blocks = []
    selected = set()
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        if use_ratio:
            # Ratio-based selection
            best_block = -1
            best_ratio = -1
            for bi in range(num_blocks):
                if bi in selected:
                    continue
                new_count = block_uncovered_count[bi]
                if new_count <= 0:
                    continue
                total_in_block = len(block_triple_indices[bi])
                ratio = new_count / (1.0 + 0.1 * total_in_block)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_block = bi
        else:
            best_block = int(np.argmax(block_uncovered_count))
        
        best_count = block_uncovered_count[best_block]
        if best_count <= 0:
            break
        
        blocks.append(list(all_blocks[best_block]))
        selected.add(best_block)
        
        # Update
        for ti in block_triple_indices[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                num_uncovered -= 1
                for bi2 in triple_to_blocks[ti]:
                    block_uncovered_count[bi2] -= 1
    
    return blocks, triple_to_blocks, block_triple_indices


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Remove redundant blocks: if a block's t-subsets are all covered by other blocks,
    remove it. Uses iterative single-pass removal for efficiency.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    n = len(blocks)
    if n == 0:
        return blocks
    
    # For each block, compute which triples it covers
    block_triples = []
    for block in blocks:
        triples = set()
        for triple in itertools.combinations(block, t):
            triples.add(triple_to_idx[triple])
        block_triples.append(triples)
    
    # For each triple, how many blocks cover it
    triple_count = np.zeros(num_triples, dtype=np.int32)
    for bt in block_triples:
        for ti in bt:
            triple_count[ti] += 1
    
    kept = list(range(n))
    kept_set = set(kept)
    
    # Iteratively remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Recompute triple counts for kept blocks
        triple_count = np.zeros(num_triples, dtype=np.int32)
        for bi in kept:
            for ti in block_triples[bi]:
                triple_count[ti] += 1
        
        new_kept = []
        for bi in kept:
            # Check if bi is redundant: all its triples are covered by at least one other kept block
            is_redundant = True
            for ti in block_triples[bi]:
                if triple_count[ti] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                changed = True
            else:
                new_kept.append(bi)
        
        kept = new_kept
    
    return [blocks[i] for i in kept]


def _local_search(blocks, v, k, t, deadline, max_iters=100):
    """
    Local search: aggressively try to remove blocks, with randomized order.
    Also tries swapping blocks to see if it enables further removals.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    n = len(blocks)
    if n == 0:
        return blocks
    
    # Precompute block triples
    block_triples = []
    for block in blocks:
        triples = set()
        for triple in itertools.combinations(block, t):
            triples.add(triple_to_idx[triple])
        block_triples.append(triples)
    
    # Maintain triple_cover_count
    triple_cover_count = np.zeros(num_triples, dtype=np.int32)
    for bt in block_triples:
        for ti in bt:
            triple_cover_count[ti] += 1
    
    kept = list(range(n))
    kept_set = set(kept)
    
    # Multiple passes with different random orders
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        made_change = False
        
        # Random order for this pass
        order = list(kept)
        random.shuffle(order)
        
        for bi in order:
            if time.time() > deadline:
                break
            if bi not in kept_set:
                continue
            # Check if removing bi leaves all triples covered
            can_remove = True
            for ti in block_triples[bi]:
                if triple_cover_count[ti] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove bi
                for ti in block_triples[bi]:
                    triple_cover_count[ti] -= 1
                kept_set.discard(bi)
                made_change = True
        
        if not made_change:
            break
    
    return [blocks[i] for i in kept]


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1)
    such that every t-subset is contained in at least one block.
    Uses ratio-based greedy followed by aggressive redundancy removal and local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Phase 1: Greedy set cover with ratio-based selection
    blocks, triple_to_blocks, block_triple_indices = _greedy_cover_with_tracking(v, k, t, deadline, use_ratio=True)
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant(blocks, v, k, t, deadline)
    
    # Phase 3: Local search for further reduction
    blocks = _local_search(blocks, v, k, t, deadline, max_iters=100)
    
    return blocks

# EVOLVE-BLOCK-END
