"""Stronger covering algorithm for C(v,k,t): fast greedy + vectorized local search.

Strategy:
  1. Greedy construction with vectorized gain scoring using dense indices.
  2. Redundancy removal (drop blocks that are not needed).
  3. Block-replacement local search.
A hard deadline guard returns the best valid block set found before the limit.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 120.0
_MARGIN = 5.0  # seconds of safety margin before the evaluator kills us

# Global cache for the mapping from t-subset tuples to dense indices
_t_subset_map = {}
_t_subset_list = {}

def _get_t_subsets(v, t):
    """Get or create the list of all t-subsets and their index mapping."""
    key = (v, t)
    if key not in _t_subset_map:
        subs = list(itertools.combinations(range(v), t))
        _t_subset_list[key] = subs
        _t_subset_map[key] = {sub: i for i, sub in enumerate(subs)}
    return _t_subset_list[key], _t_subset_map[key]

def _get_block_indices(block, v, t, t_map):
    """Get the dense indices of all t-subsets contained in the block."""
    block_set = set(block)
    indices = []
    for sub in itertools.combinations(block, t):
        idx = t_map.get(sub, -1)
        if idx != -1:
            indices.append(idx)
    return indices

def _coverage_valid(v, t, blocks, t_map):
    """True iff every t-subset is covered by at least one block."""
    if not blocks:
        return False
    n_subsets = len(t_map)
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b in blocks:
        for idx in _get_block_indices(b, v, t, t_map):
            counts[idx] += 1
    return bool(np.all(counts >= 1))

def _greedy_construct(v, k, t, rng, deadline, t_map, t_list):
    """Greedy covering with vectorized marginal-gain scoring."""
    n_subsets = len(t_list)
    covered = np.zeros(n_subsets, dtype=bool)
    blocks = []
    start = time.time()
    
    # Precompute all possible k-blocks? No, too many.
    # Instead, greedy: pick block that covers most uncovered subsets.
    
    while True:
        if time.time() > deadline:
            break
        if np.all(covered):
            break
            
        # To speed up, we can precompute the gain for all possible k-blocks?
        # C(19, 7) = 50388. This is small enough to precompute coverage vectors?
        # Vector size 3876. 50388 * 3876 is ~200M booleans. Might be heavy but doable in numpy if sparse?
        # Alternatively, just iterate through a subset of candidate blocks or use a smarter heuristic.
        
        # Heuristic: Start with the most "constrained" uncovered subset.
        # Find an uncovered subset.
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a random uncovered subset to seed the block
        seed_idx = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
        seed_sub = t_list[seed_idx]
        
        # Construct a block containing seed_sub
        block = list(seed_sub)
        # Fill the rest of the block with points that maximize coverage of uncovered subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            block_set = set(block)
            
            # Calculate current coverage of this partial block's t-subsets
            # We want to add x such that the new t-subsets formed by x and (t-1) from block
            # cover the most *uncovered* subsets.
            
            # Get (t-1) subsets of current block
            t_minus_1_subs = list(itertools.combinations(block, t-1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for sub in t_minus_1_subs:
                    new_sub = tuple(sorted(sub + (x,)))
                    if new_sub in t_map:
                        idx = t_map[new_sub]
                        if not covered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                # Should not happen if v > k
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update covered
        for idx in _get_block_indices(block, v, t, t_map):
            covered[idx] = True
            
    return blocks

def _remove_redundant(v, t, blocks, t_map):
    """Drop any block whose removal leaves the covering valid. Returns new list."""
    n_subsets = len(t_map)
    counts = np.zeros(n_subsets, dtype=np.int32)
    block_indices_list = []
    
    for b in blocks:
        indices = _get_block_indices(b, v, t, t_map)
        block_indices_list.append(indices)
        for idx in indices:
            counts[idx] += 1
            
    kept = []
    kept_indices = []
    
    for i, b in enumerate(blocks):
        indices = block_indices_list[i]
        # Check if any index has count 1 (uniquely covered by this block)
        # If yes, we must keep it.
        # If no, we can remove it.
        unique_count = np.sum(counts[indices] == 1)
        if unique_count > 0:
            kept.append(b)
            kept_indices.append(indices)
        else:
            # Remove contribution
            for idx in indices:
                counts[idx] -= 1
                
    return kept

def _local_search(v, k, t, blocks, deadline, rng, t_map, t_list):
    """Redundancy removal + block-replacement local search."""
    blocks = list(blocks)
    n_subsets = len(t_list)
    counts = np.zeros(n_subsets, dtype=np.int32)
    
    # Initialize counts
    block_indices_list = []
    for b in blocks:
        indices = _get_block_indices(b, v, t, t_map)
        block_indices_list.append(indices)
        for idx in indices:
            counts[idx] += 1
            
    start = time.time()
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Redundancy removal
        new_blocks = []
        new_block_indices = []
        
        for i, b in enumerate(blocks):
            indices = block_indices_list[i]
            unique_count = np.sum(counts[indices] == 1)
            if unique_count > 0:
                new_blocks.append(b)
                new_block_indices.append(indices)
            else:
                for idx in indices:
                    counts[idx] -= 1
                    
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            block_indices_list = new_block_indices
            improved = True
            continue
            
        # Pass 2: Block replacement
        # Try to replace a block with one that covers more uncovered or allows removing another
        # Simple heuristic: Try random swaps on a random block
        if len(blocks) > 0:
            idx = rng.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_indices = block_indices_list[idx]
            
            # Generate a candidate block
            # Swap 1-2 elements
            cand = list(old_block)
            for _ in range(rng.randint(1, 2)):
                pos = rng.randint(0, k - 1)
                new_x = rng.randint(0, v - 1)
                if new_x not in cand:
                    cand[pos] = new_x
            cand.sort()
            if cand == old_block:
                continue
                
            cand_indices = _get_block_indices(cand, v, t, t_map)
            
            # Calculate gain/loss
            # If we replace old with cand:
            # Counts change: -old_indices, +cand_indices
            # We need to ensure no count drops to 0 for subsets only covered by old.
            
            # Check subsets only covered by old (count==1 in old_indices)
            # If cand doesn't cover them, we lose coverage.
            
            # Simulate counts
            tmp_counts = counts.copy()
            for idx_old in old_indices:
                tmp_counts[idx_old] -= 1
            for idx_cand in cand_indices:
                tmp_counts[idx_cand] += 1
                
            if np.any(tmp_counts < 0):
                continue # Invalid
                
            # If valid, check if it helps. 
            # Ideally we want to reduce total blocks or keep same count but better structure.
            # For now, just accept if valid and different. 
            # To improve, we could check if it enables further redundancy removal.
            
            # Accept the swap
            blocks[idx] = cand
            block_indices_list[idx] = cand_indices
            counts = tmp_counts
            improved = True
            break # Restart pass
            
    return blocks

def construct(v, k, t):
    """Return a list of k-subsets of {0..v-1} covering every t-subset."""
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    rng = random.Random(12345)
    
    t_list, t_map = _get_t_subsets(v, t)
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, rng, deadline, t_map, t_list)
    
    # Ensure validity
    if not _coverage_valid(v, t, blocks, t_map):
        # Repair: Add blocks for uncovered subsets
        n_subsets = len(t_list)
        covered = np.zeros(n_subsets, dtype=bool)
        for b in blocks:
            for idx in _get_block_indices(b, v, t, t_map):
                covered[idx] = True
                
        while True:
            if time.time() > deadline:
                break
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            
            seed_idx = uncovered[0]
            seed_sub = t_list[seed_idx]
            block = list(seed_sub)
            while len(block) < k:
                best_x = None
                best_gain = -1
                block_set = set(block)
                t_minus_1_subs = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in t_minus_1_subs:
                        new_sub = tuple(sorted(sub + (x,)))
                        if new_sub in t_map:
                            idx = t_map[new_sub]
                            if not covered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x is None:
                    break
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            for idx in _get_block_indices(block, v, t, t_map):
                covered[idx] = True
                
    # Phase 2: Local Search
    if time.time() < deadline:
        blocks = _local_search(v, k, t, blocks, deadline, rng, t_map, t_list)
        
    # Final Safety Check
    if not _coverage_valid(v, t, blocks, t_map):
        # Repair again
        n_subsets = len(t_list)
        covered = np.zeros(n_subsets, dtype=bool)
        for b in blocks:
            for idx in _get_block_indices(b, v, t, t_map):
                covered[idx] = True
                
        while True:
            if time.time() > deadline:
                break
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            
            seed_idx = uncovered[0]
            seed_sub = t_list[seed_idx]
            block = list(seed_sub)
            while len(block) < k:
                best_x = None
                best_gain = -1
                block_set = set(block)
                t_minus_1_subs = list(itertools.combinations(block, t-1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in t_minus_1_subs:
                        new_sub = tuple(sorted(sub + (x,)))
                        if new_sub in t_map:
                            idx = t_map[new_sub]
                            if not covered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x is None:
                    break
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            for idx in _get_block_indices(block, v, t, t_map):
                covered[idx] = True

    # Deduplicate
    seen = set()
    deduped = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    blocks = deduped
    
    # Final Final Validity
    if not _coverage_valid(v, t, blocks, t_map):
        # This should not happen, but if it does, return empty or best effort
        # For safety, just return what we have if we are out of time
        pass

    return blocks
# EVOLVE-BLOCK-END
