"""Covering design C(v,k,t) via vectorized multi-start greedy + local reduction."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a vectorized multi-start greedy
    approach with local block reduction.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets it belongs to
    # point_tsubs[p] = list of tsub indices that contain p
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)

    # Convert to numpy arrays for speed
    point_tsub_arrays = [np.array(pts, dtype=np.int32) for pts in point_tsubs]

    def greedy_construct(random_seed=None):
        """Run one greedy construction, optionally randomized."""
        rng = random.Random(random_seed)
        
        # uncovered_mask[i] = True if t-subset i is still uncovered
        uncovered_mask = np.ones(n_tsubs, dtype=bool)
        blocks = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset to start the block
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            start_tsub = all_tsubs[start_idx]
            
            # Build block starting with the t-subset
            block = list(start_tsub)
            rng.shuffle(block)
            block_set = set(block)
            
            # Compute current coverage of this block
            block_tsubs = set()
            for comb in itertools.combinations(block, t):
                block_tsubs.add(comb)
            
            # Iteratively add points that maximize coverage of uncovered t-subsets
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                best_point = None
                best_gain = 0
                
                # For each candidate point not in block, compute gain
                candidates = [x for x in range(v) if x not in block_set]
                
                # Vectorized gain computation
                # For each candidate x, gain = number of uncovered t-subsets 
                # that contain x and are subsets of (block + {x})
                # A t-subset S is covered by block+{x} iff S ⊆ block ∪ {x}
                # i.e., S ∩ (complement of (block ∪ {x})) = ∅
                # Equivalently: S ⊆ block ∪ {x}
                
                # More efficient: for each candidate x, the t-subsets it can cover
                # with the current block are those t-subsets S where:
                #   S ⊆ (block ∪ {x}) and |S ∩ block| >= t-1 (since x is new)
                # Actually: S ⊆ block ∪ {x} means S \ {x} ⊆ block
                # So we need t-subsets S such that S \ {x} ⊆ block
                # This means S contains x and S\{x} is a (t-1)-subset of block
                # OR S doesn't contain x and S ⊆ block (but those are already covered)
                
                # So gain(x) = number of uncovered t-subsets S where x ∈ S and S\{x} ⊆ block
                # = number of uncovered t-subsets in point_tsubs[x] that are subsets of block ∪ {x}
                # A t-subset S containing x is a subset of block ∪ {x} iff S\{x} ⊆ block
                # S\{x} has size t-1, so we need (t-1)-subsets of block
                
                # Precompute: for the current block, which (t-1)-subsets exist?
                # For each candidate x, gain = count of uncovered tsubs in point_tsubs[x]
                # where the other t-1 elements are all in block
                
                # Efficient approach: 
                # For each (t-1)-subset R of block, and for each x not in block,
                # the t-subset R ∪ {x} is covered if we add x.
                # So gain(x) = number of (t-1)-subsets R of block such that 
                # R ∪ {x} is an uncovered t-subset.
                
                # Let's compute this differently:
                # For each x in candidates, look at point_tsubs[x] (all tsubs containing x)
                # and count how many are uncovered AND have their other t-1 elements in block.
                
                # Vectorized: for candidate x, mask = uncovered_mask[point_tsub_arrays[x]]
                # Then filter: for each such tsub, check if all other elements are in block.
                
                # This is still expensive. Let's use a different approach:
                # Precompute for the current block, the set of (t-1)-subsets.
                # Then for each candidate x, gain = |{R ∈ C(block, t-1) : R ∪ {x} is uncovered}|
                
                block_tminus1 = list(itertools.combinations(block, t - 1))
                
                # For each candidate, compute gain
                gains = {}
                for x in candidates:
                    gain = 0
                    for R in block_tminus1:
                        ts = tuple(sorted(R + (x,)))
                        if ts in tsub_to_idx:
                            idx = tsub_to_idx[ts]
                            if uncovered_mask[idx]:
                                gain += 1
                    gains[x] = gain
                
                if max(gains.values()) == 0:
                    # No point can improve coverage; pick a random one
                    best_point = rng.choice(candidates)
                else:
                    # Pick the point with max gain, break ties randomly
                    max_gain = max(gains.values())
                    best_candidates = [x for x, g in gains.items() if g == max_gain]
                    best_point = rng.choice(best_candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            # Finalize block
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered mask
            for comb in itertools.combinations(block, t):
                idx = tsub_to_idx[comb]
                uncovered_mask[idx] = False
        
        # Convert to lists
        return [list(b) for b in blocks]

    def reduce_blocks(blocks):
        """Try to remove redundant blocks."""
        if not blocks:
            return blocks
        
        # Count how many times each t-subset is covered
        coverage_count = defaultdict(int)
        for block in blocks:
            for comb in itertools.combinations(block, t):
                coverage_count[comb] += 1
        
        reduced = []
        for i, block in enumerate(blocks):
            if time.time() > deadline:
                break
            
            # Check if this block is redundant (all its t-subsets are covered by other blocks)
            redundant = True
            for comb in itertools.combinations(block, t):
                if coverage_count[comb] <= 1:
                    redundant = False
                    break
            
            if not redundant:
                reduced.append(block)
                # Update coverage counts
                for comb in itertools.combinations(block, t):
                    coverage_count[comb] -= 1
            # If redundant, skip it and don't update counts
        
        return reduced

    # Multi-start greedy with different random seeds
    best_blocks = None
    best_len = float('inf')
    
    num_starts = 10  # Try multiple starts
    
    for seed in range(num_starts):
        if time.time() > deadline:
            break
        
        blocks = greedy_construct(random_seed=seed)
        
        if not blocks:
            continue
        
        # Try to reduce
        blocks = reduce_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # If no valid result found, fall back to deterministic greedy
    if best_blocks is None:
        best_blocks = greedy_construct(random_seed=None)
    
    return best_blocks
