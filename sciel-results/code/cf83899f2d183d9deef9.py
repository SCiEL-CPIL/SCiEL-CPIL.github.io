"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from math import comb


def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using multi-start greedy with best-superset selection,
    aggressive reduction, and local search polish.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point, which t-subsets it participates in
    # point_to_tsubs[x] = set of t-subset IDs containing x
    point_to_tsubs = [set() for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for x in s:
            point_to_tsubs[x].add(i)
    
    def greedy_cover_best_superset(seed=None):
        """Run one greedy pass using best-superset selection. Returns blocks list."""
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(range(N_t))
        blocks = []
        
        while uncovered:
            # Find the k-subset that covers the most uncovered t-subsets
            # Use a heuristic: start from a random uncovered t-subset, then greedily extend
            start_id = random.choice(list(uncovered))
            start_subset = all_t_subsets[start_id]
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                # Find point that maximizes coverage of uncovered t-subsets
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        cid = subset_to_id.get(cand)
                        if cid is not None and cid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Remove all t-subsets covered by this block
            for comb in itertools.combinations(block, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    uncovered.discard(cid)
        
        return blocks
    
    def greedy_cover_random_start(seed=None):
        """Greedy with randomization in point selection. Returns blocks list."""
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(range(N_t))
        blocks = []
        
        while uncovered:
            start_id = random.choice(list(uncovered))
            start_subset = all_t_subsets[start_id]
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                # Find top candidates, then pick randomly among top few
                gains = []
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        cid = subset_to_id.get(cand)
                        if cid is not None and cid in uncovered:
                            gain += 1
                    gains.append((gain, x))
                
                gains.sort(reverse=True)
                # Pick from top 3 with some randomness
                top_n = min(3, len(gains))
                if top_n > 0 and random.random() < 0.3:
                    _, best_point = gains[random.randint(0, top_n - 1)]
                else:
                    _, best_point = gains[0]
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            for comb in itertools.combinations(block, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    uncovered.discard(cid)
        
        return blocks
    
    def reduce_blocks_aggressive(blocks):
        """Aggressive reduction: repeatedly remove redundant blocks."""
        if not blocks:
            return blocks
        
        coverage_count = np.zeros(N_t, dtype=np.int32)
        block_to_tsubs = []
        
        for b in blocks:
            tsubs = []
            for comb in itertools.combinations(b, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    tsubs.append(cid)
                    coverage_count[cid] += 1
            block_to_tsubs.append(tsubs)
        
        reduced = list(blocks)
        reduced_tsubs = list(block_to_tsubs)
        changed = True
        
        while changed:
            changed = False
            for i in range(len(reduced)):
                is_redundant = True
                for cid in reduced_tsubs[i]:
                    if coverage_count[cid] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for cid in reduced_tsubs[i]:
                        coverage_count[cid] -= 1
                    reduced.pop(i)
                    reduced_tsubs.pop(i)
                    changed = True
                    break
        
        return reduced
    
    def local_search_polish(blocks, time_limit=5.0):
        """Try to replace pairs of blocks with a single better block."""
        if not blocks or len(blocks) < 2:
            return blocks
        
        start = time.time()
        blocks = list(blocks)
        
        # Build coverage count
        coverage_count = np.zeros(N_t, dtype=np.int32)
        block_to_tsubs = []
        for b in blocks:
            tsubs = []
            for comb in itertools.combinations(b, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    tsubs.append(cid)
                    coverage_count[cid] += 1
            block_to_tsubs.append(tsubs)
        
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            for i in range(len(blocks)):
                if time.time() - start > time_limit:
                    break
                # Check if removing block i leaves any uncovered t-subsets
                unique_to_i = []
                for cid in block_to_tsubs[i]:
                    if coverage_count[cid] == 1:
                        unique_to_i.append(cid)
                
                if not unique_to_i:
                    # Block i is redundant
                    for cid in block_to_tsubs[i]:
                        coverage_count[cid] -= 1
                    blocks.pop(i)
                    block_to_tsubs.pop(i)
                    improved = True
                    break
                elif len(unique_to_i) <= 1 and len(blocks) > 1:
                    # Try to find another block that can cover the unique t-subset(s)
                    # or find a replacement block
                    pass
        
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts with different strategies
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        seed = restart_count * 7919
        
        # Alternate between strategies
        if restart_count % 3 == 0:
            blocks = greedy_cover_random_start(seed=seed)
        else:
            blocks = greedy_cover_best_superset(seed=seed)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Try reduction
        if time.time() < deadline - 8.0:
            reduced = reduce_blocks_aggressive(best_blocks)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
    
    # Final polish with local search
    if best_blocks is not None and time.time() < deadline - 3.0:
        polished = local_search_polish(best_blocks, time_limit=deadline - time.time() - 2.0)
        if len(polished) < best_count:
            best_count = len(polished)
            best_blocks = polished
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(N_t))
        blocks = []
        while uncovered:
            start_id = min(uncovered)
            start_subset = all_t_subsets[start_id]
            block = list(start_subset)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        cid = subset_to_id.get(cand)
                        if cid is not None and cid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            for comb in itertools.combinations(block, t):
                cid = subset_to_id.get(comb)
                if cid is not None:
                    uncovered.discard(cid)
        
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
