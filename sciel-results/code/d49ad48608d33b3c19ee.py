"""
Covering design C(19,7,4) via randomized greedy with multi-start and local repair.
"""
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    """
    deadline = time.time() + 140  # leave 10s margin before 150s kill

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    total_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    def greedy_cover(seed, max_blocks=None):
        """
        Greedy covering with randomization.
        Returns (blocks, uncovered_count) or None if timeout.
        """
        rng = random.Random(seed)
        uncovered = set(range(total_tsubs))  # indices of uncovered t-subsets
        blocks = []
        
        if max_blocks is None:
            max_blocks = 200  # safe upper bound
        
        for _ in range(max_blocks):
            if time.time() >= deadline:
                return None
            if not uncovered:
                break
            
            # Pick a random uncovered t-subset to start a new block
            target = next(iter(uncovered))  # first uncovered (deterministic within seed context)
            # Actually, let's pick randomly among uncovered for diversity
            target = rng.choice(list(uncovered))
            
            start_subset = all_tsubs[target]
            block = list(start_subset)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_gain = -1
                best_points = []
                for x in all_points:
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in tsub_to_idx:
                            idx = tsub_to_idx[candidate]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain and gain > 0:
                        best_points.append(x)
                
                if not best_points or best_gain <= 0:
                    # No improvement possible; pick a random point
                    available = [x for x in all_points if x not in block]
                    if available:
                        block.append(rng.choice(available))
                    else:
                        break
                else:
                    block.append(rng.choice(best_points))
            
            block.sort()
            blocks.append(tuple(block))
            
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
        
        return blocks, len(uncovered)

    def local_search(blocks, uncovered_set, seed):
        """
        Try to improve by removing redundant blocks or swapping points.
        Returns improved (blocks, uncovered_count) or None if timeout.
        """
        rng = random.Random(seed)
        blocks = list(blocks)
        uncovered = set(uncovered_set)
        
        # Precompute coverage: for each block, which t-subsets it covers
        def compute_block_coverage(block):
            cov = set()
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    cov.add(idx)
            return cov
        
        # Try removing redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            block_covers = [compute_block_coverage(b) for b in blocks]
            # For each block, check if all its covered t-subsets are also covered by other blocks
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() >= deadline:
                    return None
                # Union of all other blocks' coverage
                other_cov = set()
                for j in range(len(blocks)):
                    if j != i:
                        other_cov.update(block_covers[j])
                if block_covers[i].issubset(other_cov):
                    # Block i is redundant
                    blocks.pop(i)
                    block_covers.pop(i)
                    changed = True
                    break
        
        return blocks, len(uncovered)

    best_blocks = None
    best_uncovered = total_tsubs + 1
    
    num_starts = 50
    for seed in range(num_starts):
        if time.time() >= deadline - 5:
            break
        
        result = greedy_cover(seed)
        if result is None:
            continue
        blocks, unc = result
        if unc < best_uncovered:
            best_uncovered = unc
            best_blocks = blocks
        
        # Try local search on the best so far
        if best_blocks is not None and best_uncovered > 0:
            ls_result = local_search(best_blocks, set(uncovered), seed)
            if ls_result is not None:
                ls_blocks, ls_unc = ls_result
                if ls_unc < best_uncovered:
                    best_uncovered = ls_unc
                    best_blocks = ls_blocks
        
        if best_uncovered == 0:
            break
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        best_blocks = []
        uncovered = set(range(total_tsubs))
        while uncovered:
            target = min(uncovered)
            block = list(all_tsubs[target])
            while len(block) < k:
                best_gain = -1
                best_pt = None
                for x in all_points:
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(candidate)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = x
                if best_pt is None:
                    avail = [x for x in all_points if x not in block]
                    if avail:
                        best_pt = avail[0]
                    else:
                        break
                block.append(best_pt)
            block.sort()
            best_blocks.append(tuple(block))
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
    
    return [list(b) for b in best_blocks]
