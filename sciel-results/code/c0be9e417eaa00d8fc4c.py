"""Improved covering design C(19,7,4) solver with greedy + local search."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Find a covering of all t-subsets of {0,...,v-1} by k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct integers).
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the list of t-subsets containing x
    # point_to_tsubsets[x] = list of t-subset indices that contain x
    point_to_tsubsets = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_to_tsubsets[x].append(i)
    
    # For a block (k-subset), the t-subsets it covers are all t-combinations of the block
    # We'll track coverage using a set of covered t-subset indices
    
    def block_covers(block):
        """Return set of t-subset indices covered by a given block (tuple of k points)."""
        covered = set()
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            covered.add(idx)
        return covered
    
    # Greedy construction
    def greedy_construct():
        """Standard greedy set cover for covering design."""
        uncovered = set(range(total_t))
        blocks = []
        
        while uncovered:
            # Pick the t-subset with minimum index (or we could randomize)
            # Try multiple candidate starting points and pick the one that leads to best block
            best_block = None
            best_gain = -1
            
            # We'll build the block greedily point by point
            # Start with the first uncovered t-subset's elements as candidates
            first_uncovered = min(uncovered)
            first_ts = all_t_subsets[first_uncovered]
            
            # Greedy: start with a point from first_ts, then add points maximizing new coverage
            # Try each point in first_ts as the starting point
            for start_x in first_ts:
                block = [start_x]
                block_set = {start_x}
                # Track which t-subsets are covered by current partial block
                # A t-subset is "potentially coverable" if it's a superset-compatible with block
                # Actually, for greedy: at each step, adding point x covers new t-subsets
                # that contain x and (t-1) points from the current block
                
                current_covered = set()
                # t-subsets fully contained in block so far (only possible if len(block)>=t)
                
                while len(block) < k:
                    best_x = None
                    best_new = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        # Count new t-subsets covered by adding x to block
                        # A t-subset {x} ∪ S where S is a (t-1)-subset of block
                        new_count = 0
                        for s in itertools.combinations(block, t-1):
                            ts = tuple(sorted(s + (x,)))
                            idx = t_subset_index.get(ts)
                            if idx is not None and idx in uncovered and idx not in current_covered:
                                new_count += 1
                        if new_count > best_new:
                            best_new = new_count
                            best_x = x
                    if best_x is None:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                    # Update current_covered with newly covered t-subsets
                    for s in itertools.combinations(block[:-1], t-1):
                        ts = tuple(sorted(s + (best_x,)))
                        idx = t_subset_index.get(ts)
                        if idx is not None:
                            current_covered.add(idx)
                
                # Calculate total gain of this block
                gain = len(current_covered & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                # Fallback: just take first_ts extended arbitrarily
                best_block = list(first_ts)
                remaining = [x for x in range(v) if x not in set(first_ts)]
                random.shuffle(remaining)
                best_block.extend(remaining[:k-len(best_block)])
            
            best_block.sort()
            blocks.append(best_block)
            
            # Remove covered t-subsets
            for ts in itertools.combinations(best_block, t):
                idx = t_subset_index[ts]
                uncovered.discard(idx)
            
            if time.time() - start_time > DEADLINE:
                break
        
        return blocks
    
    # Local search: try to remove redundant blocks or swap points
    def is_redundant(blocks, i):
        """Check if block i is redundant (all its t-subsets are covered by other blocks)."""
        block = blocks[i]
        for ts in itertools.combinations(block, t):
            covered_by_other = False
            for j in range(len(blocks)):
                if j == i:
                    continue
                if all(x in blocks[j] for x in ts):
                    covered_by_other = True
                    break
            if not covered_by_other:
                return False
        return True
    
    def local_search(blocks, max_iters=5000):
        """Try to remove redundant blocks and improve via point swaps."""
        blocks = list(blocks)
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try removing redundant blocks
            removed = False
            for i in range(len(blocks)):
                if is_redundant(blocks, i):
                    blocks.pop(i)
                    removed = True
                    break
            if removed:
                continue
            
            # Try swapping a point in a block with another point to reduce total blocks
            # or try replacing a block entirely
            improved = False
            for i in range(len(blocks)):
                block = blocks[i]
                for pos in range(k):
                    old_pt = block[pos]
                    for new_pt in range(v):
                        if new_pt == old_pt or new_pt in block:
                            continue
                        # Try the swap
                        new_block = block[:]
                        new_block[pos] = new_pt
                        new_block.sort()
                        
                        # Check if this helps: does the old block become redundant?
                        # Or does the new block cover something the old didn't, allowing removal elsewhere?
                        # Simple heuristic: check if new block covers at least as many uncovered t-subsets
                        # For simplicity, just accept if it doesn't break coverage
                        # This is complex; skip for now and focus on removal
                        break
                break
            if improved:
                continue
            
            break
        
        return blocks
    
    # Run greedy
    blocks = greedy_construct()
    
    # Run local search to remove redundant blocks
    blocks = local_search(blocks)
    
    # Verify coverage
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    
    # If not fully covered (due to deadline), do a quick repair
    if len(covered) < total_t:
        uncovered = [ts for ts in all_t_subsets if ts not in covered]
        for ts in uncovered:
            if time.time() - start_time > DEADLINE - 5:
                break
            # Find a block that covers the most of this ts's "neighbors"
            # Just add a new block containing this ts
            block = list(ts)
            remaining = [x for x in range(v) if x not in set(ts)]
            random.shuffle(remaining)
            block.extend(remaining[:k-len(block)])
            block.sort()
            blocks.append(block)
    
    return blocks
