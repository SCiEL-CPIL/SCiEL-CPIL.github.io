"""Probabilistic construction via Johnson-scheme-inspired random blocks + targeted local repair."""
import itertools
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using:
    1. Random block generation (Johnson-scheme-inspired) to get a good initial cover.
    2. Targeted local search to repair uncovered t-subsets by optimizing block overlap.
    
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1).
    """
    # Generate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Coverage array: count how many times each t-subset is covered
    covered = np.zeros(n_tsubs, dtype=np.int32)
    
    def get_tsubs_of_block(block):
        """Get indices of all t-subsets contained in a block."""
        block_set = set(block)
        idxs = []
        for ts in itertools.combinations(sorted(block), t):
            idxs.append(tsub_to_idx[ts])
        return idxs
    
    def add_block(block):
        """Add a block to the covering and update coverage counts."""
        for idx in get_tsubs_of_block(block):
            covered[idx] += 1
    
    def remove_block(block):
        """Remove a block from the covering and update coverage counts."""
        for idx in get_tsubs_of_block(block):
            covered[idx] -= 1
    
    def uncovered_count():
        """Return number of uncovered t-subsets."""
        return int(np.sum(covered == 0))
    
    # Phase 1: Greedy random construction with Johnson-scheme-inspired diversity
    # We use a combination of random sampling and greedy selection
    blocks = []
    random.seed(42)  # Fixed seed for reproducibility
    
    # First, do a greedy pass to get a reasonable starting point
    uncovered_set = set(range(n_tsubs))
    
    # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    # But to add diversity, we sample from top candidates
    for _ in range(200):  # Limit initial greedy blocks
        if not uncovered_set:
            break
        
        best_gain = 0
        best_block = None
        candidates = []
        
        # Sample random blocks to find good candidates
        for _ in range(50):
            block = random.sample(range(v), k)
            block_set = set(block)
            gain = 0
            for ts in itertools.combinations(sorted(block), t):
                idx = tsub_to_idx[ts]
                if idx in uncovered_set:
                    gain += 1
            if gain > 0:
                candidates.append((gain, block))
        
        if candidates:
            # Pick from top candidates with some randomness
            candidates.sort(key=lambda x: -x[0])
            top_n = min(5, len(candidates))
            _, block = random.choice(candidates[:top_n])
            blocks.append(tuple(sorted(block)))
            add_block(list(block))
            # Update uncovered set
            for idx in get_tsubs_of_block(list(block)):
                if covered[idx] == 1:
                    uncovered_set.discard(idx)
        
        if len(blocks) > 200:
            break
    
    # Phase 2: Continue greedy until all covered or we have enough blocks
    while uncovered_set and len(blocks) < 1000:
        best_gain = 0
        best_block = None
        
        # Try several random blocks
        for _ in range(30):
            block = random.sample(range(v), k)
            block_set = set(block)
            gain = 0
            for ts in itertools.combinations(sorted(block), t):
                idx = tsub_to_idx[ts]
                if idx in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = tuple(sorted(block))
        
        if best_block is None or best_gain == 0:
            # No improvement found with random sampling; try a more targeted approach
            # Pick an uncovered t-subset and build a block around it
            ts = next(iter(uncovered_set))
            base = all_tsubs[ts]
            remaining = list(set(range(v)) - set(base))
            random.shuffle(remaining)
            block = list(base) + remaining[:k-t]
            block = tuple(sorted(block))
            blocks.append(block)
            add_block(list(block))
            for idx in get_tsubs_of_block(list(block)):
                if covered[idx] == 1:
                    uncovered_set.discard(idx)
        else:
            blocks.append(best_block)
            add_block(list(best_block))
            for idx in get_tsubs_of_block(list(best_block)):
                if covered[idx] == 1:
                    uncovered_set.discard(idx)
    
    # Phase 3: Local search to reduce block count
    # Try to remove redundant blocks and replace poorly-overlapping blocks
    current_count = len(blocks)
    
    # First, try removing redundant blocks (blocks that don't cover any uniquely covered t-subset)
    changed = True
    while changed and len(blocks) > 50:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            block = list(blocks[i])
            # Check if removing this block leaves all t-subsets covered
            remove_block(block)
            if uncovered_count() == 0:
                # Safe to remove
                blocks.pop(i)
                changed = True
                break
            else:
                # Restore
                add_block(block)
    
    # Second, try replacing blocks: for each block, try to find a replacement that covers
    # at least the same t-subsets but with less overlap with other blocks
    for _ in range(50):  # Iterations
        improved = False
        for i in range(len(blocks)):
            old_block = list(blocks[i])
            # Find t-subsets that are only covered by this block
            remove_block(old_block)
            uniquely_covered = [idx for idx in get_tsubs_of_block(old_block) if covered[idx] == 0]
            add_block(old_block)
            
            if not uniquely_covered:
                # This block is redundant; try to remove it
                remove_block(old_block)
                if uncovered_count() == 0:
                    blocks.pop(i)
                    improved = True
                    break
                add_block(old_block)
            
            # Try to find a replacement block that covers all uniquely_covered t-subsets
            # and has minimal overlap
            if uniquely_covered:
                # Get the points that must be in the new block
                required_points = set()
                for idx in uniquely_covered:
                    required_points.update(all_tsubs[idx])
                
                if len(required_points) > k:
                    continue
                
                # Try random replacements
                best_new_block = None
                best_overlap = float('inf')
                
                for _ in range(20):
                    new_block = list(required_points)
                    remaining = list(set(range(v)) - set(new_block))
                    random.shuffle(remaining)
                    new_block += remaining[:k - len(new_block)]
                    new_block = tuple(sorted(new_block))
                    
                    # Calculate overlap: how many t-subsets of new_block are already covered by other blocks
                    remove_block(old_block)
                    overlap = 0
                    for idx in get_tsubs_of_block(list(new_block)):
                        if covered[idx] > 0:
                            overlap += 1
                    add_block(old_block)
                    
                    # Check if new_block covers all uniquely_covered
                    new_block_set = set(new_block)
                    covers_all = True
                    for idx in uniquely_covered:
                        if not set(all_tsubs[idx]).issubset(new_block_set):
                            covers_all = False
                            break
                    
                    if covers_all and overlap < best_overlap:
                        best_overlap = overlap
                        best_new_block = new_block
                
                if best_new_block and best_new_block != tuple(sorted(old_block)):
                    # Replace
                    remove_block(old_block)
                    blocks[i] = best_new_block
                    add_block(list(best_new_block))
                    improved = True
        
        if not improved:
            break
    
    # Final: ensure all t-subsets are covered
    # If any remain uncovered, add blocks greedily
    while uncovered_count() > 0:
        # Find an uncovered t-subset
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        idx = uncovered_indices[0]
        ts = all_tsubs[idx]
        
        # Build a block around this t-subset
        base = list(ts)
        remaining = list(set(range(v)) - set(base))
        random.shuffle(remaining)
        block = tuple(sorted(base + remaining[:k-t]))
        blocks.append(block)
        add_block(list(block))
    
    # Convert to list of lists
    return [list(b) for b in blocks]
