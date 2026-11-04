"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def _build_block_gain_table(points, t):
    """For each point x, map from tuple of (t-1) points to count of uncovered t-subsets it covers.
    Actually we need: for a partial block B, gain(x) = number of uncovered t-subsets that contain x and (t-1) points from B.
    We precompute for each x and each combination of (t-1) points, how many uncovered t-subsets are formed.
    But uncovered changes dynamically, so this is tricky.
    
    Instead, let's use a different approach: maintain a count array for each t-subset.
    """
    pass

def construct(v, k, t):
    """
    Greedy covering design with local search improvements.
    """
    deadline = time.time() + 110  # 110s deadline to be safe within 120s
    
    # Initialize uncovered t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered_count = [1] * len(all_t_subsets)  # 1 if uncovered, 0 if covered
    uncovered_set = set(all_t_subsets)
    
    # Map from t-subset to index
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    blocks = []
    
    # Precompute: for each point x, list of t-subsets containing x
    t_subs_containing = [[] for _ in range(v)]
    for ts in all_t_subsets:
        for x in ts:
            t_subs_containing[x].append(ts)
    
    # Greedy phase
    while uncovered_set:
        if time.time() > deadline:
            break
        
        # Pick the first uncovered t-subset (or one with fewest options)
        # For speed, just pick the first
        first_uncovered = min(uncovered_set)
        block = list(first_uncovered)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            block_set = set(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x >= 0:
                block.append(best_x)
            else:
                # No point improves, pick random
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    block.append(random.choice(remaining))
                break
        
        if len(block) < k:
            break
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            if ts in uncovered_set:
                uncovered_set.discard(ts)
    
    # Local search phase: try to remove redundant blocks
    if time.time() < deadline - 10:
        # Sort blocks by "importance" - blocks that cover unique t-subsets
        # Try removing each block and see if coverage is still complete
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Count how many t-subsets are covered only by this block
            # For efficiency, check if removing block i leaves any t-subset uncovered
            # This is expensive, so do a quick check
            
            # Build coverage count for all t-subsets
            cover_count = defaultdict(int)
            for j, b in enumerate(blocks):
                if j == i:
                    continue
                for ts in itertools.combinations(b, t):
                    cover_count[ts] += 1
            
            # Check if any t-subset has cover_count 0
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                if cover_count[ts] == 0:
                    redundant = False
                    break
            
            if redundant:
                blocks.pop(i)
    
    # Local search phase: try replacing blocks
    if time.time() < deadline - 5:
        # Try to replace blocks with better ones
        for _ in range(50):  # Limit iterations
            if time.time() > deadline - 2:
                break
            
            # Pick a random block to try replacing
            if not blocks:
                break
            idx = random.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            
            # Try greedy reconstruction of this block
            # Remove it temporarily
            blocks.pop(idx)
            
            # Find t-subsets that were only covered by this block
            # Rebuild uncovered
            uncovered_set = set()
            cover_count = defaultdict(int)
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    cover_count[ts] += 1
            
            for ts in itertools.combinations(range(v), t):
                if cover_count[ts] == 0:
                    uncovered_set.add(ts)
            
            if not uncovered_set:
                # No need to replace
                blocks.insert(idx, old_block)
                continue
            
            # Greedily construct a new block
            first_uncovered = min(uncovered_set)
            new_block = list(first_uncovered)
            
            while len(new_block) < k:
                best_x = -1
                best_gain = -1
                new_block_set = set(new_block)
                
                for x in range(v):
                    if x in new_block_set:
                        continue
                    
                    gain = 0
                    for rest in itertools.combinations(new_block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered_set:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x >= 0:
                    new_block.append(best_x)
                else:
                    remaining = [x for x in range(v) if x not in new_block_set]
                    if remaining:
                        new_block.append(random.choice(remaining))
                    break
            
            if len(new_block) < k:
                blocks.insert(idx, old_block)
                continue
            
            new_block.sort()
            blocks.insert(idx, new_block)
            
            # Check if this replacement improved things (fewer blocks or same)
            # For now, just keep it if it's valid
    
    # Ensure validity: if we ran out of time, make sure we have a valid covering
    # by adding more blocks if needed
    if time.time() < deadline - 1:
        # Check coverage
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        
        all_t = set(itertools.combinations(range(v), t))
        missing = all_t - covered
        
        # Add blocks for missing t-subsets
        while missing and time.time() < deadline - 0.5:
            first = min(missing)
            block = list(first)
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in missing:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x >= 0:
                    block.append(best_x)
                    block_set.add(best_x)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        block.append(random.choice(remaining))
                        block_set.add(block[-1])
                    else:
                        break
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    missing.discard(ts)
    
    # Remove duplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks

# EVOLVE-BLOCK-END
