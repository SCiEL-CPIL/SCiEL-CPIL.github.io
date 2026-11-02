import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [set() for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for pt in ts:
            point_tsubs[pt].add(i)
    
    def time_left():
        return deadline - time.time()
    
    def greedy_cover(seed=0):
        """Greedy covering with smart block construction."""
        rng = random.Random(seed)
        uncovered = [True] * n_tsubs
        uncovered_count = n_tsubs
        blocks = []
        
        # For each point, count how many uncovered t-subsets it's in
        point_uncovered_count = [0] * v
        for i, ts in enumerate(all_tsubs):
            if uncovered[i]:
                for pt in ts:
                    point_uncovered_count[pt] += 1
        
        # Precompute for efficiency: for a partial block, which t-subsets would be covered
        # by adding a point
        def compute_block_coverage(block_set, candidate_points):
            """Return dict of point -> count of newly covered t-subsets."""
            gains = {}
            for pt in candidate_points:
                if pt in block_set:
                    continue
                # t-subsets that include pt and (t-1) elements from block_set
                # that are currently uncovered
                count = 0
                if len(block_set) >= t - 1:
                    for combo in itertools.combinations(sorted(block_set), t - 1):
                        ts = tuple(sorted(combo + (pt,)))
                        idx = tsub_to_idx.get(ts, -1)
                        if idx >= 0 and uncovered[idx]:
                            count += 1
                gains[pt] = count
            return gains
        
        while uncovered_count > 0 and time_left() > 10:
            # Find the hardest uncovered t-subset (one with fewest possible completions)
            # Or just pick randomly from uncovered for diversity
            uncovered_indices = [i for i in range(n_tsubs) if uncovered[i]]
            if not uncovered_indices:
                break
            
            # Pick a random uncovered t-subset to anchor the block
            anchor_idx = rng.choice(uncovered_indices)
            anchor = all_tsubs[anchor_idx]
            
            # Build block starting with anchor, greedily adding points
            block_set = set(anchor)
            remaining = [pt for pt in range(v) if pt not in block_set]
            
            while len(block_set) < k and remaining:
                # Compute gains for each candidate point
                gains = compute_block_coverage(block_set, remaining)
                
                # Pick the point with max gain, break ties randomly
                max_gain = max(gains.values()) if gains else 0
                if max_gain == 0:
                    # No gain, pick randomly
                    pt = rng.choice(remaining)
                else:
                    best_points = [pt for pt, g in gains.items() if g == max_gain]
                    pt = rng.choice(best_points)
                
                block_set.add(pt)
                remaining.remove(pt)
                
                # Update uncovered
                for ts in itertools.combinations(sorted(block_set), t):
                    idx = tsub_to_idx.get(ts, -1)
                    if idx >= 0 and uncovered[idx]:
                        uncovered[idx] = False
                        uncovered_count -= 1
                        for p in ts:
                            point_uncovered_count[p] -= 1
                        break  # Only update once per ts
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Final update of uncovered for the complete block
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts, -1)
                if idx >= 0 and uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    for p in ts:
                        point_uncovered_count[p] -= 1
        
        return blocks, uncovered_count
    
    def local_search(blocks, max_iterations=50):
        """Try to improve by swapping points in blocks."""
        # Count coverage of each t-subset
        tsub_coverage = [0] * n_tsubs
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts, -1)
                if idx >= 0:
                    tsub_coverage[idx] += 1
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time_left() > 5:
            improved = False
            iteration += 1
            for bi, block in enumerate(blocks):
                if time_left() < 5:
                    break
                block_set = set(block)
                for ai in block:
                    for bi2 in range(v):
                        if bi2 in block_set or bi2 == ai:
                            continue
                        # Try swapping ai with bi2
                        new_block = [bi2 if x == ai else x for x in block]
                        new_block.sort()
                        
                        # Check if this improves coverage
                        # Count how many t-subsets we lose vs gain
                        lost = 0
                        gained = 0
                        for ts in itertools.combinations(block, t):
                            idx = tsub_to_idx.get(ts, -1)
                            if idx >= 0 and tsub_coverage[idx] == 1:
                                lost += 1
                        for ts in itertools.combinations(new_block, t):
                            idx = tsub_to_idx.get(ts, -1)
                            if idx >= 0 and tsub_coverage[idx] == 0:
                                gained += 1
                        
                        if gained > lost:
                            # Apply swap
                            blocks[bi] = new_block
                            # Update coverage
                            for ts in itertools.combinations(block, t):
                                idx = tsub_to_idx.get(ts, -1)
                                if idx >= 0:
                                    tsub_coverage[idx] -= 1
                            for ts in itertools.combinations(new_block, t):
                                idx = tsub_to_idx.get(ts, -1)
                                if idx >= 0:
                                    tsub_coverage[idx] += 1
                            improved = True
                            break
                    if improved:
                        break
        
        return blocks
    
    # Run multiple greedy passes with different seeds
    best_blocks = None
    best_uncovered = n_tsubs
    
    seeds = [0, 1, 42, 123, 999, 2024, 7, 13, 100, 555]
    for seed in seeds:
        if time_left() < 30:
            break
        blocks, uncovered_count = greedy_cover(seed)
        if uncovered_count == 0:
            # Complete covering, try to improve
            blocks = local_search(blocks)
            if best_blocks is None or len(blocks) < len(best_blocks):
                best_blocks = blocks
                best_uncovered = 0
        elif best_uncovered > uncovered_count:
            best_uncovered = uncovered_count
            best_blocks = blocks
    
    # If we have a complete covering, try more local search
    if best_uncovered == 0 and time_left() > 20:
        best_blocks = local_search(best_blocks, max_iterations=100)
    
    # Final verification: ensure all t-subsets are covered
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered_set = set(all_tsubs)
        blocks = []
        while uncovered_set:
            ts = min(uncovered_set)
            block = list(ts)
            while len(block) < k:
                best_pt = -1
                best_gain = -1
                for pt in range(v):
                    if pt in block:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(combo + (pt,)))
                        if new_ts in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = pt
                if best_pt == -1:
                    break
                block.append(best_pt)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered_set.discard(tuple(ts))
        best_blocks = blocks
    
    # Verify and return
    covered = [False] * n_tsubs
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts, -1)
            if idx >= 0:
                covered[idx] = True
    
    # If not fully covered, do a final greedy pass to cover remaining
    if not all(covered):
        uncovered_remaining = [all_tsubs[i] for i in range(n_tsubs) if not covered[i]]
        for ts in uncovered_remaining:
            if time_left() < 5:
                break
            # Find a block that can cover this
            block = list(ts)
            while len(block) < k:
                best_pt = -1
                best_gain = -1
                for pt in range(v):
                    if pt in block:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(combo + (pt,)))
                        if new_ts in uncovered_remaining:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = pt
                if best_pt == -1:
                    # Just pick any point not in block
                    for pt in range(v):
                        if pt not in block:
                            best_pt = pt
                            break
                if best_pt == -1:
                    break
                block.append(best_pt)
            block.sort()
            best_blocks.append(block)
    
    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
