"""Rewritten covering-design solver: fast greedy + local improvement with deadline guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter

def _time_budget():
    return 115.0  # safety margin before 120s kill

def construct(v, k, t):
    t0 = time.time()
    budget = _time_budget()
    deadline = t0 + budget

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    total_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Track coverage: cover_count[i] = number of blocks covering t-subset i
    cover_count = [0] * total_tsubs

    # Greedy construction
    blocks = []
    uncovered_count = total_tsubs

    # For speed, maintain a list of uncovered t-subsets (indices)
    uncovered = set(range(total_tsubs))

    while uncovered and (time.time() - t0) < deadline - 5:
        # Pick the least-covered uncovered t-subset to start a new block
        # For speed, just pick a random one from uncovered (or the first)
        # Better: pick one with minimum cover_count to prioritize hard-to-cover triples
        # But for initial greedy, just pick the first uncovered
        start_ts_idx = min(uncovered, key=lambda i: cover_count[i])
        start_ts = all_tsubs[start_ts_idx]
        
        # Build block greedily: start with points from start_ts, then add points
        # that cover the most uncovered t-subsets
        block = list(start_ts)
        remaining_k = k - t

        # For each additional point, pick the one that covers the most uncovered t-subsets
        for _ in range(remaining_k):
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset {a1,...,at} is covered if it contains x and t-1 points from block
                gain = 0
                # For each (t-1)-subset of block, check if adding x forms a t-subset in uncovered
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(candidate)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                # Fallback: pick any point not in block
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if idx in uncovered:
                cover_count[idx] += 1
                uncovered.discard(idx)
            else:
                cover_count[idx] += 1
            # If cover_count was 0 and now >0, it's no longer uncovered
            # But we already handled that above

        # Remove from uncovered all t-subsets covered by this block
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if idx in uncovered:
                uncovered.discard(idx)

    # Local improvement: try to remove redundant blocks
    # A block is redundant if removing it doesn't uncover any t-subset
    # i.e., every t-subset in the block is covered by at least one other block
    improved = True
    while improved and (time.time() - t0) < deadline - 5:
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if (time.time() - t0) > deadline - 5:
                break
            # Check if block i is redundant
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                idx = tsub_to_idx[ts]
                # Count how many blocks (excluding i) cover this t-subset
                count = 0
                for j, b in enumerate(blocks):
                    if j == i:
                        continue
                    # Check if ts is subset of b
                    if all(x in b for x in ts):
                        count += 1
                        if count >= 1:
                            break
                if count == 0:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                improved = True

    # Second pass: try to replace pairs of blocks with a single better block
    # This is more complex; skip for now due to time constraints

    # Ensure we don't exceed deadline
    if (time.time() - t0) > deadline - 2:
        # Return whatever we have
        pass

    # Final cleanup: remove duplicates (shouldn't be any) and sort
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
# EVOLVE-BLOCK-END
