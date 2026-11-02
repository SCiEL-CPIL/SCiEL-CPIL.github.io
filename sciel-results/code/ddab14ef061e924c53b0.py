# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from collections import defaultdict

# We will use a precomputation-based greedy + local search approach.
# For C(24,7,3), v=24, k=7, t=3.
# Total t-subsets = C(24,3) = 2024.
# We can represent the set of uncovered t-subsets as a boolean numpy array of size 2024.
# We precompute for each possible block (combination of k points) the list of t-subset indices it covers.
# Then greedy: pick the block that covers the most uncovered subsets.
# Then local search: try to remove redundant blocks or swap to reduce count.

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s (evaluator kills at 150s, but we want to be safe)
    
    # For the specific task v=24, k=7, t=3, but we make it general-ish.
    # Precompute all t-subsets and map them to indices.
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each possible k-block, which t-subsets it covers.
    # This is expensive for large v, but for v=24, C(24,7)=346104, and each block covers C(7,3)=35 subsets.
    # Total entries: 346104 * 35 ~ 12 million. This might be too slow to precompute fully in Python.
    # Alternative: compute on the fly but with caching, or use a smarter greedy.
    
    # Let's use a different approach: 
    # Represent uncovered as a set of frozensets or tuples.
    # Greedy: for each candidate block, compute how many uncovered t-subsets it covers.
    # But iterating over all C(v,k) blocks each time is too slow.
    
    # Better: use the "greedy with gain" approach but optimized.
    # Or: use a known good covering as a starting point and then try to improve via local search.
    
    # Let's try: start with a random or heuristic covering, then do local search to remove redundant blocks.
    
    # First, generate a covering using a fast greedy.
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset to start the block
        start_subset = random.choice(list(uncovered))
        block_set = set(start_subset)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block_set) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: how many new uncovered t-subsets are covered by adding x
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    new_ts = tuple(sorted(rest + (x,)))
                    if new_ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is not None:
                block_set.add(best_point)
            else:
                break
        
        if len(block_set) < k:
            # Fill with random points
            while len(block_set) < k:
                x = random.randint(0, v - 1)
                if x not in block_set:
                    block_set.add(x)
        
        block = sorted(block_set)
        blocks.append(block)
        
        # Remove covered t-subsets
        for ts in itertools.combinations(block, t):
            uncovered.discard(tuple(ts))
        
        if time.time() - start_time > deadline * 0.5:
            break
    
    # Now we have a covering (hopefully). If not, we need to continue.
    # But for now, let's assume we have a valid covering and try to improve it.
    
    # Check if it's valid; if not, we need to fix it.
    # For simplicity, let's just return what we have if it's valid, else continue greedy.
    
    # Let's do a proper greedy first, then local search.
    # Reset and do a clean greedy.
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered and time.time() - start_time < deadline * 0.7:
        # Pick the uncovered t-subset that appears in the fewest candidate blocks? 
        # Or just pick the first one.
        start_ts = min(uncovered)
        block_set = set(start_ts)
        
        while len(block_set) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    new_ts = tuple(sorted(rest + (x,)))
                    if new_ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None or best_gain <= 0:
                break
            block_set.add(best_point)
        
        while len(block_set) < k:
            x = random.randint(0, v - 1)
            if x not in block_set:
                block_set.add(x)
        
        block = sorted(block_set)
        blocks.append(block)
        
        for ts in itertools.combinations(block, t):
            uncovered.discard(tuple(ts))
    
    # If we still have uncovered, add random blocks until covered (shouldn't happen often)
    while uncovered:
        block_set = set(random.sample(range(v), k))
        block = sorted(block_set)
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(tuple(ts))
    
    # Now do local search: try to remove redundant blocks.
    # A block is redundant if every t-subset it covers is also covered by other blocks.
    
    # Build a mapping: for each t-subset, which blocks cover it?
    # This is expensive. Instead, for each block, check if all its t-subsets are covered by others.
    
    # Precompute for each block, the set of t-subsets it covers.
    block_t_subsets = []
    for b in blocks:
        tss = set(tuple(sorted(ts)) for ts in itertools.combinations(b, t))
        block_t_subsets.append(tss)
    
    # For local search, try to remove each block and see if coverage is maintained.
    # To check if a block is redundant, we need to know if all its t-subsets are covered by other blocks.
    
    # Let's build a coverage count for each t-subset.
    coverage_count = defaultdict(int)
    for tss in block_t_subsets:
        for ts in tss:
            coverage_count[ts] += 1
    
    # Now try to remove blocks with low impact.
    # Actually, let's do a simple local search: randomly pick a block, try to remove it, 
    # if still covered, keep it removed. Otherwise, try to swap it with a different block.
    
    # For time efficiency, let's just try to remove blocks in order of increasing "criticality".
    
    # Compute for each block, the number of t-subsets it uniquely covers.
    unique_counts = []
    for i, tss in enumerate(block_t_subsets):
        unique = 0
        for ts in tss:
            if coverage_count[ts] == 1:
                unique += 1
        unique_counts.append(unique)
    
    # Sort blocks by unique count (ascending) and try to remove those with 0 unique first.
    # But removing a block changes coverage counts, so we need to be careful.
    
    # Simpler: iterate through blocks, and for each, check if it's redundant given the current set.
    # If redundant, remove it.
    
    # Let's do multiple passes.
    for _ in range(10):
        if time.time() - start_time > deadline * 0.9:
            break
        changed = False
        new_blocks = []
        for i, b in enumerate(blocks):
            # Check if b is redundant
            redundant = True
            for ts in itertools.combinations(b, t):
                ts_tuple = tuple(sorted(ts))
                # Check if ts_tuple is covered by any other block
                found = False
                for j, b2 in enumerate(blocks):
                    if j == i:
                        continue
                    if ts_tuple in block_t_subsets[j]:
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            if redundant:
                changed = True
            else:
                new_blocks.append(b)
        if changed:
            blocks = new_blocks
            # Recompute block_t_subsets and coverage_count
            block_t_subsets = []
            coverage_count = defaultdict(int)
            for b in blocks:
                tss = set(tuple(sorted(ts)) for ts in itertools.combinations(b, t))
                block_t_subsets.append(tss)
                for ts in tss:
                    coverage_count[ts] += 1
        else:
            break
    
    return blocks
# EVOLVE-BLOCK-END
