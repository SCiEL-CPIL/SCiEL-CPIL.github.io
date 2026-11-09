"""Improved covering design C(19,7,4) with time-bounded search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets it participates in
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for pt in ts:
            point_tsubs[pt].append(i)
    
    # Greedy covering with randomization and local search
    def greedy_cover(seed=0, use_random=False):
        rng = random.Random(seed)
        uncovered = set(range(n_tsubs))  # set of uncovered t-subset indices
        blocks = []
        
        while uncovered:
            # Pick a starting t-subset
            if use_random and len(uncovered) > 1:
                # Weighted random selection: prefer t-subsets that appear in fewer potential blocks
                # Simpler: just pick randomly
                start_idx = rng.choice(list(uncovered))
            else:
                start_idx = min(uncovered)
            
            start_tsub = all_tsubs[start_idx]
            block = list(start_tsub)
            
            # Add points greedily to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_gain = -1
                best_pt = -1
                candidates = [pt for pt in range(v) if pt not in block]
                
                if use_random:
                    # Evaluate a random subset of candidates if too many
                    if len(candidates) > 10:
                        candidates = rng.sample(candidates, 10)
                
                for pt in candidates:
                    # Count how many uncovered t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (pt,)))
                        idx = tsub_to_idx.get(ts, -1)
                        if idx >= 0 and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = pt
                
                if best_pt == -1 or best_gain == 0:
                    # No improvement; just add a random point
                    if use_random:
                        best_pt = rng.choice(candidates)
                    else:
                        best_pt = candidates[0]
                
                block.append(best_pt)
                block.sort()
            
            # Add this block and update uncovered
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts, -1)
                if idx >= 0:
                    uncovered.discard(idx)
            
            # Check deadline
            if time.time() > deadline:
                break
        
        return blocks, len(uncovered) > 0

    # First pass: deterministic greedy
    blocks, incomplete = greedy_cover(seed=0, use_random=False)
    
    if not incomplete:
        return blocks
    
    # If incomplete due to time, try to complete with a second greedy pass
    # Count currently covered t-subsets
    covered_set = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered_set.add(ts)
    
    uncovered = set(all_tsubs) - covered_set
    
    # Continue greedy from where we left off
    while uncovered and time.time() < deadline:
        start_tsub = min(uncovered)
        block = list(start_tsub)
        
        while len(block) < k:
            best_gain = -1
            best_pt = -1
            for pt in range(v):
                if pt in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (pt,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pt = pt
            
            if best_pt == -1:
                best_pt = next(pt for pt in range(v) if pt not in block)
            
            block.append(best_pt)
            block.sort()
        
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    
    return blocks

# EVOLVE-BLOCK-END
