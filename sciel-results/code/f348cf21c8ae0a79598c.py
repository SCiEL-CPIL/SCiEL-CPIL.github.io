"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
from math import comb


def construct(v, k, t):
    deadline = time.time() + 130.0
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    uncovered = set(range(n_tsubs))
    blocks = []
    
    # Precompute for each point x, the list of t-subset indices containing x
    point_to_tsubs = [[] for _ in range(v)]
    for idx, s in enumerate(all_tsubs):
        for x in s:
            point_to_tsubs[x].append(idx)
    
    # Precompute for each pair of points (a,b), the list of t-subset indices containing both
    # Only needed if t >= 2
    if t >= 2:
        pair_to_tsubs = {}
        for idx, s in enumerate(all_tsubs):
            for a, b in itertools.combinations(s, 2):
                key = (min(a,b), max(a,b))
                if key not in pair_to_tsubs:
                    pair_to_tsubs[key] = []
                pair_to_tsubs[key].append(idx)
    else:
        pair_to_tsubs = {}
    
    while uncovered and time.time() < deadline:
        # Pick the t-subset with the highest "degree" (most uncovered t-subsets it can help cover)
        # Actually, pick the t-subset that appears in the most potential blocks? 
        # Simpler: pick the uncovered t-subset with the smallest index (deterministic) 
        # OR pick one that maximizes something. Let's stick to picking the first uncovered for simplicity
        # but try to pick a "central" one if possible. For now, first uncovered.
        seed_idx = min(uncovered)
        seed = list(all_tsubs[seed_idx])
        block = set(seed)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in all_points:
                if x in block:
                    continue
                # Calculate gain: how many new uncovered t-subsets would be covered by adding x
                # A t-subset is covered by block+{x} if it contains x and (t-1) elements from block
                gain = 0
                # Iterate over all (t-1)-subsets of current block
                for rest in itertools.combinations(block, t - 1):
                    # Check if rest + (x,) forms a t-subset in uncovered
                    # Since block elements are distinct and x not in block, sorted(rest+(x,)) is the t-subset
                    # We need to check if this t-subset is in uncovered
                    try:
                        idx = tsub_to_idx[tuple(sorted(rest + (x,)))]
                        if idx in uncovered:
                            gain += 1
                    except KeyError:
                        pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.add(best_x)
        
        block = sorted(block)
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if idx in uncovered:
                uncovered.remove(idx)
        
        if not uncovered:
            break
            
    # If we ran out of time, we might have uncovered subsets.
    # The task requires a valid covering. If we don't have one, we must complete it.
    # But the deadline is strict. If we can't finish, we should return what we have?
    # No, "Valid when every t-subset ... lies in at least one block".
    # So we MUST return a valid covering.
    # If we hit the deadline, we should finish the greedy process quickly?
    # Or just let it run? The evaluator kills after 150s.
    # If we return an invalid covering, score is 0.
    # So we should ensure we always return a valid covering.
    # The loop above stops at deadline. If uncovered is not empty, we need to cover it.
    # Let's continue the greedy process without the time check until it's done,
    # hoping it finishes. If it doesn't, it will be killed, scoring 0.
    # To be safe, if we are close to deadline and uncovered is large, 
    # we should maybe just return the blocks we have? No, invalid.
    # Better to just let the greedy finish. The greedy is fast enough for most cases.
    # The 130s deadline is to avoid the 150s kill if the greedy is slow.
    # If greedy is slow, we are in trouble anyway.
    
    # Continue if uncovered remains
    while uncovered:
        seed_idx = min(uncovered)
        seed = list(all_tsubs[seed_idx])
        block = set(seed)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in all_points:
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    try:
                        idx = tsub_to_idx[tuple(sorted(rest + (x,)))]
                        if idx in uncovered:
                            gain += 1
                    except KeyError:
                        pass
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.add(best_x)
        
        block = sorted(block)
        blocks.append(block)
        
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if idx in uncovered:
                uncovered.remove(idx)
        
        if not uncovered:
            break

    return blocks
# EVOLVE-BLOCK-END
