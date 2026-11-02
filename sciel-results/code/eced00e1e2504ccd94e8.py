"""Improved covering design solver using randomized greedy with local search."""
import time
import random
import itertools
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each point, precompute which t-subsets it participates in
    # point_tsubs[p] = list of t-subset indices containing p
    point_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsubs[p].append(i)
    
    # For each pair of points, precompute t-subsets containing both
    # This helps in gain calculation during block construction
    # Actually, let's use a different approach: for a partial block B of size r,
    # the gain of adding point x is the number of uncovered t-subsets 
    # that contain x and are contained in B ∪ {x}
    # A t-subset S is contained in B ∪ {x} iff S ⊆ B ∪ {x}
    # Since |S| = t, and we need x ∈ S, we need S ∩ B has size t-1 and x ∉ B
    # So gain(x) = number of uncovered t-subsets S such that x ∈ S and S\{x} ⊆ B
    
    # Let's track coverage as a bitset or boolean array
    # uncovered[i] = True if t-subset i is not yet covered
    
    # Greedy with randomization and local search
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = bytearray(num_tsubsets)  # 1 = uncovered
        for i in range(num_tsubsets):
            uncovered[i] = 1
        
        blocks = []
        remaining = num_tsubsets
        
        while remaining > 0:
            if time.time() > deadline:
                return None  # timeout
            
            # Pick a random uncovered t-subset to seed the block
            # Find an uncovered one
            idx = rng.randrange(num_tsubsets)
            while uncovered[idx] == 0:
                idx = (idx + 1) % num_tsubsets
            
            seed_tsub = all_tsubsets[idx]
            block = list(seed_tsub)
            rng.shuffle(block)
            
            # Greedily add points to maximize gain
            while len(block) < k:
                if time.time() > deadline:
                    return None
                
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets S 
                    # where x ∈ S and S\{x} ⊆ block
                    gain = 0
                    for ts_idx in point_tsubs[x]:
                        if uncovered[ts_idx] == 0:
                            continue
                        ts = all_tsubsets[ts_idx]
                        # Check if ts\{x} ⊆ block
                        ok = True
                        for p in ts:
                            if p != x and p not in block_set:
                                ok = False
                                break
                        if ok:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point != -1:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    # No useful point, just pick a random one
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_point = rng.choice(candidates)
                    else:
                        break
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            block_set = set(block)
            for ts_idx in range(num_tsubsets):
                if uncovered[ts_idx] == 0:
                    continue
                ts = all_tsubsets[ts_idx]
                if all(p in block_set for p in ts):
                    uncovered[ts_idx] = 0
                    remaining -= 1
        
        return blocks
    
    # Try multiple random seeds and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # First, try a deterministic greedy (seed 0)
    blocks = greedy_cover(0)
    if blocks is not None:
        best_blocks = blocks
        best_count = len(blocks)
    
    # Try more seeds with time remaining
    seed = 1
    while time.time() < deadline - 5:
        blocks = greedy_cover(seed)
        if blocks is not None and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    return best_blocks
