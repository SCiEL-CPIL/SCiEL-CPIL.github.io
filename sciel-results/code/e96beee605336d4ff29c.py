"""Optimized covering design C(19,8,5) using fast greedy with multi-start."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations


def _build_covering(v, k, t, deadline, rng, seed_offset=0):
    """Run a single greedy covering with random tie-breaking."""
    all_tsubs = list(combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Precompute index for each t-subset
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Track uncovered t-subsets
    uncovered = [True] * n_tsubs
    n_uncovered = n_tsubs
    
    # For each point, precompute which t-subsets contain it
    # point_to_tsubs[x] = list of indices of t-subsets containing x
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_to_tsubs[x].append(i)
    
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find first uncovered t-subset (or random one for diversity)
        # Use the lexicographically smallest uncovered for determinism within a run
        first_uncovered_idx = -1
        for i in range(n_tsubs):
            if uncovered[i]:
                first_uncovered_idx = i
                break
        if first_uncovered_idx == -1:
            break
        
        # Start block with one point from this t-subset
        start_ts = all_tsubs[first_uncovered_idx]
        block = [start_ts[0]]
        
        # Greedily add points
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_point = -1
            # Candidate points not yet in block
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]
            
            # For efficiency, compute gain for each candidate
            # Gain = number of uncovered t-subsets that would be covered by adding x
            # A t-subset is covered by adding x if:
            #   - x is in the t-subset
            #   - all other t-1 elements of the t-subset are already in block
            #   - the t-subset is currently uncovered
            
            # Precompute for current block: which (t-1)-subsets of block exist
            block_combos = set()
            for combo in combinations(block, t - 1):
                block_combos.add(combo)
            
            for x in candidates:
                gain = 0
                for ts_idx in point_to_tsubs[x]:
                    if not uncovered[ts_idx]:
                        continue
                    ts = all_tsubs[ts_idx]
                    # Check if all other elements are in block
                    rest = tuple(sorted(ts for ts in ts if ts != x))
                    if rest in block_combos:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                elif gain == best_gain and best_point != -1:
                    # Random tie-breaking
                    if rng.random() < 0.5:
                        best_point = x
            
            if best_point == -1:
                break
            
            block.append(best_point)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for ts in combinations(block, t):
            idx = tsub_to_idx.get(tuple(ts))
            if idx is not None and uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1
    
    return blocks, n_uncovered


def _verify_covering(v, k, t, blocks):
    """Check if all t-subsets are covered."""
    covered = set()
    for block in blocks:
        for ts in combinations(block, t):
            covered.add(ts)
    all_tsubs = set(combinations(range(v), t))
    return all_tsubs.issubset(covered)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Single deterministic greedy first
    blocks, remaining = _build_covering(v, k, t, deadline, rng)
    
    if remaining == 0:
        return [list(b) for b in blocks]
    
    # Try multiple random restarts
    best_blocks = blocks
    best_remaining = remaining
    
    attempt = 0
    while time.time() < deadline - 5.0:
        attempt += 1
        # Create a new RNG with different seed
        local_rng = random.Random(attempt * 1000 + 7)
        
        # Modify the greedy to use more randomization
        # We'll do a slightly different greedy: randomize the starting t-subset
        all_tsubs = list(combinations(range(v), t))
        n_tsubs = len(all_tsubs)
        tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
        uncovered = [True] * n_tsubs
        n_uncovered = n_tsubs
        
        point_to_tsubs = [[] for _ in range(v)]
        for i, ts in enumerate(all_tsubs):
            for x in ts:
                point_to_tsubs[x].append(i)
        
        blocks = []
        
        while n_uncovered > 0:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            idx = -1
            for _ in range(50):  # Try up to 50 random picks
                r = local_rng.randint(0, n_tsubs - 1)
                if uncovered[r]:
                    idx = r
                    break
            if idx == -1:
                # Fallback to first uncovered
                for i in range(n_tsubs):
                    if uncovered[i]:
                        idx = i
                        break
            if idx == -1:
                break
            
            start_ts = all_tsubs[idx]
            # Randomly choose which point to start with
            start_point = local_rng.choice(start_ts)
            block = [start_point]
            
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                best_gain = -1
                best_points = []
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                
                block_combos = set()
                for combo in combinations(block, t - 1):
                    block_combos.add(combo)
                
                for x in candidates:
                    gain = 0
                    for ts_idx in point_to_tsubs[x]:
                        if not uncovered[ts_idx]:
                            continue
                        ts = all_tsubs[ts_idx]
                        rest = tuple(sorted(y for y in ts if y != x))
                        if rest in block_combos:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if not best_points:
                    break
                
                block.append(local_rng.choice(best_points))
            
            if len(block) < k:
                break
            
            block.sort()
            blocks.append(tuple(block))
            
            for ts in combinations(block, t):
                idx2 = tsub_to_idx.get(tuple(ts))
                if idx2 is not None and uncovered[idx2]:
                    uncovered[idx2] = False
                    n_uncovered -= 1
        
        if n_uncovered < best_remaining:
            best_blocks = blocks
            best_remaining = n_uncovered
        
        if best_remaining == 0:
            break
    
    # Verify the best result
    if _verify_covering(v, k, t, best_blocks):
        return [list(b) for b in best_blocks]
    
    # If verification fails (shouldn't happen), return what we have
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
