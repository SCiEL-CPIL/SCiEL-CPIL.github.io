"""Covering design C(v,k,t) via accelerated greedy + local search with deadline guard."""
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Precompute all t-subsets and map them to indices for fast lookup
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each point, precompute which t-subsets it participates in
    # point_tsubs[p] = list of t-subset indices containing point p
    point_tsubs = [[] for _ in range(v)]
    for ts in all_tsubs:
        idx = tsub_to_idx[ts]
        for p in ts:
            point_tsubs[p].append(idx)
    
    # Greedy construction: repeatedly pick the block (k-subset) that covers the most uncovered t-subsets
    # To speed up, we use a heuristic: build blocks greedily by adding points that maximize coverage
    
    uncovered = [1] * n_tsubs  # 1 if uncovered, 0 if covered
    blocks = []
    block_set = set()
    
    def time_check():
        return time.time() > deadline
    
    # First pass: pure greedy
    while True:
        if time_check():
            break
        # Find the most uncovered t-subset to anchor a new block
        # Use a random-ish approach: pick from the uncovered ones
        uncovered_indices = [i for i in range(n_tsubs) if uncovered[i]]
        if not uncovered_indices:
            break
        
        # Pick a random uncovered t-subset as seed
        seed_idx = random.choice(uncovered_indices[:min(len(uncovered_indices), 1000)])
        seed_tsub = all_tsubs[seed_idx]
        
        # Build block starting from seed, greedily adding points
        block = list(seed_tsub)
        
        while len(block) < k:
            if time_check():
                break
            best_point = -1
            best_gain = -1
            for p in range(v):
                if p in block:
                    continue
                # Count how many currently uncovered t-subsets would be covered by adding p
                gain = 0
                for rest_idx in point_tsubs[p]:
                    if uncovered[rest_idx]:
                        # Check if all other elements of this t-subset are in block
                        ts = all_tsubs[rest_idx]
                        if all(x in block or x == p for x in ts):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point == -1:
                break
            block.append(best_point)
        
        if len(block) < k:
            # Pad with random points
            remaining = [p for p in range(v) if p not in block]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
        
        block.sort()
        block_tuple = tuple(block)
        if block_tuple not in block_set:
            block_set.add(block_tuple)
            blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = 0
    
    # Check if we have full coverage
    if any(uncovered):
        # Continue greedy until done or timeout
        while True:
            if time_check():
                break
            uncovered_indices = [i for i in range(n_tsubs) if uncovered[i]]
            if not uncovered_indices:
                break
            seed_idx = random.choice(uncovered_indices[:min(len(uncovered_indices), 1000)])
            seed_tsub = all_tsubs[seed_idx]
            block = list(seed_tsub)
            while len(block) < k:
                if time_check():
                    break
                best_point = -1
                best_gain = -1
                for p in range(v):
                    if p in block:
                        continue
                    gain = 0
                    for rest_idx in point_tsubs[p]:
                        if uncovered[rest_idx]:
                            ts = all_tsubs[rest_idx]
                            if all(x in block or x == p for x in ts):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point == -1:
                    break
                block.append(best_point)
            if len(block) < k:
                remaining = [p for p in range(v) if p not in block]
                random.shuffle(remaining)
                block.extend(remaining[:k - len(block)])
            block.sort()
            block_tuple = tuple(block)
            if block_tuple not in block_set:
                block_set.add(block_tuple)
                blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                uncovered[idx] = 0
    
    # Now do local search: try to remove redundant blocks
    # A block is redundant if every t-subset in it is covered by other blocks
    def is_redundant(block_idx):
        block = blocks[block_idx]
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            # Check if any other block covers this t-subset
            found = False
            for j, b in enumerate(blocks):
                if j == block_idx:
                    continue
                if all(x in b for x in ts):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Local search: iteratively try to remove redundant blocks
    improved = True
    while improved and not time_check():
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time_check():
                break
            if is_redundant(i):
                blocks.pop(i)
                block_set.discard(tuple(sorted(blocks[i])) if i < len(blocks) else None)
                # Rebuild block_set
                block_set = set(tuple(b) for b in blocks)
                improved = True
    
    # Try to replace pairs of blocks with a single better block
    if not time_check():
        for i in range(len(blocks)):
            if time_check():
                break
            for j in range(i + 1, len(blocks)):
                if time_check():
                    break
                # Try to find a single block that covers all t-subsets covered by blocks[i] and blocks[j]
                # that are not covered by other blocks
                critical_tsubs = []
                for ts in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[ts]
                    covered_by_other = False
                    for m, b in enumerate(blocks):
                        if m == i or m == j:
                            continue
                        if all(x in b for x in ts):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        critical_tsubs.append(ts)
                for ts in itertools.combinations(blocks[j], t):
                    idx = tsub_to_idx[ts]
                    covered_by_other = False
                    for m, b in enumerate(blocks):
                        if m == i or m == j:
                            continue
                        if all(x in b for x in ts):
                            covered_by_other = True
                            break
                    if not covered_by_other and ts not in critical_tsubs:
                        critical_tsubs.append(ts)
                
                if len(critical_tsubs) == 0:
                    continue
                
                # Try to find a k-subset covering all critical t-subsets
                # This is hard in general; skip for now
                pass
    
    return blocks
