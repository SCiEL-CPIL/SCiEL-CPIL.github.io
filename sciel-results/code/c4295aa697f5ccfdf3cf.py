import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute for each point, the pairs of other points that form a triplet with it
    # pairs_for_point[p] = list of (a, b, triplet_id) such that {p, a, b} is a triplet
    pairs_for_point = [[] for _ in range(v)]
    for tid, triplet in enumerate(all_triplets):
        for i in range(t):
            p = triplet[i]
            others = [x for j, x in enumerate(triplet) if j != i]
            # Since t=3, others has size 2
            a, b = others[0], others[1]
            pairs_for_point[p].append((a, b, tid))

    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet to start
            # To avoid O(N) list conversion, we can just pick a random index and check if it's in uncovered?
            # No, `uncovered` is a set. We need a random element.
            # For small N (2024), list(uncovered) is fast enough (~microseconds).
            start_tid = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_tid]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                # Gain(p) = number of uncovered triplets {p, a, b} such that a, b in block_set
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for a, b, tid in pairs_for_point[p]:
                        if a in block_set and b in block_set:
                            if tid in uncovered:
                                gain += 1
                    if gain > 0:
                        gains[p] = gain
                
                if not gains:
                    # No point increases coverage, pick random
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    # Pick point with max gain. Tie-break randomly.
                    max_gain = max(gains.values())
                    best_candidates = [p for p, g in gains.items() if g == max_gain]
                    best_point = rng.choice(best_candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                uncovered.discard(tid)
        
        return blocks

    # 1. Initial Construction
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy instances
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Try to improve immediately? No, let's collect a few first.
            
    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # 2. Local Search Optimization
    # Maintain count of how many blocks cover each triplet
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta=1):
        for block in blocks:
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    
    def is_redundant(block):
        for triplet in itertools.combinations(block, t):
            if count[triplet_to_id[triplet]] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
            
        # Precompute triplet ID sets for each block
        block_ids = []
        for b in blocks:
            ids = set()
            for triplet in itertools.combinations(b, t):
                ids.add(triplet_to_id[triplet])
            block_ids.append(ids)
        
        # Try all pairs
        for i in range(n):
            if time.time() > deadline - 5:
                return blocks, False
            b1 = blocks[i]
            ids_b1 = block_ids[i]
                
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    return blocks, False
                b2 = blocks[j]
                ids_b2 = block_ids[j]
                
                # Find critical triplets: those covered ONLY by b1 and/or b2
                critical_points = set()
                
                # Triplets in b1
                for tid in ids_b1:
                    c = count[tid]
                    cov_b1 = 1
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        critical_points.update(all_triplets[tid])
                        
                # Triplets in b2 (not in b1)
                for tid in ids_b2:
                    if tid in ids_b1:
                        continue
                    c = count[tid]
                    cov_b2 = 1
                    if c == cov_b2:
                        critical_points.update(all_triplets[tid])
                
                if not critical_points:
                    # Both redundant
                    update_counts([b1, b2], -1)
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
                
                if len(critical_points) <= k:
                    # Can form one block from critical points
                    new_block = list(critical_points)
                    # Fill greedily
                    point_freq = np.zeros(v, dtype=np.int32)
                    for idx, b in enumerate(blocks):
                        if idx == i or idx == j:
                            continue
                        for p in b:
                            point_freq[p] += 1
                            
                    candidates = [p for p in range(v) if p not in new_block]
                    candidates.sort(key=lambda p: point_freq[p], reverse=True)
                    
                    for p in candidates:
                        if len(new_block) >= k:
                            break
                        new_block.append(p)
                    
                    new_block.sort()
                    
                    # Verify validity: new_block must cover all critical triplets
                    # Since critical_points are in new_block, and critical triplets are subsets of critical_points,
                    # they are covered.
                    
                    update_counts([b1, b2], -1)
                    update_counts([new_block], 1)
                    
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Main Optimization Loop
    progress_made = True
    max_iterations = 200
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 2:
            break
        progress_made = False
        
        # 1. Remove redundants
        while True:
            current_blocks, removed = try_remove_redundant(current_blocks)
            if not removed:
                break
            progress_made = True
        
        # 2. Try 2-for-1 swaps
        if not progress_made:
            current_blocks, swapped = try_swap_2_for_1(current_blocks)
            if swapped:
                progress_made = True
        
        if not progress_made:
            break

    # Final pass to ensure no redundants remain
    current_blocks, _ = try_remove_redundant(current_blocks)
    
    return current_blocks
