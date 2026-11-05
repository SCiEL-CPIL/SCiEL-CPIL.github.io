import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Map triplet id to the triplet itself for reverse lookup if needed
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Precompute for each point, which triplets contain it
    point_to_triplets = defaultdict(list)
    for tid, triplet in enumerate(all_triplets):
        for p in triplet:
            point_to_triplets[p].append(tid)
    
    # Precompute block masks for all possible blocks? Too many (C(24,7) = 346104)
    # Instead, compute on the fly but cache
    block_mask_cache = {}
    
    def get_block_mask(block):
        key = tuple(sorted(block))
        if key in block_mask_cache:
            return block_mask_cache[key]
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_mask_cache[key] = mask
        return mask

    # Enumerate blocks containing a set of points
    def blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))

    # Weighted Greedy Construction with dynamic point weights
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        
        # Point weights: initially 1.0, increased for points in uncovered triplets
        point_weights = np.ones(v, dtype=np.float64)
        
        # Track uncovered count per point for weight updates
        uncovered_per_point = np.zeros(v, dtype=np.int64)
        for p in range(v):
            for tid in point_to_triplets[p]:
                if (uncovered >> tid) & 1:
                    uncovered_per_point[p] += 1
        
        while uncovered:
            # Pick a random uncovered triplet to guide the search
            idx = -1
            for _ in range(10):
                cand = random.randint(0, n_triplets - 1)
                if (uncovered >> cand) & 1:
                    idx = cand
                    break
            if idx == -1:
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1
            
            start_triplet = id_to_triplet[idx]
            
            # Find best block containing start_triplet using weighted ratio
            best_block = None
            best_score = -1
            best_mask = 0
            
            # Enumerate all blocks containing start_triplet
            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                
                # Compute product of point weights for this block
                weight_product = 1.0
                for p in block:
                    weight_product *= point_weights[p]
                
                # Score: gain / weight_product
                # Higher weight_product means points are "expensive" (many uncovered triplets)
                # So we prefer blocks that cover triplets with less "pressure" points
                score = gain / max(weight_product, 1e-10)
                
                if score > best_score:
                    best_score = score
                    best_block = block
                    best_mask = b_mask
            
            blocks.append(best_block)
            uncovered &= ~best_mask
            
            # Update point weights
            # Increase weight for points that still have many uncovered triplets
            for p in range(v):
                for tid in point_to_triplets[p]:
                    if (uncovered >> tid) & 1:
                        uncovered_per_point[p] += 1
                # Weight proportional to sqrt of uncovered count (smooth)
                point_weights[p] = 1.0 + 0.5 * np.sqrt(max(uncovered_per_point[p], 1))
            
        return blocks

    # Local Search Optimizer
    def optimize(blocks, time_limit):
        # Maintain counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        # Optimization Loop
        no_improve_count = 0
        
        while time.time() < time_limit:
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                b = blocks[i]
                redundant = True
                for triplet in itertools.combinations(b, t):
                    if counts[triplet_to_id[triplet]] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for triplet in itertools.combinations(b, t):
                        counts[triplet_to_id[triplet]] -= 1
                    improved = True
                    break # Restart loop as indices changed
            
            # 2. Try 2-for-1
            if not improved:
                # Identify blocks with high unique load
                unique_loads = []
                for i in range(n):
                    b = blocks[i]
                    load = 0
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            load += 1
                    unique_loads.append((load, i))
                
                # Sort by load descending, try top pairs
                unique_loads.sort(reverse=True)
                tried_pairs = set()
                
                for load1, i in unique_loads[:10]: # Top 10 blocks
                    if load1 == 0:
                        continue
                    for load2, j in unique_loads[:10]:
                        if i == j:
                            continue
                        if (i, j) in tried_pairs or (j, i) in tried_pairs:
                            continue
                        tried_pairs.add((i, j))
                        
                        b1 = blocks[i]
                        b2 = blocks[j]
                        
                        # Compute needed triplets
                        ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                        ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))
                        all_involved = ids_b1 | ids_b2
                        
                        needed_triplets = []
                        points_needed = set()
                        for tid in all_involved:
                            if counts[tid] == 1:
                                needed_triplets.append(tid)
                                points_needed.update(id_to_triplet[tid])
                        
                        if not needed_triplets:
                            # Both redundant? Should have been caught, but just in case
                            blocks.pop(j)
                            for triplet in itertools.combinations(b2, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.pop(i)
                            for triplet in itertools.combinations(b1, t):
                                counts[triplet_to_id[triplet]] -= 1
                            improved = True
                            break
                        
                        if len(points_needed) > k:
                            continue
                        
                        # Enumerate blocks containing points_needed
                        found_replacement = False
                        for new_block in blocks_containing(points_needed):
                            if set(new_block) == set(b1) or set(new_block) == set(b2):
                                continue
                            
                            # Check coverage
                            covers = True
                            new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                            for tid in needed_triplets:
                                if tid not in new_ids:
                                    covers = False
                                    break
                            
                            if covers:
                                # Perform swap
                                blocks.pop(j)
                                for triplet in itertools.combinations(b2, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.pop(i)
                                for triplet in itertools.combinations(b1, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                
                                blocks.append(new_block)
                                for triplet in itertools.combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1
                                
                                improved = True
                                found_replacement = True
                                break
                            
                        if improved:
                            break
                        if time.time() > time_limit - 2:
                            break

            # 3. Try 1-for-1
            if not improved:
                # Pick block with highest unique load
                i = random.randint(0, n-1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                
                needed_triplets = []
                points_needed = set()
                for tid in ids_b1:
                    if counts[tid] == 1:
                        needed_triplets.append(tid)
                        points_needed.update(id_to_triplet[tid])
                        
                if not needed_triplets:
                    continue
                
                if len(points_needed) > k:
                    continue
                    
                for new_block in blocks_containing(points_needed):
                    if set(new_block) == set(b1):
                        continue
                    new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                    covers = True
                    for tid in needed_triplets:
                        if tid not in new_ids:
                            covers = False
                            break
                    if covers:
                        # Replace
                        blocks.pop(i)
                        for triplet in itertools.combinations(b1, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks.append(new_block)
                        for triplet in itertools.combinations(new_block, t):
                            counts[triplet_to_id[triplet]] += 1
                        improved = True
                        break
                        
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple times with different random seeds
    # Allocate time for each restart
    n_restarts = 10
    time_per_restart = (deadline - start_time) / n_restarts
    
    for seed in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        # Greedy init
        init_blocks = greedy_construct(seed)
        
        # Optimize
        opt_blocks = optimize(init_blocks, deadline - 5)
        
        # Verify validity
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
                
        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
                
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_construct(0)
        
    return best_blocks

if __name__ == "__main__":
    pass
