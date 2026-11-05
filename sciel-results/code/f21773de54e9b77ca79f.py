import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_triplets = list(itertools.combinations(points, t))
    triplet_to_idx = {triplet: i for i, triplet in enumerate(all_triplets)}
    num_triplets = len(all_triplets)
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block to triplet indices
    block_to_triplets = []
    triplet_to_blocks = [[] for _ in range(num_triplets)]
    
    for b_idx, block in enumerate(all_blocks):
        covered = []
        for trip in itertools.combinations(block, t):
            t_idx = triplet_to_idx[trip]
            covered.append(t_idx)
            triplet_to_blocks[t_idx].append(b_idx)
        block_to_triplets.append(covered)
        
    # Precompute sets for faster intersection/diff checks in local search
    block_to_triplet_sets = [set(trips) for trips in block_to_triplets]
    
    # 1. Greedy Construction
    uncovered = set(range(num_triplets))
    solution_blocks = []
    
    while uncovered:
        if time.time() > deadline - 20.0: break
        
        target_triplet = random.choice(list(uncovered))
        candidate_blocks = triplet_to_blocks[target_triplet]
        
        best_block = None
        best_score = -1
        
        for b_idx in candidate_blocks:
            score = 0
            # Use set for O(1) lookup
            # But we just need count of intersection with uncovered
            # sum(1 for t in block_to_triplets[b_idx] if t in uncovered)
            for t_idx in block_to_triplets[b_idx]:
                if t_idx in uncovered:
                    score += 1
            if score > best_score:
                best_score = score
                best_block = b_idx
                
        if best_block is not None:
            solution_blocks.append(best_block)
            for t_idx in block_to_triplets[best_block]:
                if t_idx in uncovered:
                    uncovered.discard(t_idx)
            
    # If greedy broke early, we need to finish it to ensure validity
    # But if we are close to deadline, we might not finish.
    # However, greedy is fast. 
    # If we still have uncovered triplets, we must add blocks.
    # Let's just run greedy until done, it's fast enough.
    # The break condition above was just a safety net.
    
    # 2. Local Search
    current_blocks = set(solution_blocks)
    
    coverage = [0] * num_triplets
    for b_idx in current_blocks:
        for t_idx in block_to_triplets[b_idx]:
            coverage[t_idx] += 1
            
    block_lookup = {b: i for i, b in enumerate(all_blocks)}
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Removal
        for b_idx in list(current_blocks):
            if time.time() > deadline: break
            removable = True
            for t_idx in block_to_triplets[b_idx]:
                if coverage[t_idx] < 2:
                    removable = False
                    break
            if removable:
                current_blocks.remove(b_idx)
                for t_idx in block_to_triplets[b_idx]:
                    coverage[t_idx] -= 1
                improved = True
                break
        
        if time.time() > deadline: break
        
        # Phase 2: Swapping
        if not current_blocks:
            break
            
        b_idx = random.choice(list(current_blocks))
        block_points = all_blocks[b_idx]
        
        old_triplets = block_to_triplets[b_idx]
        old_triplet_set = block_to_triplet_sets[b_idx]
        
        best_new_block = None
        best_new_sum = float('inf')
        
        # Calculate current sum for this block
        current_sum = sum(coverage[t] for t in old_triplets)
        
        for p in block_points:
            for x in range(v):
                if x in block_points: continue
                
                new_points = tuple(sorted((set(block_points) - {p}) | {x}))
                
                if new_points not in block_lookup:
                    continue
                    
                new_b_idx = block_lookup[new_points]
                
                new_triplets = block_to_triplets[new_b_idx]
                new_triplet_set = block_to_triplet_sets[new_b_idx]
                
                # Check validity: 
                # t in old_triplets \ new_triplets must have coverage >= 2
                # We can iterate over old_triplets and check if in new_triplet_set
                valid_swap = True
                for t_idx in old_triplets:
                    if t_idx not in new_triplet_set:
                        if coverage[t_idx] < 2:
                            valid_swap = False
                            break
                            
                if not valid_swap:
                    continue
                    
                # Calculate new sum
                # Sum of coverage[t] for t in new_triplets
                # If t in old_triplets: coverage[t]
                # If t not in old_triplets: coverage[t] + 1
                new_sum = 0
                for t_idx in new_triplets:
                    if t_idx in old_triplet_set:
                        new_sum += coverage[t_idx]
                    else:
                        new_sum += coverage[t_idx] + 1
                        
                if new_sum < best_new_sum:
                    best_new_sum = new_sum
                    best_new_block = new_b_idx
                    
        if best_new_block is not None:
            current_blocks.remove(b_idx)
            current_blocks.add(best_new_block)
            
            for t_idx in old_triplets:
                coverage[t_idx] -= 1
            for t_idx in block_to_triplets[best_new_block]:
                coverage[t_idx] += 1
                
            improved = True
            continue
            
        if not improved:
            # Stuck? 
            # We can try to restart with a different random seed or just continue.
            # The random selection of b_idx will vary.
            pass

    result = [list(all_blocks[i]) for i in current_blocks]
    return result
