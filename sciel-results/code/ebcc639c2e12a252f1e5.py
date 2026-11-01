import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_to_idx = {ts: i for i, ts in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Map block to list of triple indices
    block_triples = []
    block_masks = []
    all_ones_mask = (1 << n_triples) - 1
    
    for block in all_blocks:
        t_indices = []
        for ts in itertools.combinations(block, t):
            t_indices.append(triple_to_idx[ts])
        block_triples.append(t_indices)
        
        mask = 0
        for idx in t_indices:
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # Precompute triple_to_blocks for fast intersection
    # This might be memory heavy: 2024 lists of ~ (C(24,7)/C(24,3)) blocks?
    # C(24,7) = 346104. C(24,3) = 2024.
    # Each block covers 35 triples.
    # Total entries = 346104 * 35 = 12,113,640.
    # This is fine.
    triple_to_blocks = [[] for _ in range(n_triples)]
    for b_idx, t_indices in enumerate(block_triples):
        for t_idx in t_indices:
            triple_to_blocks[t_idx].append(b_idx)

    def _greedy(rng):
        covered = 0
        solution = []
        while covered != all_ones_mask:
            best_block = -1
            best_gain = -1
            # Sample a subset of blocks if n_blocks is large? 
            # 346k blocks is manageable for one pass if optimized.
            # But we do this many times.
            # Let's iterate all.
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & (~covered & all_ones_mask)).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                elif gain == best_gain and gain > 0:
                    # Random tie break
                    if rng.random() < 0.5:
                        best_block = b_idx
            if best_block == -1:
                break
            solution.append(best_block)
            covered |= block_masks[best_block]
        return solution

    def _local_search(solution, rng):
        if not solution:
            return solution
        
        # Initialize coverage counts
        cover_count = [0] * n_triples
        covered_mask = 0
        sol_set = set(solution)
        
        for b_idx in solution:
            covered_mask |= block_masks[b_idx]
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        current_solution = solution[:]
        
        # Time check inside loop
        while time.time() < deadline:
            improved = False
            
            # 1. Remove redundant blocks
            for i in range(len(current_solution) - 1, -1, -1):
                b_idx = current_solution[i]
                # Check if redundant
                redundant = True
                for t_idx in block_triples[b_idx]:
                    if cover_count[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    del current_solution[i]
                    sol_set.remove(b_idx)
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    covered_mask ^= block_masks[b_idx] # XOR works if we know it's the only one? No, use AND NOT
                    # Actually covered_mask is union. Removing b_idx might not clear bits if others cover them.
                    # But if redundant, others DO cover them. So covered_mask stays same?
                    # Wait, if redundant, cover_count[t] > 1 for all t in b.
                    # So covered_mask does NOT change.
                    # So we just update cover_count and solution.
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. 1-1 Swap
            # Try to replace one block with another
            # Randomly pick an index to try
            if len(current_solution) > 0:
                i = rng.randint(0, len(current_solution) - 1)
                b_old = current_solution[i]
                
                # Temporarily remove b_old
                for t_idx in block_triples[b_old]:
                    cover_count[t_idx] -= 1
                
                # Find uncovered triples
                uncovered_indices = []
                for t_idx in block_triples[b_old]:
                    if cover_count[t_idx] == 0:
                        uncovered_indices.append(t_idx)
                
                if not uncovered_indices:
                    # It was redundant, remove it
                    del current_solution[i]
                    sol_set.remove(b_old)
                    # covered_mask unchanged
                    improved = True
                else:
                    # Need to find b_new that covers all uncovered_indices
                    # And is not in current solution
                    # Intersect triple_to_blocks for all uncovered_indices
                    candidates = None
                    for t_idx in uncovered_indices:
                        if candidates is None:
                            candidates = set(triple_to_blocks[t_idx])
                        else:
                            candidates &= set(triple_to_blocks[t_idx])
                        
                        if not candidates:
                            break
                    
                    if candidates:
                        # Remove current solution blocks from candidates
                        candidates -= sol_set
                        
                        if candidates:
                            # Pick best candidate
                            # Best: maximizes additional coverage? Or just pick random?
                            # Let's pick one that covers the most OTHER uncovered triples?
                            # There are no other uncovered triples, we only removed b_old.
                            # So any candidate that covers the holes works.
                            # To improve future potential, pick one that overlaps less with existing?
                            # Or just pick random.
                            b_new = rng.choice(list(candidates))
                            
                            # Perform swap
                            current_solution[i] = b_new
                            sol_set.add(b_new)
                            
                            # Update cover_count
                            for t_idx in block_triples[b_new]:
                                cover_count[t_idx] += 1
                            
                            # Update covered_mask
                            covered_mask |= block_masks[b_new]
                            # Note: covered_mask might have bits from b_old that are now 0?
                            # No, we found b_new covering all holes.
                            # So covered_mask is still all_ones?
                            # Yes, if the solution was valid before, and we covered the holes, it's valid.
                            improved = True
                    else:
                        # No candidate found, revert
                        for t_idx in block_triples[b_old]:
                            cover_count[t_idx] += 1
            
            if not improved:
                # Random perturbation: Swap two random blocks
                if len(current_solution) >= 2:
                    i, j = rng.sample(range(len(current_solution)), 2)
                    b1, b2 = current_solution[i], current_solution[j]
                    # Remove
                    for t_idx in block_triples[b1]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_triples[b2]:
                        cover_count[t_idx] -= 1
                    sol_set.discard(b1)
                    sol_set.discard(b2)
                    
                    # Add random blocks
                    b_new1 = rng.randint(0, n_blocks - 1)
                    b_new2 = rng.randint(0, n_blocks - 1)
                    
                    # Avoid duplicates and self
                    while b_new1 == b_new2 or b_new1 in sol_set or b_new2 in sol_set:
                        b_new1 = rng.randint(0, n_blocks - 1)
                        b_new2 = rng.randint(0, n_blocks - 1)
                        
                    current_solution[i] = b_new1
                    current_solution[j] = b_new2
                    sol_set.add(b_new1)
                    sol_set.add(b_new2)
                    
                    for t_idx in block_triples[b_new1]:
                        cover_count[t_idx] += 1
                    for t_idx in block_triples[b_new2]:
                        cover_count[t_idx] += 1
                    
                    covered_mask = 0
                    for b in current_solution:
                        covered_mask |= block_masks[b]
                    
                elif len(current_solution) == 1:
                    b_old = current_solution[0]
                    for t_idx in block_triples[b_old]:
                        cover_count[t_idx] -= 1
                    sol_set.discard(b_old)
                    
                    b_new = rng.randint(0, n_blocks - 1)
                    current_solution[0] = b_new
                    sol_set.add(b_new)
                    
                    for t_idx in block_triples[b_new]:
                        cover_count[t_idx] += 1
                    covered_mask = block_masks[b_new]

        return current_solution

    best_solution = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple rounds
    while time.time() < deadline - 5.0:
        # Greedy
        sol = _greedy(rng)
        
        # Local Search
        improved_sol = _local_search(sol, rng)
        
        if len(improved_sol) < best_size:
            best_size = len(improved_sol)
            best_solution = improved_sol[:]
            
        # If we found the lower bound, stop?
        # Lower bound is 69.
        if best_size <= 69:
            break

    if best_solution is None:
        # Fallback
        best_solution = _greedy(rng)
        
    # Convert indices to blocks
    return [list(all_blocks[idx]) for idx in best_solution]
