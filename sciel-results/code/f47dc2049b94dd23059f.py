import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    # 1. Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # 2. Generate all candidate blocks and their masks
    # For v=24, k=7, num_blocks = 346104. This is manageable.
    all_blocks = []
    all_block_masks = []
    
    # Precompute 24-cell vertices for geometric filtering
    # The 24-cell has vertices at:
    # (±1, 0, 0, 0) and permutations (24 vertices)
    # and (±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs (24 vertices)
    # Total 48 vertices? No, the 24-cell has 24 vertices.
    # The standard coordinates for the 24-cell are:
    # (±1, 0, 0, 0) and permutations -> 24 vertices? 
    # Let's check: 4 positions for non-zero, 2 signs -> 8. Plus 4! / (2! 2!) = 6 permutations of (1,1,0,0)? No.
    # The 24-cell vertices are all permutations of (±1, ±1, 0, 0) with an even number of minus signs?
    # Actually, the 24-cell is the set of all permutations of (±1, ±1, 0, 0) with even number of minus signs?
    # No, the 24-cell vertices are:
    # 1. Permutations of (±1, ±1, 0, 0) with even number of minus signs: 
    #    Choose 2 positions for non-zeros: C(4,2)=6. 
    #    Signs: 4 combos, 2 have even minus signs (++, --). So 6*2 = 12.
    # 2. Permutations of (±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs:
    #    4 signs, 2 have even minus signs. So 2.
    # Total 14? No.
    
    # Let's use a known set of 24 points with high symmetry.
    # The 24-cell coordinates are often given as:
    # (±1, 0, 0, 0) and permutations (24 vertices? No, 4*2 = 8)
    # (±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs (24 vertices? No, 8 total, 4 even)
    # Wait, the 24-cell has 24 vertices.
    # The vertices are all permutations of (±1, ±1, 0, 0) with even number of minus signs?
    # Let's list them properly.
    # The 24-cell is the convex hull of the 24 points:
    # (±1, 0, 0, 0) and permutations: 4 * 2 = 8 points.
    # (±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs: 2^4 = 16 points, half have even minus signs -> 8 points.
    # Total 16 points? No.
    
    # Actually, the 24-cell vertices are:
    # All permutations of (±1, ±1, 0, 0) with even number of minus signs?
    # Let's count:
    # Positions for 0: 4 choices.
    # Signs for the two 1s: 4 combos. Even minus signs: ++, --. So 2.
    # Permutations of the two 1s: 1 (they are identical).
    # So 4 * 2 = 8 points.
    # Plus the (±1/2, ±1/2, ±1/2, ±1/2) points?
    # There are 16 such points. Half have even number of minus signs?
    # Even number of minus signs: 0, 2, 4.
    # 0 minus: 1.
    # 2 minus: C(4,2) = 6.
    # 4 minus: 1.
    # Total 8 points.
    # 8 + 8 = 16 points. Still not 24.
    
    # Correction: The 24-cell vertices are all permutations of (±1, 0, 0, 0) AND (±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs?
    # No, the standard 24-cell has 24 vertices.
    # Let's use the 24 points from the "24-cell" construction:
    # The 24 vertices are:
    # (±1, 0, 0, 0) and permutations (24 vertices? No, 4 positions * 2 signs = 8)
    # (±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs (8 vertices)
    # This gives 16.
    
    # I will use a different high-symmetry set of 24 points if the 24-cell doesn't map directly, 
    # or simply use random points if the geometric filter is too complex to implement correctly in time.
    # However, the "Geometric Projection Strategy" suggests using symmetry.
    # Let's use the 24 points as the vertices of the 24-cell properly.
    # The 24-cell vertices are:
    # 1. (±1, ±1, 0, 0) and permutations with even number of minus signs?
    #    Permutations of (1,1,0,0): 12.
    #    Signs: 4. Even minus: 2.
    #    12 * 2 / 2 (identical 1s) = 12?
    #    Let's just generate 24 random points with high symmetry by using the 24-cell coordinates from a library or known list.
    
    # Known 24-cell vertices (scaled to integer coordinates for simplicity, or just use floats):
    # The 24-cell is the set of all permutations of (±1, ±1, 0, 0) with an even number of minus signs?
    # No, that's 24 vertices.
    # Let's verify:
    # Permutations of (1,1,0,0): 12.
    # For each, assign signs to the two non-zeros.
    # Even number of minus signs: (++, --).
    # So for each permutation, 2 sign choices.
    # 12 * 2 = 24.
    # Yes! The 24 vertices are all permutations of (±1, ±1, 0, 0) with an even number of minus signs.
    
    vertices = []
    seen = set()
    for perm in itertools.permutations([1, 1, 0, 0]):
        # perm is a tuple of 4 ints, two 1s and two 0s.
        # We need to assign signs to the two 1s.
        # Even number of minus signs: 0 or 2.
        # Case 1: Both positive (++, 0 minus)
        v1 = [p if p == 0 else 1 for p in perm]
        v1_tuple = tuple(v1)
        if v1_tuple not in seen:
            seen.add(v1_tuple)
            vertices.append(v1_tuple)
        
        # Case 2: Both negative (--, 2 minus)
        v2 = [p if p == 0 else -1 for p in perm]
        v2_tuple = tuple(v2)
        if v2_tuple not in seen:
            seen.add(v2_tuple)
            vertices.append(v2_tuple)
            
    # We should have 24 unique vertices.
    # Map point index i to vertex i.
    
    # Compute pairwise squared distances
    n_points = len(vertices)
    dist_sq = [[0.0] * n_points for _ in range(n_points)]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = 0.0
            for dim in range(4):
                diff = vertices[i][dim] - vertices[j][dim]
                d += diff * diff
            dist_sq[i][j] = d
            dist_sq[j][i] = d
            
    # Generate all blocks and compute "energy" (sum of pairwise squared distances)
    # High energy = well-spread block.
    block_energies = []
    for block in itertools.combinations(range(v), k):
        energy = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                energy += dist_sq[block[i]][block[j]]
        block_energies.append((energy, block))
        
    # Sort by energy descending
    block_energies.sort(key=lambda x: -x[0])
    
    # Select top candidates. 
    # The top 10% or 20% of blocks by energy are likely to be "good" for covering.
    # Let's take the top 20% (approx 70,000 blocks) to keep search fast but diverse.
    num_candidates = max(1000, len(block_energies) // 5)
    candidate_blocks = [b for _, b in block_energies[:num_candidates]]
    
    # Create masks for candidate blocks only
    blocks = candidate_blocks
    block_masks = []
    for block in blocks:
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(blocks)
    
    # Precompute block lists for each triple for fast intersection lookups
    # triple_to_blocks: list of indices of blocks containing this triple
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        mask = block_masks[i]
        # Iterate set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(i)
            m ^= lsb
            
    def run_greedy(seed_val):
        random.seed(seed_val)
        uncovered = (1 << num_triples) - 1
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline:
                return None # Timeout
            
            best_gain = -1
            candidates = []
            
            for i in range(num_blocks):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_gain == 0:
                return selected_indices # Should not happen if valid input
                
            # Randomly pick from best candidates
            best_idx = random.choice(candidates)
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected_indices

    def refine_indices(selected_indices):
        if selected_indices is None:
            return []
            
        solution_set = set(selected_indices)
        
        coverage_count = [0] * num_triples
        for idx in solution_set:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                coverage_count[idx_t] += 1
                m ^= lsb
                
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for idx in list(solution_set):
                mask = block_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if coverage_count[idx_t] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    solution_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        coverage_count[idx_t] -= 1
                        m ^= lsb
                    changed = True
                    
        return list(solution_set)

    def local_search_2exchange(solution_indices):
        """
        Attempts to reduce the number of blocks by replacing pairs of blocks
        with a single block that covers all 'unique' triples of the pair.
        """
        if solution_indices is None:
            return []
            
        sol_set = set(solution_indices)
        if not sol_set:
            return []
            
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Recalculate coverage counts and unique status
            coverage_count = [0] * num_triples
            for idx in sol_set:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    coverage_count[idx_t] += 1
                    m ^= lsb
            
            # List of blocks in solution to iterate
            sol_list = list(sol_set)
            
            # Check all pairs
            for i in range(len(sol_list)):
                for j in range(i + 1, len(sol_list)):
                    b1 = sol_list[i]
                    b2 = sol_list[j]
                    
                    mask1 = block_masks[b1]
                    mask2 = block_masks[b2]
                    mask_pair = mask1 | mask2
                    
                    critical_triples = []
                    m = mask_pair
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        if coverage_count[idx_t] == 1:
                            critical_triples.append(idx_t)
                        elif coverage_count[idx_t] == 2:
                            # Check if in both
                            if (mask1 & lsb) and (mask2 & lsb):
                                critical_triples.append(idx_t)
                        m ^= lsb
                    
                    if not critical_triples:
                        # Both blocks are redundant relative to each other? 
                        sol_set.remove(b1)
                        sol_set.remove(b2)
                        improved = True
                        break # Restart loop with new state
                    else:
                        # Find a block b_k that covers all critical_triples
                        critical_triples.sort(key=lambda x: len(triple_to_blocks[x]))
                        
                        # Get candidates from the first triple
                        candidates = triple_to_blocks[critical_triples[0]]
                        
                        # Intersect with others
                        for t_idx in critical_triples[1:]:
                            if not candidates:
                                break
                            # Intersect candidates with triple_to_blocks[t_idx]
                            candidates = set(candidates).intersection(triple_to_blocks[t_idx])
                            candidates = list(candidates)
                            
                        if candidates:
                            # Check if any candidate is NOT in sol_set
                            for b_k in candidates:
                                if b_k not in sol_set:
                                    # Replace b1, b2 with b_k
                                    sol_set.remove(b1)
                                    sol_set.remove(b2)
                                    sol_set.add(b_k)
                                    improved = True
                                    break # Restart loop
                            
                            if improved:
                                break
                            
            # If we finished the double loop without 'improved', we break the while loop
            # Note: If we removed blocks, we break out of inner loops and restart while.
            
        return list(sol_set)

    best_solution_indices = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Since the candidate pool is smaller (20% of all blocks), greedy is faster.
    # We can afford more restarts.
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        greedy_result = run_greedy(i)
        if greedy_result is None:
            continue
            
        refined_indices = refine_indices(greedy_result)
        
        if len(refined_indices) < best_size:
            best_size = len(refined_indices)
            best_solution_indices = refined_indices
            
    if best_solution_indices is not None:
        # Run local search on the best greedy result
        final_indices = local_search_2exchange(best_solution_indices)
        if len(final_indices) < len(best_solution_indices):
            best_solution_indices = final_indices
            
    if best_solution_indices is None:
        # Fallback
        greedy_result = run_greedy(0)
        best_solution_indices = refine_indices(greedy_result) if greedy_result else []

    # Convert indices to blocks
    best_solution = [blocks[i] for i in best_solution_indices]
    best_solution.sort()
    return [list(b) for b in best_solution]
