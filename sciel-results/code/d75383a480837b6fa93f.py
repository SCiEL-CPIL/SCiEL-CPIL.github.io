<<<<<<< SEARCH
def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # Safety margin
    rng = random.Random(42)
    
    # Phase 1: Greedy Construction
    blocks = _greedy(v, k, t, deadline, rng)
    
    # Phase 2: Remove Redundant Blocks
    blocks = _remove_redundant(blocks, v, t, deadline)
    
    # Phase 3: Local Search (Optional, if time permits)
    # Try to swap points or remove/add blocks to reduce count?
    # Given the complexity, sticking to Greedy + Redundancy Removal is safer for 150s limit.
    # But we can try a few restarts with different seeds if time allows.
    
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try a few more greedy runs with different random seeds if time allows
    seed = 43
    while time.time() < deadline - 10.0:
        rng2 = random.Random(seed)
        candidate_blocks = _greedy(v, k, t, deadline, rng2)
        candidate_blocks = _remove_redundant(candidate_blocks, v, t, deadline)
        
        if len(candidate_blocks) < best_count:
            best_blocks = candidate_blocks
            best_count = len(candidate_blocks)
        
        seed += 1
        # Limit restarts to avoid infinite loop if greedy is slow
        if seed > 100: 
            break

    return best_blocks
=======
def _ilp_solve(deadline, upper_bound=None):
    """Solve the set cover ILP using scipy's milp solver"""
    start_time = time.time()
    n_triples = len(idx_to_triple)
    n_blocks = len(all_blocks_list)
    
    # Objective: minimize sum of x_i
    c = np.ones(n_blocks)
    
    # Constraints: A @ x >= 1 for each triple
    A = incidence.toarray()  # n_triples x n_blocks
    lb = np.ones(n_triples)
    ub = np.full(n_triples, np.inf)
    
    # Variable bounds: 0 <= x_i <= 1
    var_lb = np.zeros(n_blocks)
    var_ub = np.ones(n_blocks)
    integrality = np.ones(n_blocks)  # all binary
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(var_lb, var_ub)
    
    # Add time limit in seconds
    time_limit = max(1, int(deadline - time.time() - 5))
    
    try:
        result = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': time_limit, 'mip_rel_gap': 0.01}
        )
        
        if result.status == 0:  # Optimal or near-optimal found
            selected = np.where(result.x > 0.5)[0]
            if upper_bound is not None and len(selected) > upper_bound:
                return None
            return [list(all_blocks_list[i]) for i in selected]
        elif result.status in [1, 2]:  # Locally optimal or infeasible
            # Return best feasible solution if available
            if hasattr(result, 'x') and result.x is not None:
                selected = np.where(result.x > 0.5)[0]
                if len(selected) > 0:
                    if upper_bound is not None and len(selected) > upper_bound:
                        return None
                    return [list(all_blocks_list[i]) for i in selected]
            return None
        else:
            return None
    except Exception as e:
        return None

def _local_search(blocks, v, t, deadline, max_iter=5000):
    """Local search: try removing blocks and replacing with others"""
    if not blocks:
        return blocks
    
    n_triples = len(idx_to_triple)
    cover_count = np.zeros(n_triples, dtype=np.int32)
    block_triples = []
    
    for b in blocks:
        tris = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            cover_count[idx] += 1
            tris.append(idx)
        block_triples.append(tris)
    
    best_blocks = blocks
    best_count = len(blocks)
    iterations = 0
    
    while iterations < max_iter and time.time() < deadline:
        iterations += 1
        improved = False
        
        # Try removing each block if possible
        for i in range(len(best_blocks)):
            tris = block_triples[i]
            if np.all(cover_count[tris] > 1):
                # Remove block i
                best_blocks.pop(i)
                block_triples.pop(i)
                cover_count[tris] -= 1
                best_count -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing a block with a better one
        # Find a block that covers the most "expensive" triples
        # Simple heuristic: try to find a block that covers more uncovered-by-removal triples
        for i in range(len(best_blocks)):
            tris_i = block_triples[i]
            # Count how many triples would become uncovered if we remove block i
            would_uncover = np.sum(cover_count[tris_i] == 1)
            
            if would_uncover == 0:
                # This block is redundant, remove it
                best_blocks.pop(i)
                block_triples.pop(i)
                cover_count[tris_i] -= 1
                best_count -= 1
                improved = True
                break
            elif would_uncover == 1:
                # Only one triple would be uncovered; try to find another block covering it
                unique_tris = tris_i[cover_count[tris_i] == 1]
                target_triple_idx = unique_tris[0]
                
                # Find a block that covers this triple and many others currently covered only by block i
                best_replace = None
                best_replace_score = 0
                for j in range(len(best_blocks)):
                    if j == i:
                        continue
                    tris_j = block_triples[j]
                    # Score: how many of block i's uniquely-covered triples does block j cover?
                    overlap = np.sum((cover_count[tris_j] == 1) & (np.isin(tris_j, tris_i)))
                    if overlap > best_replace_score:
                        best_replace_score = overlap
                        best_replace = j
                
                if best_replace is not None and best_replace_score >= would_uncover:
                    # Replace block i with a new block that covers the target
                    # Find a block covering target_triple_idx not already in best_blocks
                    existing_blocks = set(tuple(b) for b in best_blocks)
                    for block in all_blocks_list:
                        if tuple(block) in existing_blocks:
                            continue
                        # Check if it covers the target triple
                        if target_triple_idx in block_to_triples.get(tuple(block), []):
                            # Compute new cover counts
                            new_tris = block_to_triples[tuple(block)]
                            # Remove old block
                            cover_count[tris_i] -= 1
                            # Add new block
                            cover_count[new_tris] += 1
                            
                            # Check validity: all triples still covered
                            if np.all(cover_count >= 1):
                                best_blocks[i] = list(block)
                                block_triples[i] = new_tris
                                improved = True
                                break
                            else:
                                # Revert
                                cover_count[new_tris] -= 1
                                cover_count[tris_i] += 1
                    
                    if improved:
                        break
        
        if not improved:
            break
    
    return best_blocks

def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # Safety margin
    rng = random.Random(42)
    
    # Phase 1: Fast Greedy Construction
    blocks = _greedy(v, k, t, deadline, rng)
    
    # Phase 2: Remove Redundant Blocks
    blocks = _remove_redundant(blocks, v, t, deadline)
    
    best_blocks = blocks
    best_count = len(blocks)
    
    # Phase 3: Local Search
    if time.time() < deadline - 20.0:
        ls_blocks = _local_search(blocks, v, t, deadline - 10.0)
        if len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)
    
    # Phase 4: Try ILP if time permits
    if time.time() < deadline - 30.0:
        ilp_result = _ilp_solve(deadline, upper_bound=best_count)
        if ilp_result is not None and len(ilp_result) < best_count:
            best_blocks = ilp_result
            best_count = len(ilp_result)
    
    # Phase 5: More local search on best result
    if time.time() < deadline - 10.0 and best_blocks:
        final_blocks = _local_search(best_blocks, v, t, deadline - 5.0)
        if len(final_blocks) < best_count:
            best_blocks = final_blocks
    
    return best_blocks
>>>>>>> REPLACE
