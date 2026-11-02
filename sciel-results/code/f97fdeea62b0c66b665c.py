<<<<<<< SEARCH
def _local_search_sa(blocks, v, k, t, rng, time_limit):
    """
    Simulated Annealing style local search.
    We maintain a valid covering. Moves are:
    1. Remove a redundant block (if any exist, though we usually start reduced).
    2. Swap a block with a random candidate.
    3. Swap a block with a candidate that covers "rare" triples (heuristic).
    
    To allow escaping local minima, we might temporarily accept a "worse" state?
    No, the constraint is strict validity. We cannot have an invalid state.
    So "SA" here means we explore the space of valid coverings.
    The "energy" is the number of blocks. We only move if size decreases or stays same (to explore).
    Actually, since we can only remove blocks to improve, and swaps keep size constant,
    the only way to improve is to find a swap that creates a redundancy, allowing a removal.
    
    Strategy:
    - Try to swap block i with a candidate.
    - If valid, perform swap.
    - Try to remove any now-redundant block.
    - If size decreased, keep.
    - If size same, we might keep the swap to explore, or revert. 
      Keeping the swap allows different configurations.
      We will accept swaps that maintain validity, and then aggressively reduce.
      If reduction fails, we can revert or keep. Keeping might help future swaps.
      Let's keep valid swaps that don't increase size (impossible) and try to reduce.
      If reduction doesn't reduce size, we can revert to avoid random walk, OR keep to diversify.
      Given the small search space of valid swaps, random walk is okay if we eventually reduce.
      However, to be safe and efficient, let's only accept swaps that lead to a reduction,
      OR perform a "neutral" swap with some probability to escape plateaus.
      
    Let's refine:
    1. Try swap.
    2. If valid, apply.
    3. Reduce.
    4. If reduced, accept.
    5. If not reduced, revert swap (backtrack) to maintain "best effort" trajectory,
       UNLESS we are stuck and want to explore. 
       Since we don't have a cost function other than size, and size doesn't change on swap,
       the only signal is "did it enable a removal?".
       If not, reverting is safer for convergence, but keeping helps diversity.
       Let's try: Accept swap if it reduces size. If not, revert.
       To escape local minima, we rely on the random selection of candidates.
       
    Additional Idea: "Directed" swaps.
    Instead of random candidate, pick a candidate that covers the triple with the lowest coverage count (among those covered by block i).
    This tries to increase redundancy of rare triples? No, that decreases their count if we remove block i?
    Wait. If we swap block i (which covers rare triple X) with block j (which does not cover X),
    X becomes uncovered -> Invalid.
    So we can only swap if all triples in block i are covered by others (count > 1).
    If a triple has count 1, block i is "essential" for that triple.
    
    So, for block i:
    - Find triples in block i with count == 1. These MUST be in the new candidate.
    - Find triples in block i with count > 1. These CAN be dropped.
    - The new candidate must contain all "must" triples.
    - If "must" triples > t, and they don't fit in a k-block, no swap is possible for this block (unless we change the definition).
      Actually, if a block has multiple count-1 triples, the new block must contain all of them.
      If the union of "must" triples has size > k, no valid swap exists for this block.
    
    This is a much more constrained and directed search.
    Let's implement this logic.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    def get_state(blocks):
        triple_cover_count = defaultdict(int)
        block_triples = []
        for block in blocks:
            tris = list(itertools.combinations(block, t))
            block_triples.append(tris)
            for tri in tris:
                triple_cover_count[tri] += 1
        return triple_cover_count, block_triples

    triple_cover_count, block_triples = get_state(blocks)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        n_blocks = len(blocks)
        
        # Shuffle indices
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            original_block = blocks[i]
            original_triples = block_triples[i]
            
            # Identify "must" triples (count == 1)
            must_triples = [tri for tri in original_triples if triple_cover_count[tri] == 1]
            
            # If no must triples, any valid block that doesn't break other counts?
            # Actually, if no must triples, we can swap with ANY block.
            # But we want to find a swap that creates redundancy.
            
            # The new block MUST contain all points in the union of must_triples.
            # Let M be the set of points in must_triples.
            must_points = set()
            for tri in must_triples:
                must_points.update(tri)
                
            if len(must_points) > k:
                # Cannot form a block containing all must points
                continue
                
            # We need to choose (k - len(must_points)) points from the remaining (v - len(must_points)) points.
            # To maximize chance of creating redundancy, we should prefer points that are in "dense" areas?
            # Or just random?
            # Let's try a few random completions.
            
            found_better = False
            # Limit number of candidates to try per block to save time
            max_candidates = 5
            for _ in range(max_candidates):
                # Randomly select the remaining points
                available_points = [p for p in range(v) if p not in must_points]
                if len(available_points) < k - len(must_points):
                    continue
                extra_points = rng.sample(available_points, k - len(must_points))
                candidate = sorted(must_points | set(extra_points))
                
                if tuple(candidate) == tuple(original_block):
                    continue
                
                new_triples = list(itertools.combinations(candidate, t))
                old_triples_set = set(original_triples)
                new_triples_set = set(new_triples)
                
                # Check validity:
                # 1. All must_triples must be in new_triples (guaranteed by construction)
                # 2. All other old_triples (count > 1) that are NOT in new_triples must have count > 1
                #    (i.e., covered by others)
                
                valid = True
                # Check lost triples
                for tri in old_triples_set:
                    if tri not in new_triples_set:
                        if triple_cover_count[tri] == 1:
                            # This triple is only covered by block i, and we are removing it.
                            # But wait, if it's in must_triples, it's in new_triples.
                            # So if it's lost, it wasn't a must triple, so count was > 1.
                            # So it should be covered by others.
                            # The check `triple_cover_count[tri] == 1` would be false here if logic holds.
                            # Let's be safe.
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                # Perform Swap
                for tri in original_triples:
                    triple_cover_count[tri] -= 1
                for tri in new_triples:
                    triple_cover_count[tri] += 1
                    
                blocks[i] = candidate
                block_triples[i] = new_triples
                
                # Try to reduce
                new_blocks = _reduce_covering(blocks, v, k, t)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
                    found_better = True
                    # Recalculate state
                    triple_cover_count, block_triples = get_state(blocks)
                    break # Restart loop with new size
                else:
                    # Revert
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    for tri in original_triples:
                        triple_cover_count[tri] += 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples
                    
            if found_better:
                break # Restart outer loop

    return blocks
=======
def _local_search_sa(blocks, v, k, t, rng, time_limit):
    """
    Local search that tries to swap blocks to create redundancies, enabling removals.
    Key improvement: smarter candidate generation and more aggressive reduction attempts.
    """
    start_time = time.time()
    blocks = [list(b) for b in blocks]
    
    def get_state(blocks):
        triple_cover_count = defaultdict(int)
        block_triples = []
        for block in blocks:
            tris = list(itertools.combinations(block, t))
            block_triples.append(tris)
            for tri in tris:
                triple_cover_count[tri] += 1
        return triple_cover_count, block_triples

    triple_cover_count, block_triples = get_state(blocks)
    
    improved = True
    stall_counter = 0
    max_stall = 200
    
    while improved and time.time() < time_limit and stall_counter < max_stall:
        improved = False
        n_blocks = len(blocks)
        
        # Shuffle indices
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            original_block = blocks[i]
            original_triples = block_triples[i]
            
            # Identify "must" triples (count == 1)
            must_triples = [tri for tri in original_triples if triple_cover_count[tri] == 1]
            
            # The new block MUST contain all points in the union of must_triples.
            must_points = set()
            for tri in must_triples:
                must_points.update(tri)
                
            if len(must_points) > k:
                continue
                
            found_better = False
            # Try multiple candidate strategies
            max_candidates = 10
            for _ in range(max_candidates):
                available_points = [p for p in range(v) if p not in must_points]
                if len(available_points) < k - len(must_points):
                    continue
                
                # Strategy: prefer points that are in blocks covering many triples with low count
                # This increases the chance of creating new redundancies
                if rng.random() < 0.5:
                    # Random selection
                    extra_points = rng.sample(available_points, k - len(must_points))
                else:
                    # Heuristic: prefer points that appear in blocks with low average triple count
                    point_scores = defaultdict(float)
                    for p in available_points:
                        # Count how many low-count triples contain p
                        for tri in itertools.combinations(range(v), t):
                            if p in tri and triple_cover_count[tri] <= 2:
                                point_scores[p] += 1
                        # Add some randomness
                        point_scores[p] += rng.random() * 0.1
                    sorted_points = sorted(available_points, key=lambda p: point_scores[p], reverse=True)
                    extra_points = sorted_points[:k - len(must_points)]
                
                candidate = sorted(must_points | set(extra_points))
                
                if tuple(candidate) == tuple(original_block):
                    continue
                
                new_triples = list(itertools.combinations(candidate, t))
                old_triples_set = set(original_triples)
                new_triples_set = set(new_triples)
                
                # Check validity
                valid = True
                for tri in old_triples_set:
                    if tri not in new_triples_set:
                        if triple_cover_count[tri] == 1:
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                # Perform Swap
                for tri in original_triples:
                    triple_cover_count[tri] -= 1
                for tri in new_triples:
                    triple_cover_count[tri] += 1
                    
                blocks[i] = candidate
                block_triples[i] = new_triples
                
                # Try to reduce
                new_blocks = _reduce_covering(blocks, v, k, t)
                
                if len(new_blocks) < len(blocks):
                    blocks = new_blocks
                    improved = True
                    found_better = True
                    stall_counter = 0
                    triple_cover_count, block_triples = get_state(blocks)
                    break
                else:
                    # Revert
                    for tri in new_triples:
                        triple_cover_count[tri] -= 1
                    for tri in original_triples:
                        triple_cover_count[tri] += 1
                    blocks[i] = original_block
                    block_triples[i] = original_triples
                    
            if found_better:
                break
            else:
                stall_counter += 1

    return blocks
>>>>>>> REPLACE

