def _repair(blocks, counts, seed=0):
    rng = random.Random(seed)
    blocks = list(blocks)
    counts = counts.copy()
    current_set = set(blocks)
    
    while True:
        uncovered_indices = np.nonzero(counts == 0)[0]
        if len(uncovered_indices) == 0:
            break
        t_idx = rng.choice(uncovered_indices)
        
        best_b = -1
        best_score = -1
        candidates = _blocks_containing_triple[t_idx]
        
        for b_idx in candidates:
            if b_idx in current_set:
                continue
            score = 0
            for u in _block_triples[b_idx]:
                if counts[u] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_b = b_idx
                if score == 35: # Max possible
                    break
                    
        if best_b == -1:
            # No new block can cover t_idx? Should not happen.
            break
            
        blocks.append(best_b)
        current_set.add(best_b)
        for u in _block_triples[best_b]:
            counts[u] += 1
            
    return blocks, counts
