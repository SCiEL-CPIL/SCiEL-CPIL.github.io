<<<<<<< SEARCH
def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(all_subsets)))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                uncovered.discard(idx)
                
    return blocks
=======
def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(all_subsets)))
    blocks = []
    
    # Precompute point frequencies for scarcity weighting
    point_freq = [0] * v
    for s in all_subsets:
        for p in s:
            point_freq[p] += 1
    
    while uncovered:
        # Pick an uncovered t-subset with highest scarcity weight
        best_seed_idx = None
        best_seed_weight = -1
        # Sample up to 20 uncovered subsets to find a good seed
        sample_size = min(20, len(uncovered))
        sample = rng.sample(uncovered, sample_size) if len(uncovered) > sample_size else list(uncovered)
        for idx in sample:
            seed = all_subsets[idx]
            weight = sum(point_freq[p] for p in seed)  # lower is better (more scarce)
            if weight < best_seed_weight:
                best_seed_weight = weight
                best_seed_idx = idx
        
        seed = all_subsets[best_seed_idx]
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage, with scarcity bias
        while len(block) < k:
            best_x = None
            best_score = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if idx in uncovered:
                            gain += 1
                # Score = coverage gain + small scarcity bonus
                scarcity_bonus = -0.01 * point_freq[x]
                score = gain + scarcity_bonus
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                uncovered.discard(idx)
                
    return blocks
>>>>>>> REPLACE
