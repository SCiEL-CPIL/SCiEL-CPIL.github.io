def _construct_k3_6_1(q, n, radius, deadline):
    """
    Specialized construction for q=3, n=6, radius=1.
    Starts with the 81-word direct product.
    Uses a scan-based Add-Remove local search.
    """
    # Start with tetracode product (81 words)
    codewords = _tetracode(q, n, radius)
    
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute balls for all 729 possible words as numpy arrays of flat indices
    balls_cache = {}
    for w in all_words:
        idxs = []
        idxs.append(int(np.ravel_multi_index(w, (q,)*n)))
        for pos in range(n):
            orig_val = w[pos]
            for v in range(q):
                if v == orig_val:
                    continue
                w2 = list(w)
                w2[pos] = v
                idxs.append(int(np.ravel_multi_index(tuple(w2), (q,)*n)))
        balls_cache[w] = np.array(idxs, dtype=np.int32)
        
    # Precompute set of ball indices for fast subset check?
    # Or just use np.isin?
    # Let's use a set for Ball(w) for O(1) membership.
    ball_sets = {w: set(idxs.tolist()) for w, idxs in balls_cache.items()}
    
    # Initial coverage count
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        idxs = balls_cache[tuple(cw)]
        cover_count[idxs] += 1
        
    # Helper to compute unique words for a codeword
    def _get_unique_words(cw_tuple):
        idxs = balls_cache[cw_tuple]
        unique_mask = cover_count[idxs] == 1
        return idxs[unique_mask]

    # Local Search
    no_improve_streak = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        if no_improve_streak > max_no_improve:
            break
            
        # Compute unique words for all current codewords
        unique_words_map = {}
        for i, cw in enumerate(codewords):
            unique_words_map[i] = _get_unique_words(tuple(cw))
            
        best_gain = 0
        best_w = None
        best_removals = []
        
        current_set = set(tuple(cw) for cw in codewords)
        
        # Scan all words w not in current_set
        for w in all_words:
            if w in current_set:
                continue
                
            w_ball_set = ball_sets[w]
            
            removable_indices = []
            for i, cw in enumerate(codewords):
                u_w = unique_words_map[i]
                if len(u_w) == 0:
                    # Already removable without adding w?
                    # We should have removed it in a previous step.
                    # But maybe it became removable due to previous changes?
                    # We recompute unique_words_map at start of loop.
                    # So if u_w is empty, it is removable.
                    removable_indices.append(i)
                else:
                    # Check if u_w is subset of w_ball_set
                    # Since u_w is small, check each element
                    is_subset = True
                    for idx in u_w:
                        if idx not in w_ball_set:
                            is_subset = False
                            break
                    if is_subset:
                        removable_indices.append(i)
                        
            gain = len(removable_indices) - 1 # Add 1, remove len(removable_indices)
            
            if gain > best_gain:
                best_gain = gain
                best_w = w
                best_removals = removable_indices
                
        if best_w is not None and best_gain > 0:
            # Perform the change
            # Add best_w
            idxs_w = balls_cache[best_w]
            cover_count[idxs_w] += 1
            codewords.append(list(best_w))
            
            # Remove best_removals
            # Remove from end to start to keep indices valid
            best_removals.sort(reverse=True)
            for i in best_removals:
                cw_tuple = tuple(codewords[i])
                idxs_cw = balls_cache[cw_tuple]
                cover_count[idxs_cw] -= 1
                codewords.pop(i)
                
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            
    return codewords
