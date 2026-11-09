<<<<<<< SEARCH
    # Multiple greedy initializations with different orderings
    n_restarts = 100
    for restart in range(n_restarts):
        if _deadline_guard():
            break
        
        if restart == 0:
            order = list(range(511, 0, -1))
        elif restart == 1:
            order = sorted(range(511, 0, -1), key=lambda a: (masks[a].bit_count(), -a))
        else:
            order = list(range(511, 0, -1))
            n_swaps = rng.randint(5, 150)
            for _ in range(n_swaps):
                i, j = rng.randint(0, 510), rng.randint(0, 510)
                order[i], order[j] = order[j], order[i]
        
        chosen = _greedy_select(masks, order)
        
        # Hill-climb to local optimum with add + swap moves
        chosen, score = _hill_climb(chosen, masks)
        
        # Final polish pass
        chosen, score = _polish(chosen, masks)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen[:]
=======
    # Multiple greedy initializations with different orderings
    n_restarts = 200
    for restart in range(n_restarts):
        if _deadline_guard():
            break
        
        if restart == 0:
            order = list(range(511, 0, -1))
        elif restart == 1:
            order = sorted(range(511, 0, -1), key=lambda a: (masks[a].bit_count(), -a))
        elif restart == 2:
            # Sort by weight / bit_count to prioritize high-value low-conflict numbers
            order = sorted(range(511, 0, -1), key=lambda a: (masks[a].bit_count() / max(1, 1.0/(512-a)), -a))
        elif restart == 3:
            # Reverse: prioritize low-weight first, then high
            order = sorted(range(511, 0, -1), key=lambda a: (-1.0/(512-a), masks[a].bit_count()))
        else:
            order = list(range(511, 0, -1))
            n_swaps = rng.randint(5, 150)
            for _ in range(n_swaps):
                i, j = rng.randint(0, 510), rng.randint(0, 510)
                order[i], order[j] = order[j], order[i]
        
        chosen = _greedy_select(masks, order)
        
        # Hill-climb to local optimum with add + 1-1 swap + 2-1 swap moves
        chosen, score = _hill_climb(chosen, masks)
        
        # Final polish pass
        chosen, score = _polish(chosen, masks)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen[:]
>>>>>>> REPLACE
