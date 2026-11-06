while time:
    # best-improvement step over hot removals x all nonedges (or sampled)
    improved = best_improve()
    if not improved:
        # kick
        do random kicks
        if still not improved after several kicks: restart
    record best
    if max_book<=7: return
