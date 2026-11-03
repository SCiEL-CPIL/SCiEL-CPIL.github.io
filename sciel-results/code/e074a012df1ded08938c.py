<<<<<<< SEARCH
    # Try multiple random orderings and keep the best result
    best_selected = None
    best_count = float('inf')
    
    # Number of restarts based on remaining time
    # Each greedy run takes roughly: n_iterations * (n_blocks / 2) average scans
    # Let's try a few restarts
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create a random order for tie-breaking
        if restart == 0:
            # First run: natural order
            block_order = list(range(n_blocks))
        else:
            block_order = list(range(n_blocks))
            random.shuffle(block_order)
        
        selected, remaining = greedy_with_order(block_order)
        
        if remaining != 0:
            # Incomplete coverage, skip
            continue
        
        # Remove redundant blocks
        selected = remove_redundant(selected, block_masks, n_t)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = selected
        
        # If we've achieved a good solution, we can stop early
        if best_count <= 300:
            break
    
    if best_selected is None:
        # Fallback: use the first greedy result
        block_order = list(range(n_blocks))
        selected, remaining = greedy_with_order(block_order)
        best_selected = selected
    
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in best_selected]
=======
    # Try multiple random orderings and keep the best result
    best_selected = None
    best_count = float('inf')
    
    # Number of restarts based on remaining time
    # Each greedy run takes roughly: n_iterations * (n_blocks / 2) average scans
    # Let's try a few restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create a random order for tie-breaking
        if restart == 0:
            # First run: natural order
            block_order = list(range(n_blocks))
        elif restart == 1:
            # Second run: sorted by coverage count (most covered first)
            block_order = sorted(range(n_blocks), key=lambda i: t_coverage_counts[i], reverse=True)
        else:
            block_order = list(range(n_blocks))
            random.shuffle(block_order)
        
        selected, remaining = greedy_with_order(block_order)
        
        if remaining != 0:
            # Incomplete coverage, skip
            continue
        
        # Remove redundant blocks
        selected = remove_redundant(selected, block_masks, n_t)
        
        # Local search
        selected = local_search(selected, block_masks, n_t, deadline)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = selected
        
        # If we've achieved a good solution, we can stop early
        if best_count <= 300:
            break
    
    if best_selected is None:
        # Fallback: use the first greedy result
        block_order = list(range(n_blocks))
        selected, remaining = greedy_with_order(block_order)
        best_selected = selected
    
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in best_selected]
>>>>>>> REPLACE
