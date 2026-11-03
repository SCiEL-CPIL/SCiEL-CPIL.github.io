<<<<<<< SEARCH
    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Run restarts if time permits
    while time.time() < deadline - 10.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code
    
    # Final polish: run local search on best_code with remaining time
    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
=======
    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code)
    
    # Run restarts if time permits
    while time.time() < deadline - 15.0:
        cand_code = greedy_cover_randomized()
        cand_code = local_search_reduce(cand_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code
    
    # Iterated Local Search: perturb and repair on best_code
    while time.time() < deadline - 5.0:
        cand_code = perturb_and_repair(best_code)
        if len(cand_code) < len(best_code):
            best_code = cand_code
    
    # Final polish: run local search on best_code with remaining time
    best_code = local_search_reduce(best_code)
    
    # Convert indices back to words
    return [all_words[i] for i in best_code]
>>>>>>> REPLACE
