<<<<<<< SEARCH
    # Phase 4: Final local search on best found
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iterations=200)
        if len(improved_set) > best_size and is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set

    # Final validation
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(66, 101))
=======
    # Phase 4: Final local search on best found
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iterations=300)
        if len(improved_set) > best_size and is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set

    # Phase 5: Try a few more random restarts if time permits
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            rand_order = list(range(1, 101))
            random.shuffle(rand_order)
            s, _ = greedy_construct(rand_order)
            if len(s) > best_size:
                improved_s = local_search(s, max_iterations=100)
                if len(improved_s) > best_size and is_valid(improved_s):
                    best_size = len(improved_s)
                    best_set = improved_s

    # Final validation
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(66, 101))
>>>>>>> REPLACE

