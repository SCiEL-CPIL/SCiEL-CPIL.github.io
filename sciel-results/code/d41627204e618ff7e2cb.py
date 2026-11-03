<<<<<<< SEARCH
    # Polish the best candidate first, then the next-best distinct ones
    seen = []
    for s, U in scored:
        if time.time() > deadline - 15:
            break
        key = tuple(U)
        if key in seen:
            continue
        seen.append(key)
        if s < best_score - 0.03:
            continue
        iters = 8000 if len(seen) == 1 else 3000
        improved_U, improved_score = _local_search(U, max_iters=iters, deadline=deadline - 15)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
=======
    # Polish the best candidate first, then the next-best distinct ones
    seen = []
    for s, U in scored:
        if time.time() > deadline - 15:
            break
        key = tuple(U)
        if key in seen:
            continue
        seen.append(key)
        if s < best_score - 0.03:
            continue
        iters = 10000 if len(seen) == 1 else 5000
        improved_U, improved_score = _local_search(U, max_iters=iters, deadline=deadline - 15)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
>>>>>>> REPLACE

