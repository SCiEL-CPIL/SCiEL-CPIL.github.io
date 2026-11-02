<<<<<<< SEARCH
    # Phase 3: Local search with more rounds
    for _ in range(30):
        if time.time() > deadline - 8.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_rounds=15)
        if improved:
            best_solution = optimize_redundancy_v2(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            best_solution, improved2 = local_search_2exchange(best_solution)
            if improved2:
                best_solution = optimize_redundancy_v2(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                break

    # Phase 4: SA polish
    if time.time() < deadline - 5.0:
        polished = sa_polish(best_solution, max_iters=4000)
        polished = optimize_redundancy_v2(polished)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(best_solution)
=======
    # Phase 3: Local search with more rounds and diverse moves
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_rounds=20)
        if improved:
            best_solution = optimize_redundancy_v2(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # Try 2-exchange with more attempts
            best_solution, improved2 = local_search_2exchange(best_solution, max_attempts=30)
            if improved2:
                best_solution = optimize_redundancy_v2(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                # Try 3-exchange
                best_solution, improved3 = local_search_3exchange(best_solution, max_attempts=10)
                if improved3:
                    best_solution = optimize_redundancy_v2(best_solution)
                    if len(best_solution) < best_len:
                        best_len = len(best_solution)
                else:
                    break

    # Phase 4: SA polish with more iterations
    if time.time() < deadline - 6.0:
        polished = sa_polish(best_solution, max_iters=10000)
        polished = optimize_redundancy_v2(polished)
        if len(polished) < best_len:
            best_solution = polished
            best_len = len(best_solution)
>>>>>>> REPLACE
