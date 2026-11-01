<<<<<<< SEARCH
    # Phase 2: Row swap search on the best solution
    if time.time() < deadline - 12:
        swap_rows_best, swap_scope = row_swap_search(best_rows, budget=5.0)
        if swap_scope < best_scope:
            best_scope = swap_scope
            best_rows = swap_rows_best

    # Phase 3: Fine-grained mark annealing
    if time.time() < deadline - 8:
        anneal_rows, anneal_scope = mark_anneal(best_rows, best_scope, budget=6.0)
        if anneal_scope < best_scope:
            best_scope = anneal_scope
            best_rows = anneal_rows

    # Phase 4: Final targeted descent
    if time.time() < deadline - 5:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=3.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Phase 5: One more polish pass
    if time.time() < deadline - 3:
        polished = local_polish(best_rows, max_passes=4)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
=======
    # Phase 2: Row swap search on the best solution
    if time.time() < deadline - 12:
        swap_rows_best, swap_scope = row_swap_search(best_rows, budget=5.0)
        if swap_scope < best_scope:
            best_scope = swap_scope
            best_rows = swap_rows_best

    # Phase 3: Fine-grained mark annealing
    if time.time() < deadline - 8:
        anneal_rows, anneal_scope = mark_anneal(best_rows, best_scope, budget=6.0)
        if anneal_scope < best_scope:
            best_scope = anneal_scope
            best_rows = anneal_rows

    # Phase 4: Final targeted descent
    if time.time() < deadline - 5:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=3.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Phase 5: One more polish pass
    if time.time() < deadline - 3:
        polished = local_polish(best_rows, max_passes=6)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
>>>>>>> REPLACE
