<<<<<<< SEARCH
    # 5. Final Validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]

    return best_routes
=======
    # 6. Final Validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]

    return best_routes
>>>>>>> REPLACE
