# Inside the restart loop:
        if len(candidate) >= len(best_set):
             # Run local search on candidate
             local_best = local_search(candidate)
             if len(local_best) > len(best_set):
                 best_set = local_best
