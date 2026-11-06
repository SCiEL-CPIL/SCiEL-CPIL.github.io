<<<<<<< SEARCH
    for target_n in range(8, 29):
        if time_left() < 10:
            break
        
        n = target_n
=======
    start_n = best_n + 1 if best_n >= 26 else 8
    for target_n in range(start_n, 29):
        if time_left() < 10:
            break
        
        n = target_n
>>>>>>> REPLACE

Now let me also improve the search efficiency by reducing unnecessary work and improving the simulated annealing parameters:

