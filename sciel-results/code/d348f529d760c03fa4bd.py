    no_improve_count = 0
    # ... in loop
    if current_len < best_len:
        best_network = ...
        best_len = current_len
        no_improve_count = 0
    else:
        no_improve_count += 1
        
    if no_improve_count > 5000: # Threshold
        # Restart
        current_network = list(best_network)
        current_len = best_len
        T = 5.0 # Reheat
        no_improve_count = 0
    