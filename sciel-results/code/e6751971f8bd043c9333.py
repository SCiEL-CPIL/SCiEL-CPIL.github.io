    current_val = objective(best_result)
    # ...
    if result_lbfgs.fun < current_val:
         best_result = result_lbfgs.x
         current_val = result_lbfgs.fun
