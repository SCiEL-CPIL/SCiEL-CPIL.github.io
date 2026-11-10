<<<<<<< SEARCH
    best_params = _get_initial_guess()
    best_s, best_valid, _ = _check_separation_and_containment(
        best_params[:24].reshape(12, 2), 
        best_params[24:36], 
        best_params[36]
    )
=======
    best_params = np.asarray(_get_initial_guess()).flatten()
    best_s, best_valid, _ = _check_separation_and_containment(
        best_params[:24].reshape(12, 2), 
        best_params[24:36], 
        float(best_params[36])
    )
>>>>>>> REPLACE
