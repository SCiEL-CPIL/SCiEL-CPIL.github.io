<<<<<<< SEARCH
        ths = list(init_ths)
        ths = constrain_thicknesses(ths)
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        # Nelder-Mead simplex
        step = 20.0
        alpha = 1.0
=======
        ths = list(init_ths)
        ths = constrain_thicknesses(ths)
        best_score = _mean_reflectance(materials_pattern, ths, problem)
        best_ths = list(ths)
        n = len(ths)
        
        # Nelder-Mead simplex
        step = 30.0
        alpha = 1.0
>>>>>>> REPLACE
