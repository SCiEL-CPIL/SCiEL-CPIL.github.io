<<<<<<< SEARCH
    # Evaluate and sort
    scored = []
    for mats, thicks in initial_solutions:
        score = _evaluate(mats, thicks, wls, n_amb, n_sub, materials_dict)
        scored.append((score, mats, thicks))
    scored.sort(key=lambda x: x[0])
    
    best_score = scored[0][0]
    best_materials = list(scored[0][1])
    best_thicknesses = list(scored[0][2])
    
    # SA on top solutions with different seeds
    rng = random.Random(999)
    n_sa_runs = min(6, len(scored))
    for idx in range(n_sa_runs):
        if time.time() > deadline - 20:
            break
        score, mats, thicks = scored[idx]
        sa_seed = idx * 7 + 13
        sa_rng = random.Random(sa_seed)
        
        sa_mats, sa_thicks = _sa_optimize(
            list(mats), list(thicks), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 15, sa_rng,
            T0=3.0, T1=0.0005, n_steps=40000)
        
        if _is_valid(sa_mats, sa_thicks, problem):
            sa_score = _evaluate(sa_mats, sa_thicks, wls, n_amb, n_sub, materials_dict)
            if sa_score < best_score:
                best_score = sa_score
                best_materials = list(sa_mats)
                best_thicknesses = list(sa_thicks)
=======
    # Evaluate and sort
    scored = []
    for mats, thicks in initial_solutions:
        score = _evaluate(mats, thicks, wls, n_amb, n_sub, materials_dict)
        scored.append((score, mats, thicks))
    scored.sort(key=lambda x: x[0])
    
    best_score = scored[0][0]
    best_materials = list(scored[0][1])
    best_thicknesses = list(scored[0][2])
    
    # Quick local refinement on top initial solutions to give SA better starts
    n_initial_refine = min(3, len(scored))
    for idx in range(n_initial_refine):
        if time.time() > deadline - 30: # Reserve time for SA
            break
        score, mats, thicks = scored[idx]
        try:
            sp_mats, sp_thicks = _scipy_local_opt(
                list(mats), list(thicks), problem, wls, materials_dict,
                n_amb, n_sub, deadline - 25)
            if _is_valid(sp_mats, sp_thicks, problem):
                sp_score = _evaluate(sp_mats, sp_thicks, wls, n_amb, n_sub, materials_dict)
                if sp_score < best_score:
                    best_score = sp_score
                    best_materials = list(sp_mats)
                    best_thicknesses = list(sp_thicks)
                # Update the scored list with the refined solution
                scored[idx] = (sp_score, sp_mats, sp_thicks)
        except Exception:
            pass
    
    # Re-sort after refinement
    scored.sort(key=lambda x: x[0])
    
    # SA on top solutions with different seeds
    rng = random.Random(999)
    n_sa_runs = min(4, len(scored))
    for idx in range(n_sa_runs):
        if time.time() > deadline - 10:
            break
        score, mats, thicks = scored[idx]
        sa_seed = idx * 7 + 13
        sa_rng = random.Random(sa_seed)
        
        sa_mats, sa_thicks = _sa_optimize(
            list(mats), list(thicks), problem, wls, materials_dict,
            n_amb, n_sub, deadline - 5, sa_rng,
            T0=3.0, T1=0.0005, n_steps=40000)
        
        if _is_valid(sa_mats, sa_thicks, problem):
            sa_score = _evaluate(sa_mats, sa_thicks, wls, n_amb, n_sub, materials_dict)
            if sa_score < best_score:
                best_score = sa_score
                best_materials = list(sa_mats)
                best_thicknesses = list(sa_thicks)
>>>>>>> REPLACE
