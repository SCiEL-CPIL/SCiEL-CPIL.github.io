<<<<<<< SEARCH
    # 1. Lattice Init
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=8.0)
    update_best(refined_lat)
    track_good(refined_lat)
    
    # 1b. Hexagonal Init
    if time.time() < deadline - 60:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=6.0)
        update_best(refined_hex)
        track_good(refined_hex)
    
    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                cart = bary_to_cart(bary)
                return -calc_min_area_cartesian(cart)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=35,
                popsize=15,
                tol=1e-4,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
            track_good(refined_de)
        except Exception:
            pass
    
    # 3. Basin Hopping (manual loop for time safety)
    if time.time() < deadline - 10 and best_bary is not None:
        current_bary = best_bary.copy()
        current_val = best_val
        rng_bh = np.random.default_rng(42)
        
        while time.time() < deadline - 5:
            noise = rng_bh.normal(0, 0.2, current_bary.shape)
            new_bary = clamp_bary(current_bary + noise)
            
            refined = refine_local(new_bary, time_limit=min(3.0, deadline - 5 - time.time()))
            val = calc_min_area_cartesian(bary_to_cart(refined))
            
            if val > current_val:
                current_val = val
                current_bary = refined
                update_best(refined)
                track_good(refined)
=======
    # 1. Lattice Init
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=10.0)
    update_best(refined_lat)
    track_good(refined_lat)
    
    # 1b. Boundary Init (points on edges)
    if time.time() < deadline - 80:
        bnd_init = generate_boundary_init()
        refined_bnd = refine_local(bnd_init, time_limit=10.0)
        update_best(refined_bnd)
        track_good(refined_bnd)
    
    # 1c. Hexagonal Init
    if time.time() < deadline - 70:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=8.0)
        update_best(refined_hex)
        track_good(refined_hex)
    
    # 2. Multi-start random inits with coordinate ascent
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            rand_init = generate_random_init(seed=trial)
            remaining = deadline - 12 - time.time()
            if remaining < 3:
                break
            refined_rand = refine_local(rand_init, time_limit=min(8.0, remaining * 0.8))
            update_best(refined_rand)
            track_good(refined_rand)
    
    # 3. Perturbation-based basin hopping
    if time.time() < deadline - 10 and best_bary is not None:
        current_bary = best_bary.copy()
        current_val = best_val
        rng_bh = np.random.default_rng(42)
        
        while time.time() < deadline - 5:
            # Try different noise scales
            scale = 0.15
            noise = rng_bh.normal(0, scale, current_bary.shape)
            new_bary = clamp_bary(current_bary + noise)
            
            remaining = deadline - 5 - time.time()
            if remaining < 2:
                break
            refined = refine_local(new_bary, time_limit=min(4.0, remaining * 0.7))
            val = calc_min_area_cartesian(bary_to_cart(refined))
            
            if val > current_val:
                current_val = val
                current_bary = refined
                update_best(refined)
                track_good(refined)
>>>>>>> REPLACE
