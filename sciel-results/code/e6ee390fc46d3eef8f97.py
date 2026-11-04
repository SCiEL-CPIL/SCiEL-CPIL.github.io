<<<<<<< SEARCH
    for centers_init, angles_init, ocx_init, ocy_init, otheta_init in configs:
        if time.time() > deadline - 20:
            break
        
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init
        )
        
        def outer_objective(v):
            if time.time() > deadline - 5:
                return 1e6
            ox, oy, oa = v[0], v[1], v[2]
            return compute_outer_side(sa_centers, sa_angles, ox, oy, oa)
        
        result = minimize(outer_objective, [sa_ocx, sa_ocy, sa_otheta], method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        pol_ocx, pol_ocy, pol_otheta = result.x
        pol_cost = result.fun
        
        verts = []
        for i in range(12):
            v = get_hex_vertices(sa_centers[i][0], sa_centers[i][1], 1.0, sa_angles[i])
            verts.append(v)
            
        valid = True
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and pol_cost < best_overall_cost:
            best_overall_cost = pol_cost
            best_centers = sa_centers
            best_angles = sa_angles
            best_ocx = pol_ocx
            best_ocy = pol_ocy
            best_otheta = pol_otheta
=======
    for centers_init, angles_init, ocx_init, ocy_init, otheta_init in configs:
        if time.time() > deadline - 25:
            break
        
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init
        )
        
        # Verify disjointness before polishing
        verts = []
        for i in range(12):
            v = get_hex_vertices(sa_centers[i][0], sa_centers[i][1], 1.0, sa_angles[i])
            verts.append(v)
        
        if not check_all_disjoint(verts):
            continue
        
        def outer_objective(v):
            if time.time() > deadline - 5:
                return 1e6
            ox, oy, oa = v[0], v[1], v[2]
            return compute_outer_side(sa_centers, sa_angles, ox, oy, oa)
        
        # Multiple restarts of Nelder-Mead from different starting points
        best_pol_cost = float('inf')
        best_pol_ocx, best_pol_ocy, best_pol_otheta = sa_ocx, sa_ocy, sa_otheta
        
        start_points = [
            [sa_ocx, sa_ocy, sa_otheta],
            [sa_ocx + 0.1, sa_ocy, sa_otheta + 5],
            [sa_ocx - 0.1, sa_ocy + 0.1, sa_otheta - 5],
            [sa_ocx, sa_ocy - 0.1, sa_otheta + 10],
        ]
        
        for sp in start_points:
            if time.time() > deadline - 5:
                break
            result = minimize(outer_objective, sp, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            
            if result.fun < best_pol_cost:
                best_pol_cost = result.fun
                best_pol_ocx, best_pol_ocy, best_pol_otheta = result.x
        
        if best_pol_cost < best_overall_cost:
            best_overall_cost = best_pol_cost
            best_centers = sa_centers
            best_angles = sa_angles
            best_ocx = best_pol_ocx
            best_ocy = best_pol_ocy
            best_otheta = best_pol_otheta
>>>>>>> REPLACE

