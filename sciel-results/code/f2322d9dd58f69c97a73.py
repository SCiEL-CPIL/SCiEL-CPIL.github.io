    import numpy as np
    from scipy.optimize import minimize

    N = 11

    # EVOLVE-BLOCK-START
    def construct():
        def hex_vertices(cx, cy, angle, s=1.0):
            angles = angle + np.arange(6) * np.pi / 3
            return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

        def compute_S_smooth(vertices_list, ocx, ocy, oangle, beta=20.0):
            cos_a = np.cos(oangle)
            sin_a = np.sin(oangle)
            apothem = np.sqrt(3) / 2
            
            # Collect all projections
            projs = []
            for verts in vertices_list:
                for v in verts:
                    lx = v[0] - ocx
                    ly = v[1] - ocy
                    rx = lx * cos_a + ly * sin_a
                    ry = -lx * sin_a + ly * cos_a
                    for k in range(6):
                        theta_n = np.pi / 6 + k * np.pi / 3
                        proj = abs(rx * np.cos(theta_n) + ry * np.sin(theta_n))
                        projs.append(proj)
            
            projs = np.array(projs)
            # Soft max
            # S = max(projs) / apothem
            # Smooth: (1/beta) * log(sum(exp(beta * projs))) / apothem
            # To avoid overflow, subtract max
            m = np.max(projs)
            soft_max = m + np.log(np.sum(np.exp(beta * (projs - m)))) / beta
            return soft_max / apothem

        def objective_inner(params, ocx, ocy, oangle):
            n = len(params) // 3
            vertices_list = []
            centers = []
            for i in range(n):
                cx = params[3 * i]
                cy = params[3 * i + 1]
                angle = params[3 * i + 2]
                verts = hex_vertices(cx, cy, angle)
                vertices_list.append(verts)
                centers.append([cx, cy])
            
            S = compute_S_smooth(vertices_list, ocx, ocy, oangle)
            
            # Overlap penalty
            penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    c1 = centers[i]
                    c2 = centers[j]
                    dist = np.linalg.norm(np.array(c1) - np.array(c2))
                    # Safe distance
                    d_min = 1.75 # Slightly above sqrt(3)
                    if dist < d_min + 0.5:
                        # Smooth penalty
                        viol = d_min - dist
                        if viol > 0:
                            penalty += 100.0 * viol * viol
            
            return S + penalty

        def find_best_container(vertices_list, init_params=None):
            if init_params is None:
                init_params = [0.0, 0.0, 0.0]
            
            def cont_obj(params_c):
                ocx, ocy, oangle = params_c
                return compute_S_smooth(vertices_list, ocx, ocy, oangle, beta=50.0)
            
            res = minimize(cont_obj, init_params, method='BFGS', options={'maxiter': 100})
            return res.x, res.fun

        # Initial guess: Hexagonal lattice
        # Generate 11 points in a lattice
        lattice_points = []
        # Center
        lattice_points.append([0, 0])
        # Ring 1
        for k in range(6):
            theta = k * np.pi / 3
            cx = np.sqrt(3) * np.cos(theta)
            cy = np.sqrt(3) * np.sin(theta)
            lattice_points.append([cx, cy])
        # Ring 2 (need 4 more)
        # Pick 4 from ring 2 to be symmetric?
        # Ring 2 has 12 points.
        # Let's pick angles 0, 60, 120, 180? No, that's not symmetric.
        # Let's pick 0, 90? Hexagonal lattice doesn't have 90 deg symmetry.
        # Let's just take the first 4: 0, 60, 120, 180?
        # Actually, for a "compact" shape, we should fill the gaps.
        # Let's just take 4 consecutive ones?
        # Or better, let the optimizer figure it out.
        # Start with 4 points at distance 2*sqrt(3) at angles 0, pi/3, 2*pi/3, pi.
        for k in range(4):
            theta = k * np.pi / 3
            cx = 2 * np.sqrt(3) * np.cos(theta)
            cy = 2 * np.sqrt(3) * np.sin(theta)
            lattice_points.append([cx, cy])
        
        params = np.zeros(3 * N)
        for i, (cx, cy) in enumerate(lattice_points):
            params[3 * i] = cx
            params[3 * i + 1] = cy
            params[3 * i + 2] = 0.0

        # Optimize
        best_score = np.inf
        best_params = params
        best_container = [0.0, 0.0, 0.0]
        
        # Multi-start
        np.random.seed(42)
        for i in range(5):
            p = params.copy()
            if i > 0:
                p += np.random.normal(0, 0.1, size=p.size)
            
            # Inner optimization
            res_in = minimize(objective_inner, p, args=(0,0,0), method='BFGS', options={'maxiter': 200, 'maxfun': 500})
            
            # Extract vertices
            vertices_list = []
            for j in range(N):
                cx = res_in.x[3 * j]
                cy = res_in.x[3 * j + 1]
                angle = res_in.x[3 * j + 2]
                vertices_list.append(hex_vertices(cx, cy, angle))
            
            # Container optimization
            cont_params, S_val = find_best_container(vertices_list, init_params=[0,0,0])
            
            # Check if this is better
            # Recalculate S with hard max for true score
            def compute_S_hard(vertices_list, ocx, ocy, oangle):
                cos_a = np.cos(oangle)
                sin_a = np.sin(oangle)
                apothem = np.sqrt(3) / 2
                S = 0.0
                for verts in vertices_list:
                    for v in verts:
                        lx = v[0] - ocx
                        ly = v[1] - ocy
                        rx = lx * cos_a + ly * sin_a
                        ry = -lx * sin_a + ly * cos_a
                        for k in range(6):
                            theta_n = np.pi / 6 + k * np.pi / 3
                            proj = abs(rx * np.cos(theta_n) + ry * np.sin(theta_n))
                            S = max(S, proj)
                return S / apothem

            true_S = compute_S_hard(vertices_list, cont_params[0], cont_params[1], cont_params[2])
            
            # Verify overlap
            # Simple SAT check
            def sat_overlap(v1, v2):
                polys = [v1, v2]
                for poly in polys:
                    n = len(poly)
                    for i in range(n):
                        p1 = poly[i]
                        p2 = poly[(i + 1) % n]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        proj1 = v1 @ normal
                        proj2 = v2 @ normal
                        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                            return False
                return True

            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid and true_S < best_score:
                best_score = true_S
                best_params = res_in.x
                best_container = cont_params

        # Extract final
        inner = []
        vertices_list = []
        for i in range(N):
            cx = best_params[3 * i]
            cy = best_params[3 * i + 1]
            angle = best_params[3 * i + 2]
            inner.append([cx, cy, np.degrees(angle)])
            vertices_list.append(hex_vertices(cx, cy, angle))
        
        # Final container refinement
        cont_params, S = find_best_container(vertices_list, init_params=best_container)
        ocx, ocy, oangle = cont_params
        oangle_deg = np.degrees(oangle)
        
        return inner, (ocx, ocy), S, oangle_deg
    # EVOLVE-BLOCK-END
    