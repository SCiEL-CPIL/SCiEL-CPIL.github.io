def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time

    # Precompute unit hexagon vertices and normals for fast SAT
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    unit_normals_0 = []
    for i in range(6):
        v1 = unit_verts_0[i]
        v2 = unit_verts_0[(i+1)%6]
        edge = v2 - v1
        nx, ny = edge[1], -edge[0]
        norm = np.sqrt(nx*nx + ny*ny)
        unit_normals_0.append([nx/norm, ny/norm])
    unit_normals_0 = np.array(unit_normals_0)

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_hex_normals(angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normals = unit_normals_0 @ rot.T
        return normals

    def sat_gap(poly1, poly2, normals1, normals2):
        min_gap = float('inf')
        for i in range(6):
            axis = normals1[i]
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        for i in range(6):
            axis = normals2[i]
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        return min_gap

    def calc_min_outer_side_fast(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        return max_proj * 2.0 / math.sqrt(3)

    def is_valid_fast(positions, angles, tol=1e-6):
        polys = []
        norms = []
        for i in range(N):
            polys.append(get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]))
            norms.append(get_hex_normals(angles[i]))
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j], norms[i], norms[j]) < -tol:
                    return False
        return True

    def compute_min_outer_side_fast(inner_positions, inner_angles, outer_center, outer_angle, lo=2.0, hi=20.0, iters=25):
        for _ in range(iters):
            mid = (lo + hi) / 2
            if check_disjoint_fast(inner_positions, inner_angles, mid, outer_center, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi

    def check_disjoint_fast(positions, angles, side, outer_center, outer_angle, tol=1e-6):
        theta_rad = math.radians(outer_angle)
        outer_normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            outer_normals.append([math.cos(ang), math.sin(ang)])
        outer_normals = np.array(outer_normals)
        max_proj_allowed = side * math.sqrt(3) / 2.0
        
        polys = []
        norms = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            polys.append(verts)
            norms.append(get_hex_normals(ang))
            projs = verts @ outer_normals.T
            if np.max(projs) > max_proj_allowed + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j], norms[i], norms[j]) < -tol:
                    return False
        return True

    # --- PHASE 1: INITIALIZE WITH A KNOWN VALID CONFIGURATION ---
    # Hexagonal lattice packing of 12 hexagons
    # Centers: (0,0), 6 at dist 2, 5 at dist 2*sqrt(3) or similar
    # Let's use a standard hexagonal cluster.
    # Radius 1: 1 center, 6 neighbors. Total 7.
    # Radius 2: 12 more neighbors. Total 19.
    # We need 12. Take center + 6 + 5 from the next ring.
    
    sqrt3 = math.sqrt(3)
    v1 = np.array([1.5, sqrt3/2])
    v2 = np.array([0, sqrt3])
    
    lattice_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            lattice_pts.append(p)
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    sorted_idx = np.argsort(dists)
    base_points = lattice_pts[sorted_idx[:12]]
    
    # Center the cluster
    centroid = np.mean(base_points, axis=0)
    base_points = base_points - centroid
    
    # Initial angles: all 0
    init_positions = base_points.tolist()
    init_angles = [0.0] * 12
    init_outer_angle = 0.0
    
    # Calculate initial side
    init_side = calc_min_outer_side_fast(init_positions, init_angles, init_outer_angle)
    # Add a small buffer to ensure validity if numerical errors occur
    init_side += 0.01
    
    best_positions = [p.copy() for p in init_positions]
    best_angles = init_angles.copy()
    best_outer_angle = init_outer_angle
    best_side = init_side
    
    # --- PHASE 2: SIMULATED ANNEALING ---
    # Objective: Minimize side length.
    # Move: Perturb one hexagon's position and angle, or outer angle.
    # Accept: If valid and side improves, or with probability exp(-delta_side / T).
    
    T = 0.5
    T_min = 0.001
    cooling = 0.995
    max_iters = 5000
    
    current_positions = [p.copy() for p in best_positions]
    current_angles = best_angles.copy()
    current_outer_angle = best_outer_angle
    current_side = best_side
    
    for k in range(max_iters):
        if elapsed() > DEADLINE - 5:
            break
            
        # Generate a neighbor
        new_positions = [p.copy() for p in current_positions]
        new_angles = current_angles.copy()
        new_outer_angle = current_outer_angle
        
        # Randomly choose which parameter to perturb
        r = np.random.random()
        if r < 0.7:
            # Perturb position of a random hexagon
            idx = np.random.randint(N)
            delta_x = np.random.normal(0, 0.1)
            delta_y = np.random.normal(0, 0.1)
            new_positions[idx][0] += delta_x
            new_positions[idx][1] += delta_y
        elif r < 0.9:
            # Perturb angle of a random hexagon
            idx = np.random.randint(N)
            delta_a = np.random.normal(0, 10.0)
            new_angles[idx] = (new_angles[idx] + delta_a) % 360.0
        else:
            # Perturb outer angle
            delta_a = np.random.normal(0, 5.0)
            new_outer_angle = (new_outer_angle + delta_a) % 360.0
            
        # Check validity
        if not is_valid_fast(new_positions, new_angles, tol=1e-5):
            continue
            
        # Calculate new side
        new_side = calc_min_outer_side_fast(new_positions, new_angles, new_outer_angle)
        
        # Accept if better or with probability
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_outer_angle = new_outer_angle
            current_side = new_side
            
            if current_side < best_side:
                best_positions = [p.copy() for p in current_positions]
                best_angles = current_angles.copy()
                best_outer_angle = current_outer_angle
                best_side = current_side
        else:
            delta = current_side - new_side
            if delta < 0:
                delta = 0
            if np.random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_outer_angle = new_outer_angle
                current_side = new_side
                
        T = max(T * cooling, T_min)

    # --- PHASE 3: FINAL VALIDATION AND BINARY SEARCH ---
    # Ensure the best solution found is truly valid and find the tightest side
    
    if is_valid_fast(best_positions, best_angles, tol=1e-5):
        # Binary search for the exact minimum side for this configuration
        lo = best_side * 0.9
        hi = best_side * 1.1
        for _ in range(30):
            mid = (lo + hi) / 2
            if check_disjoint_fast(best_positions, best_angles, mid, (0,0), best_outer_angle):
                hi = mid
            else:
                lo = mid
        best_side = hi
    else:
        # If best is invalid (should not happen due to checks), revert to init
        best_positions = [p.copy() for p in init_positions]
        best_angles = init_angles.copy()
        best_outer_angle = init_outer_angle
        best_side = init_side
        
        # Final check on init
        if not is_valid_fast(best_positions, best_angles, tol=1e-5):
            # Extreme fallback: Line packing
            best_positions = [[i * 2.0, 0.0] for i in range(12)]
            best_angles = [0.0] * 12
            best_outer_angle = 0.0
            best_side = calc_min_outer_side_fast(best_positions, best_angles, best_outer_angle) + 0.1

    inner = [[p[0], p[1], a] for p, a in zip(best_positions, best_angles)]
    return inner, (0.0, 0.0), best_side, best_outer_angle
