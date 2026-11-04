"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Precompute unit hexagon vertex offsets for angles 0..350 in 5-degree steps
    # For a regular hexagon of side 1, vertices at angle a are at:
    #   (cos(a + k*pi/3), sin(a + k*pi/3)) for k=0..5
    ANGLE_STEPS = 36  # 0, 5, 10, ..., 175 degrees (180 is same as 0 for hex)
    angle_offsets_deg = np.arange(ANGLE_STEPS) * 5.0
    # Vertex offsets: shape (ANGLE_STEPS, 6, 2)
    _k = np.arange(6)
    _theta = angle_offsets_deg[:, None] * (math.pi / 180.0) + _k[None, :] * (math.pi / 3.0)
    _voff = np.stack([np.cos(_theta), np.sin(_theta)], axis=-1)  # (36, 6, 2)

    # Outer hexagon edge normals for outer_angle=0: at 30, 90, 150, 210, 270, 330 degrees
    _norm_angles = np.array([30, 90, 150, 210, 270, 330]) * (math.pi / 180.0)
    _normals = np.stack([np.cos(_norm_angles), np.sin(_norm_angles)], axis=-1)  # (6, 2)
    _apothem = math.sqrt(3) / 2.0

    def _params_to_positions(params):
        """Convert 3N-param vector to (N,3) array of (x, y, angle_rad)."""
        return params.reshape(N, 3)

    def _smooth_objective(params, beta=8.0):
        """
        Smooth objective: t (epigraph for outer side) + soft-plus overlap penalty.
        params: 3N vector (x_i, y_i, theta_i in radians) for i=0..N-1.
        We optimize over 3N+1 variables: [x0,y0,th0, ..., xN-1,yN-1,thN-1, t]
        """
        pos = params[:3*N].reshape(N, 3)
        t = params[3*N]
        
        xs = pos[:, 0]
        ys = pos[:, 1]
        ths = pos[:, 2]
        
        # --- Outer side constraint (epigraph) ---
        # For each inner hex i, each vertex v, each normal n:
        #   (xs[i] + cos(th_i + k*pi/3)) * n_x + (ys[i] + sin(th_i + k*pi/3)) * n_y <= t * apothem
        # We need: t >= max over all (i, k, n) of dot(pos_i + voff_i[k], n) / apothem
        # Compute all vertex positions: (N, 6, 2)
        k_idx = np.arange(6)
        theta_all = ths[:, None] + k_idx[None, :] * (math.pi / 3.0)  # (N, 6)
        voffs = np.stack([np.cos(theta_all), np.sin(theta_all)], axis=-1)  # (N, 6, 2)
        verts = np.stack([xs[:, None, None] + voffs[..., 0], ys[:, None, None] + voffs[..., 1]], axis=-1)  # (N, 6, 2)
        
        # Projections onto all 6 normals: (N, 6, 6)
        # verts[..., :, a] @ normals[b, a]
        projs = np.einsum('nka,mb->nkmb', verts, _normals)  # (N, 6, 6)
        max_proj = projs.max()
        outer_side_needed = max_proj / _apothem
        
        # Epigraph penalty: (t - outer_side_needed)^2 if t < outer_side_needed
        epigraph_pen = np.maximum(0.0, outer_side_needed - t) ** 2
        
        # --- Overlap penalty (smooth surrogate) ---
        # For each pair (i, j), compute minimum distance between edges.
        # Smooth surrogate: softplus(-min_edge_dist / beta) * beta
        # We use a simplified but differentiable proxy:
        # For two hexagons with centers (x1,y1) and (x2,y2) and angles th1, th2,
        # the minimum edge-edge distance is complex. Instead, we use:
        #   d = center_distance - 2 * apothem_hex  (approximate, valid for aligned hexes)
        # But this ignores rotation. A better smooth proxy:
        #   For each pair, compute the 6x6 set of edge-vertex distances and take min.
        # To keep it fast, we use the center-distance minus sum of inradii as a
        # differentiable lower bound on separation, plus a rotation-aware correction.
        
        # Vectorized pair computation
        # center distances: (N, N)
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        center_dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        
        # For two unit hexagons with relative angle delta_th, the minimum
        # edge-to-edge distance when centers are at distance d is:
        #   d - 2 * r_in(delta_th)  where r_in depends on relative rotation
        # For simplicity and differentiability, use:
        #   separation = center_dist - 2 * _apothem  (conservative, ignores rotation benefit)
        # This is a lower bound on true separation, so penalty is an overestimate
        # but still differentiable and drives hexagons apart.
        
        # Actually, let's use a better proxy: the inradius of a hexagon is sqrt(3)/2
        # regardless of rotation. The minimum distance between two hexagons with
        # center separation d is at least d - 2*sqrt(3)/2 = d - sqrt(3).
        # This is tight when edges face each other. For other orientations, the
        # true min distance is larger, so this is a valid (conservative) lower bound.
        
        min_sep = center_dist - math.sqrt(3)  # (N, N), diagonal is -sqrt(3)
        # Zero out diagonal
        np.fill_diagonal(min_sep, 1e6)
        
        # Softplus penalty: sum of softplus(-min_sep[i,j] / beta) * beta for i<j
        # Use symmetric: take upper triangle
        iu = np.triu_indices(N, k=1)
        min_sep_pairs = min_sep[iu]  # (N*(N-1)/2,)
        
        # softplus(x) = log(1 + exp(x)), stable version
        x = -min_sep_pairs / beta
        softplus_x = np.where(x > 20, x, np.log1p(np.exp(x)))
        overlap_pen = beta * np.sum(softplus_x)
        
        return t + 100.0 * epigraph_pen + 50.0 * overlap_pen

    def _smooth_gradient(params, beta=8.0):
        """Finite-difference gradient (fallback if analytic is too complex)."""
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = _smooth_objective(params, beta)
        for i in range(len(params)):
            params[i] += eps
            fp = _smooth_objective(params, beta)
            params[i] -= eps
            params[i] -= eps
            fm = _smooth_objective(params, beta)
            params[i] += eps
            grad[i] = (fp - fm) / (2 * eps)
        return grad

    def _exact_side_length(inner_list):
        """Compute exact minimum outer side length for a list of (x,y,angle_deg)."""
        cx, cy = 0.0, 0.0
        max_val = 0.0
        for (ix, iy, iang) in inner_list:
            a = math.radians(iang)
            for k in range(6):
                theta = a + k * math.pi / 3.0
                vx = ix + math.cos(theta)
                vy = iy + math.sin(theta)
                dx = vx - cx
                dy = vy - cy
                for nx, ny in _normals:
                    proj = dx * nx + dy * ny
                    if proj > max_val:
                        max_val = proj
        return max_val / _apothem

    def _check_validity(inner_list, outer_side):
        """Check all inner hexes are inside outer hex and no overlaps."""
        # Containment
        for (ix, iy, iang) in inner_list:
            a = math.radians(iang)
            for k in range(6):
                theta = a + k * math.pi / 3.0
                vx = ix + math.cos(theta)
                vy = iy + math.sin(theta)
                for nx, ny in _normals:
                    if vx * nx + vy * ny > outer_side * _apothem + 1e-9:
                        return False
        # Non-overlap using exact polygon intersection
        for i in range(N):
            for j in range(i + 1, N):
                if _overlap_area_exact(inner_list[i], inner_list[j]) > 1e-8:
                    return False
        return True

    def _overlap_area_exact(h1, h2):
        """Exact overlap area of two hexagons (Sutherland-Hodgman)."""
        def hex_verts(h):
            x, y, adeg = h
            a = math.radians(adeg)
            return [(x + math.cos(a + k * math.pi / 3), y + math.sin(a + k * math.pi / 3)) for k in range(6)]
        
        def point_in_convex(p, poly):
            for i in range(6):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % 6]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                    return False
            return True
        
        def line_intersect(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-15:
                return p1
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        
        def clip(subject, clipper):
            output = list(subject)
            for i in range(6):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clipper):
                        if not point_in_convex(S, clipper):
                            output.append(line_intersect(S, E, clipper[i], clipper[(i+1)%6]))
                        output.append(E)
                    elif point_in_convex(S, clipper):
                        output.append(line_intersect(S, E, clipper[i], clipper[(i+1)%6]))
                    S = E
            return output
        
        def area_signed(verts):
            n = len(verts)
            a = 0.0
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i+1) % n]
                a += x1 * y2 - x2 * y1
            return a / 2.0
        
        v1 = hex_verts(h1)
        v2 = hex_verts(h2)
        clipped = clip(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(area_signed(clipped))

    # --- Topology-based initializations ---
    def _init_flower():
        """Classic flower: 1 center, 6 ring1, 4 ring2."""
        eps = 1e-9
        centers = [(0.0, 0.0, 30.0)]
        for k in range(6):
            ang = k * 60.0
            r = math.sqrt(3) + eps
            centers.append((r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)), 30.0))
        for k in range(4):
            ang = 30.0 + k * 60.0
            r = 3.0 + 2*eps
            centers.append((r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)), 30.0))
        return centers

    def _init_compact_grid():
        """Tighter arrangement: offset rows."""
        # Try a more compact layout
        centers = []
        # Row 0: 3 hexagons
        for k in range(3):
            centers.append(((k - 1) * 2 * math.sqrt(3), 0.0, 30.0))
        # Row 1: 4 hexagons offset
        for k in range(4):
            centers.append(((k - 1.5) * 2 * math.sqrt(3), 1.5, 30.0))
        # Row 2: 3 hexagons
        for k in range(3):
            centers.append(((k - 1) * 2 * math.sqrt(3), 3.0, 30.0))
        return centers

    def _init_hex_lattice():
        """Standard hexagonal lattice tiling subset."""
        centers = []
        # Hex lattice: neighbors at distance sqrt(3) in 6 directions
        # Place 11 in a compact cluster
        positions = [
            (0, 0),
            (math.sqrt(3), 0), (-math.sqrt(3), 0),
            (0, 1.5), (0, -1.5),
            (math.sqrt(3)/2, 1.5), (-math.sqrt(3)/2, 1.5),
            (math.sqrt(3)/2, -1.5), (-math.sqrt(3)/2, -1.5),
            (math.sqrt(3), 1.5),
            (math.sqrt(3), -1.5),
        ]
        for x, y in positions:
            centers.append((x, y, 30.0))
        return centers

    def _init_rotated_mixed():
        """Mixed rotations to exploit interlocking."""
        centers = [(0.0, 0.0, 0.0)]
        for k in range(6):
            ang = k * 60.0
            r = math.sqrt(3) + 1e-9
            # Alternate rotations
            rot = 0.0 if k % 2 == 0 else 30.0
            centers.append((r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)), rot))
        for k in range(4):
            ang = 30.0 + k * 60.0
            r = 3.0 + 2e-9
            rot = 15.0 if k % 2 == 0 else 45.0
            centers.append((r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang)), rot))
        return centers

    def _init_jittered(seed_idx):
        """Jittered version of flower for diversity."""
        np.random.seed(seed_idx)
        centers = _init_flower()
        jittered = []
        for (x, y, a) in centers:
            jittered.append((x + np.random.normal(0, 0.05), y + np.random.normal(0, 0.05), a + np.random.normal(0, 5.0)))
        return jittered

    # --- Main optimization loop ---
    from scipy.optimize import minimize
    
    init_funcs = [_init_flower, _init_compact_grid, _init_hex_lattice, _init_rotated_mixed,
                  lambda: _init_jittered(1), lambda: _init_jittered(42), lambda: _init_jittered(123)]
    
    best_score = float('inf')
    best_result = None
    
    # Get baseline from flower
    flower = _init_flower()
    side_flower = _exact_side_length(flower)
    if _check_validity(flower, side_flower):
        best_score = side_flower
        best_result = (flower, (0.0, 0.0), side_flower, 0.0)
    
    for init_idx, init_func in enumerate(init_funcs):
        if time.time() > deadline - 25:
            break
        
        try:
            init_centers = init_func()
            # Convert to param vector: [x0, y0, th0_rad, ..., xN-1, yN-1, thN-1_rad, t]
            params0 = np.zeros(3*N + 1)
            for i, (x, y, adeg) in enumerate(init_centers):
                params0[3*i] = x
                params0[3*i+1] = y
                params0[3*i+2] = math.radians(adeg)
            # Initial t estimate
            t0 = _exact_side_length(init_centers) + 0.5
            params0[3*N] = t0
            
            # Run L-BFGS-B with smooth objective
            result = minimize(
                _smooth_objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000}
            )
            
            if time.time() > deadline - 15:
                break
            
            # Extract solution
            best_params = result.x
            inner_list = []
            for i in range(N):
                x = best_params[3*i]
                y = best_params[3*i+1]
                a = best_params[3*i+2] * 180.0 / math.pi
                inner_list.append((x, y, a))
            
            t_opt = best_params[3*N]
            side_exact = _exact_side_length(inner_list)
            
            # Use the tighter of t_opt and side_exact
            side_candidate = max(t_opt, side_exact)
            
            if _check_validity(inner_list, side_candidate + 1e-9):
                if side_candidate < best_score:
                    best_score = side_candidate
                    best_result = (inner_list, (0.0, 0.0), side_candidate, 0.0)
        except Exception:
            pass
    
    # Final polish: try a few more L-BFGS runs from best result with tighter tolerance
    if best_result is not None and time.time() < deadline - 10:
        try:
            inner_best, _, _, _ = best_result
            params0 = np.zeros(3*N + 1)
            for i, (x, y, adeg) in enumerate(inner_best):
                params0[3*i] = x
                params0[3*i+1] = y
                params0[3*i+2] = math.radians(adeg)
            params0[3*N] = best_score + 0.1
            
            result = minimize(
                _smooth_objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 15000}
            )
            
            best_params = result.x
            inner_list = []
            for i in range(N):
                x = best_params[3*i]
                y = best_params[3*i+1]
                a = best_params[3*i+2] * 180.0 / math.pi
                inner_list.append((x, y, a))
            
            t_opt = best_params[3*N]
            side_exact = _exact_side_length(inner_list)
            side_candidate = max(t_opt, side_exact)
            
            if _check_validity(inner_list, side_candidate + 1e-9):
                if side_candidate < best_score:
                    best_score = side_candidate
                    best_result = (inner_list, (0.0, 0.0), side_candidate, 0.0)
        except Exception:
            pass
    
    # Fallback: if no valid result found, return the flower
    if best_result is None:
        flower = _init_flower()
        side_flower = _exact_side_length(flower)
        best_result = (flower, (0.0, 0.0), side_flower, 0.0)
    
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END
