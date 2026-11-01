"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def sat_distance(poly1, poly2):
        """
        Compute the minimum distance between two convex polygons using SAT.
        Returns positive if disjoint (distance), negative if overlapping (penetration depth).
        """
        max_gap = -np.inf
        
        # We need normals for all edges of both polygons
        # For a convex polygon, the separating axis is perpendicular to an edge.
        
        def get_normals(poly):
            normals = []
            n = len(poly)
            for i in range(n):
                v1 = poly[i]
                v2 = poly[(i + 1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normals.append(normal / norm)
            return normals
        
        normals1 = get_normals(poly1)
        normals2 = get_normals(poly2)
        
        for normal in normals1:
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap is the distance between the intervals if they don't overlap
            # If they overlap, gap is negative (penetration)
            gap = min(min2 - max1, min1 - max2)
            if gap > max_gap:
                max_gap = gap
                
        for normal in normals2:
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = min(min2 - max1, min1 - max2)
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_required = 0.0
        theta = outer_angle_rad
        # Pre-compute normals for the outer hexagon
        outer_normals = []
        for k in range(6):
            edge_angle = theta + np.radians(k * 60)
            # The normal to the edge is perpendicular to the edge direction?
            # Wait, the distance to the edge of a regular polygon centered at origin:
            # The support function of a regular hexagon of side S in direction n is S * sqrt(3) / 2 * |cos(angle_diff)|?
            # Actually, for a regular hexagon centered at (cx, cy) with side S,
            # the distance from center to edge is apothem a = S * sqrt(3) / 2.
            # The condition for a point (x,y) to be inside is:
            # dot((x-cx, y-cy), n_k) <= a for all 6 normals n_k.
            # So, required S such that dot(p, n_k) <= S * sqrt(3)/2.
            # S >= 2 * dot(p, n_k) / sqrt(3) for the max projection.
            
            nx = np.cos(edge_angle)
            ny = np.sin(edge_angle)
            outer_normals.append((nx, ny))
            
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for nx, ny in outer_normals:
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def make_hexes(params):
        """Build inner hexagon vertices from params."""
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        return inner_hexes, outer_cx, outer_cy, outer_a

    def objective(params):
        if time.time() > deadline:
            return 1e6
        
        inner_hexes, outer_cx, outer_cy, outer_a = make_hexes(params)
        
        penalty = 0.0
        # Check all pairs for overlap
        for i in range(N):
            for j in range(i + 1, N):
                dist = sat_distance(inner_hexes[i], inner_hexes[j])
                if dist < 0:
                    # Smooth penalty for overlap. 
                    # Using a high coefficient ensures the optimizer pushes them apart
                    # before significantly reducing the outer side.
                    penalty += 1000.0 * dist * dist
        
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        return side + penalty

    # Generate diverse initial configurations
    np.random.seed(42)
    configs = []
    sqrt3 = np.sqrt(3)

    # Config 1: Honeycomb lattice (3x4 staggered) - well-separated
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.05
            y = r * 1.55
            if r % 2 == 1:
                x += sqrt3 * 1.05 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Compact "Flower" arrangement (1 center, 6 ring, 5 outer)
    # This is often a good local minimum for hexagonal packings
    fx, fy, fa = [], [], []
    # Center
    fx.append(0); fy.append(0); fa.append(0)
    # Ring 1 (6 hexes)
    for i in range(6):
        angle = np.pi / 3 * i
        r = sqrt3 * 1.01 # Slight separation
        fx.append(r * np.cos(angle))
        fy.append(r * np.sin(angle))
        fa.append(0)
    # Ring 2 (5 hexes to make 12 total)
    # Place them in the gaps of the outer ring
    for i in range(5):
        angle = np.pi / 3 * i + np.pi / 6
        r = 2.5 * sqrt3 * 0.8 # Approximate
        fx.append(r * np.cos(angle))
        fy.append(r * np.sin(angle))
        fa.append(0)
    fx = np.array(fx); fy = np.array(fy); fa = np.array(fa)
    fx -= np.mean(fx); fy -= np.mean(fy)
    configs.append(np.concatenate([fx, fy, fa, [0.0, 0.0, 0.0]]))

    # Config 3: Compact circular arrangement
    cx, cy, ca = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 2.8
        cx.append(r * np.cos(angle))
        cy.append(r * np.sin(angle))
        ca.append(angle)
    cx = np.array(cx); cy = np.array(cy); ca = np.array(ca)
    cx -= np.mean(cx); cy -= np.mean(cy)
    configs.append(np.concatenate([cx, cy, ca, [0.0, 0.0, 0.0]]))

    # Config 4: Two rows of 6 with tighter spacing
    rx, ry, ra = [], [], []
    for i in range(6):
        rx.append(i * sqrt3 * 1.01 - 2.5 * sqrt3 * 1.01)
        ry.append(0.0)
        ra.append(0.0)
        rx.append(i * sqrt3 * 1.01 - 2.5 * sqrt3 * 1.01)
        ry.append(1.51)
        ra.append(0.0)
    rx = np.array(rx); ry = np.array(ry); ra = np.array(ra)
    rx -= np.mean(rx); ry -= np.mean(ry)
    configs.append(np.concatenate([rx, ry, ra, [0.0, 0.0, 0.0]]))

    # Evaluate all configs and pick best
    best_params = configs[0].copy()
    best_obj = objective(best_params)
    for cfg in configs:
        obj = objective(cfg)
        if obj < best_obj:
            best_obj = obj
            best_params = cfg.copy()

    # Phase 1: Differential Evolution for global search
    # Bounds: x,y in [-6, 6], angles in [-pi, pi]
    bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12 + [(-1, 1), (-1, 1), (-np.pi / 6, np.pi / 6)]
    
    de_result = None
    try:
        de_result = differential_evolution(
            objective, bounds,
            seed=42, maxiter=200, popsize=30,
            tol=1e-6, mut=(0.5, 1.0), recomb=0.8,
            polish=False, init='sohal'
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass

    # Phase 2: Local refinement with Nelder-Mead from best DE result
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Phase 3: Additional local refinements from perturbed best
    if time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_params + np.random.normal(0, 0.05, len(best_params))
            try:
                result = minimize(
                    objective, perturbed, method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass

    # Extract solution
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    outer_cx = params[36]
    outer_cy = params[37]
    outer_a = params[38]
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    
    # Verify validity
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dist = sat_distance(inner_hexes[i], inner_hexes[j])
            if dist < -1e-7: # Allow tiny numerical error
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid configuration with wider spacing
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    # Recompute exact side length without penalty
    exact_side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
    
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
