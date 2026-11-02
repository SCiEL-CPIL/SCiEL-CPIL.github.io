"""
Strategy: Centroid-Projected Minkowski Sum with Global Orientation Search.
Arrange 12 unit hexagons in a fixed honeycomb cluster (1 center + 6 ring + 5 ring).
The outer hexagon side length for a given orientation is computed via the Minkowski sum
support function: for each of the 6 outer normals, the required half-width is the max
over all inner vertices of the projection onto that normal, plus the support of a unit
hexagon in that direction. We minimize over the 6-fold symmetric cluster parameters
(r1, r2, a1, a2, outer_angle) using a multi-start global search.
"""
import numpy as np
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Precompute unit hexagon vertices at angle 0
    unit_hex_verts = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in range(0, 360, 60)
    ])

    # Support function of unit hexagon: for a given direction (nx, ny),
    # the max projection of unit hexagon vertices onto that direction
    def unit_hex_support(nx, ny):
        dots = unit_hex_verts[:, 0] * nx + unit_hex_verts[:, 1] * ny
        return np.max(dots)

    def get_inner_positions(r1, a1, r2, a2):
        """Generate 12 inner hexagon positions: (x, y, angle_deg)."""
        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = math.radians(60 * k)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            positions.append((x, y, a1))
        # Ring 2: 5 hexagons offset by 30 degrees
        for k in range(5):
            angle = math.radians(60 * k + 30)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            positions.append((x, y, a2))
        return positions

    def compute_min_side(r1, a1, r2, a2, outer_angle_deg):
        """
        Compute the minimal outer hexagon side length for a given cluster config.
        The outer hexagon is centered at origin with rotation outer_angle_deg.
        We use the support function approach: for each of the 6 face normals of the
        outer hexagon, the required apothem is the max over all inner vertices of
        (projection onto normal) + (support of unit hex in opposite direction).
        Actually: the outer hexagon must contain all inner hexagons. The condition
        is that for each face normal direction n_i of the outer hexagon, the maximum
        projection of any inner vertex onto n_i must be <= apothem_outer.
        But we also need to account for the inner hexagon's own extent.
        
        More precisely: the outer hexagon with apothem A and orientation theta has
        faces with normals at angles theta + 30 + 60*k. For containment, we need:
        For each normal n_k, max over all inner vertices v of dot(v, n_k) <= A.
        But inner vertices are centers of inner hexagons plus their own vertices.
        
        Let me reformulate: The set of all points occupied by inner hexagons is the
        union of 12 hexagons. The outer hexagon must contain this union.
        The outer hexagon with center c, side s, rotation phi has apothem a = s*sqrt(3)/2.
        Its faces have outward normals at angles phi + 30 + 60*k (for flat-top) or
        phi + 60*k (for pointy-top). Let's use the convention that normals are at
        phi + 30 + 60*k.
        
        For containment: for each normal n_k, the maximum of dot(p, n_k) over all
        points p in the union of inner hexagons must be <= a - dot(c, n_k).
        Since c = (0,0), we need max dot(p, n_k) <= a for all k.
        
        The maximum of dot(p, n_k) over a hexagon with center (cx, cy) and rotation
        angle alpha is dot((cx,cy), n_k) + support_of_unit_hex(n_k, alpha).
        The support of a unit hexagon rotated by alpha in direction n_k is the same
        as the support in direction (n_k rotated by -alpha), which is unit_hex_support
        applied to the rotated direction.
        """
        outer_angle_rad = math.radians(outer_angle_deg)
        # Face normals of outer hexagon
        normal_angles = outer_angle_rad + math.radians(30) + math.radians(60) * np.arange(6)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        positions = get_inner_positions(r1, a1, r2, a2)
        
        max_required = 0.0
        for (cx, cy, alpha_deg) in positions:
            alpha_rad = math.radians(alpha_deg)
            # The support of the inner hexagon in direction n is:
            # dot(center, n) + support of unit hex rotated by alpha in direction n
            # = dot(center, n) + support of unit hex in direction rotate(n, -alpha)
            for nx, ny in normals:
                # Rotate the normal by -alpha to get the direction in the hexagon's local frame
                cos_a = math.cos(-alpha_rad)
                sin_a = math.sin(-alpha_rad)
                local_nx = nx * cos_a - ny * sin_a
                local_ny = nx * sin_a + ny * cos_a
                support = unit_hex_support(local_nx, local_ny)
                val = cx * nx + cy * ny + support
                max_required = max(max_required, val)
        
        apothem = max_required
        side = apothem / (math.sqrt(3) / 2)
        return side

    def check_disjoint(r1, a1, r2, a2):
        """Check if inner hexagons are pairwise disjoint using SAT."""
        positions = get_inner_positions(r1, a1, r2, a2)
        n_pos = len(positions)
        
        for i in range(n_pos):
            for j in range(i + 1, n_pos):
                xi, yi, ai = positions[i]
                xj, yj, aj = positions[j]
                
                # Get vertices
                ai_rad = math.radians(ai)
                aj_rad = math.radians(aj)
                
                vi = np.array([
                    [xi + math.cos(ai_rad + math.radians(60*k)), yi + math.sin(ai_rad + math.radians(60*k))]
                    for k in range(6)
                ])
                vj = np.array([
                    [xj + math.cos(aj_rad + math.radians(60*k)), yj + math.sin(aj_rad + math.radians(60*k))]
                    for k in range(6)
                ])
                
                # SAT check
                axes = []
                for v in [vi, vj]:
                    for k in range(6):
                        p1 = v[k]
                        p2 = v[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            axes.append(normal / norm)
                
                separated = False
                for axis in axes:
                    proj_i = vi @ axis
                    proj_j = vj @ axis
                    if proj_i.max() < proj_j.min() - 1e-9 or proj_j.max() < proj_i.min() - 1e-9:
                        separated = True
                        break
                
                if not separated:
                    return False
        return True

    def objective(params):
        if time.time() > deadline - 2:
            return 1000.0
        r1, a1, r2, a2, o_a = params
        if r1 < 0.5 or r2 < 0.5:
            return 1000.0
        if not check_disjoint(r1, a1, r2, a2):
            return 500.0
        return compute_min_side(r1, a1, r2, a2, o_a)

    # Multi-start global search
    best_side = float('inf')
    best_params = None
    
    # Initial guesses
    initial_guesses = [
        [2.0, 0.0, 3.5, 30.0, 30.0],
        [2.0, 0.0, 3.46, 0.0, 0.0],
        [2.0, 30.0, 3.5, 0.0, 0.0],
        [2.0, 0.0, 3.5, 0.0, 0.0],
        [1.9, 0.0, 3.4, 30.0, 30.0],
        [2.1, 0.0, 3.6, 30.0, 30.0],
    ]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(20):
        r1 = np.random.uniform(1.8, 2.3)
        a1 = np.random.uniform(0, 60)
        r2 = np.random.uniform(3.2, 3.8)
        a2 = np.random.uniform(0, 60)
        o_a = np.random.uniform(0, 60)
        initial_guesses.append([r1, a1, r2, a2, o_a])
    
    for guess in initial_guesses:
        if time.time() > deadline - 5:
            break
        
        # Simple local refinement using coordinate descent
        params = np.array(guess, dtype=float)
        current_val = objective(params)
        
        # Coordinate descent
        step_sizes = [0.1, 0.5, 0.1, 0.5, 0.5]
        for it in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for d in range(5):
                for sign in [1, -1]:
                    new_params = params.copy()
                    new_params[d] += sign * step_sizes[d]
                    val = objective(new_params)
                    if val < current_val:
                        params = new_params
                        current_val = val
                        improved = True
                        break
            if not improved:
                # Try smaller steps
                step_sizes = [s * 0.5 for s in step_sizes]
                if step_sizes[0] < 0.001:
                    break
        
        if current_val < best_side and current_val < 500:
            best_side = current_val
            best_params = params.copy()
    
    # If no valid solution found, use a known valid configuration
    if best_params is None:
        best_params = np.array([2.0, 0.0, 3.5, 30.0, 30.0])
        best_side = compute_min_side(*best_params)
    
    r1, a1, r2, a2, o_a = best_params
    a1 = a1 % 360
    a2 = a2 % 360
    o_a = o_a % 360
    
    # Construct final inner positions
    inner = []
    inner.append((0.0, 0.0, 0.0))
    for k in range(6):
        angle = math.radians(60 * k)
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        inner.append((x, y, a1))
    for k in range(5):
        angle = math.radians(60 * k + 30)
        x = r2 * math.cos(angle)
        y = r2 * math.sin(angle)
        inner.append((x, y, a2))
    
    oc = (0.0, 0.0)
    oa = o_a
    os = compute_min_side(r1, a1, r2, a2, o_a)
    
    return inner, oc, os, oa
