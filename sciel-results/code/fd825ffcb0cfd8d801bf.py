# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(verts):
    """Compute outward unit normals for each edge of a hexagon given its 6 vertices."""
    norms = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        e = p2 - p1
        n = np.array([e[1], -e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            n = n / norm
        else:
            n = np.array([1.0, 0.0])
        norms.append(n)
    return np.array(norms)


def _max_overlap(v1, v2):
    """
    Compute the maximum overlap between two hexagons.
    Returns 0.0 if separated (gap > 0 on at least one normal),
    positive value if overlapping (no separating normal found).
    """
    n1 = _hex_edge_normals(v1)
    n2 = _hex_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_gap = float('inf')
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        gap = max(max1 - min2, max2 - min1)
        if gap < min_gap:
            min_gap = gap
    
    return max(0.0, 1e-6 - min_gap)


def _min_enclosing_hex_side(inner_hex_data, outer_angle_deg=0.0):
    """
    Compute minimum side of regular hexagon (centered at origin, given angle) containing all inner hexes.
    Returns side length (float).
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = outer_angle + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        projs = verts @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    return max_proj * 2.0 / np.sqrt(3.0)


def _check_validity(inner_hex_data, outer_hex_data, side):
    """Check all validity conditions. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Check every inner vertex is inside outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        for p in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < -1e-6:
                    return False
    
    # Check every pair of inner hexagons is separated
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            n1 = _hex_edge_normals(v1)
            n2 = _hex_edge_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            separated = False
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Structural change: Use a contact-graph / topology-based approach.
    Define the 11 hexagons by their positions in a honeycomb lattice with variable
    spacing and rotation, then optimize the lattice parameters to minimize the
    enclosing hexagon side while maintaining non-overlap.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Define the honeycomb topology:
    # Center hex at origin
    # 6 hexes in first ring (touching center)
    # 4 hexes in second ring (touching two first-ring hexes each)
    
    # For unit hexagons, the center-to-center distance for touching is sqrt(3)
    # (twice the apothem of a unit hexagon)
    
    # Parameterize the packing by:
    # - lattice_scale: scaling factor for inter-hex distances (1.0 = touching)
    # - cluster_rot: global rotation of the cluster
    # - outer_angle: rotation of the outer hexagon
    
    # Honeycomb positions (for scale=1, rot=0):
    # Center: (0, 0)
    # First ring (6 hexes): at distance sqrt(3) from center, angles 0, 60, 120, 180, 240, 300
    # Second ring (4 hexes): in the gaps between first-ring hexes
    
    # Actually, let me define the 11 positions more carefully.
    # A hexagonal arrangement of 11 hexagons:
    # - 1 center
    # - 6 in first shell (touching center)
    # - 4 in second shell (touching 2 first-shell hexes each)
    
    # For a regular hexagonal lattice with spacing d (center-to-center for touching):
    # d = sqrt(3) for unit hexagons
    
    # First shell positions (relative to center):
    # (d, 0), (d/2, d*sqrt(3)/2), (-d/2, d*sqrt(3)/2), (-d, 0), (-d/2, -d*sqrt(3)/2), (d/2, -d*sqrt(3)/2)
    
    # Second shell: 4 hexes in the "corner" gaps
    # These are at positions like: (3d/2, d*sqrt(3)/2), etc.
    # Actually, for 11 hexes in a hexagonal cluster, the second shell has 6 hexes,
    # but we only need 4 to total 11. Let me use a different topology.
    
    # Alternative: Use the standard 11-hex packing which is:
    # Row 1: 3 hexes
    # Row 2: 5 hexes  
    # Row 3: 3 hexes
    # Total: 11
    
    # For a honeycomb with horizontal spacing dx and vertical spacing dy:
    # dx = sqrt(3) * scale (for touching)
    # dy = 1.5 * scale (for touching, since hex height = sqrt(3), and rows offset by half)
    
    # Actually, for pointy-top hexagons:
    # Width = sqrt(3) * side, Height = 2 * side
    # Horizontal spacing (same row) = sqrt(3) * side
    # Vertical spacing (adjacent rows) = 1.5 * side
    # Offset between rows = sqrt(3)/2 * side
    
    # For unit hexes (side=1):
    # dx = sqrt(3), dy = 1.5, offset = sqrt(3)/2
    
    # 11 hexes in a hexagonal cluster:
    # Row 0 (top): 3 hexes at x = -dx, 0, dx
    # Row 1 (mid): 5 hexes at x = -dx*1.5, -dx/2, dx/2, dx*1.5, ... wait that's not right
    
    # Let me just use the known good seed and optimize it properly.
    
    # The seed from the current program:
    inner_seed = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ]
    )
    
    # NEW APPROACH: Parameterize by relative positions and optimize directly.
    # Use a different strategy: start from multiple diverse initial configurations
    # and use a more robust optimization.
    
    # Configuration 1: The seed honeycomb
    # Configuration 2: A more compact arrangement
    # Configuration 3: Random initial positions
    
    configs = []
    
    # Config 1: Seed
    configs.append(inner_seed.copy())
    
    # Config 2: Tighter honeycomb (reduce spacing to test if we can compress)
    # The seed has hexes that may not be touching; let's compute proper touching positions
    # For unit hexes, touching distance = 2 * apothem = sqrt(3) ≈ 1.732
    # But the seed uses 2.5 for horizontal, which suggests they're NOT touching.
    
    # Let me create a properly touching honeycomb:
    d = np.sqrt(3)  # touching distance
    # Center
    # First ring: 6 hexes at distance d
    # Second ring: 4 hexes
    
    # Actually, the standard compact 11-hex packing:
    # It's a hexagonal number arrangement: 1 + 6 + 4 = 11
    # But the 4 in the second ring are placed in 4 of the 6 gaps.
    
    # Let me define it properly:
    # Center: (0, 0)
    # First ring (6): at angles 0, 60, 120, 180, 240, 300, distance d
    # Second ring (4): at the "outer" positions that touch 2 first-ring hexes
    
    # First ring positions:
    first_ring = []
    for k in range(6):
        ang = np.radians(60 * k)
        first_ring.append([d * np.cos(ang), d * np.sin(ang)])
    
    # Second ring: positions that are at distance d from two first-ring hexes
    # These are at distance 2d from center, at angles 30, 90, 150, 210, 270, 330
    # But we only need 4. Let's pick 4 that give the most symmetric arrangement.
    # Actually, for 11 hexes, we want a hexagonal cluster.
    # The 11 hexes form a larger hexagon with side length 2 (in hex units).
    # That has 1 + 6 + 12 = 19 hexes for side 2, which is too many.
    # For side 1.5, it's not integer.
    
    # Let me just use the seed and optimize it with a better method.
    
    # NEW STRATEGY: Use differential evolution for global optimization
    # of the 33 parameters (11 hexes × 3 params each)
    # with a penalty function for overlaps and containment.
    
    def objective(params):
        """
        Objective function for optimization.
        params: 33 values = 11 * (x, y, angle_deg)
        Returns: enclosing_side + penalty
        """
        inner = params.reshape(11, 3)
        
        # Compute enclosing side (assuming outer hex centered at origin, angle 0)
        side = _min_enclosing_hex_side(inner, 0.0)
        
        # Compute overlap penalty
        total_overlap = 0.0
        n = len(inner)
        verts_list = []
        for i in range(n):
            x, y, a_deg = inner[i]
            verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
        
        for i in range(n):
            for j in range(i + 1, n):
                ov = _max_overlap(verts_list[i], verts_list[j])
                total_overlap += ov
        
        # Also penalize if hexes are too far apart (to keep them compact)
        # This helps the optimizer find compact solutions
        max_dist = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
                max_dist = max(max_dist, dist)
        
        compactness_penalty = max(0, max_dist - 8.0) * 0.1
        
        return side + 100.0 * total_overlap + compactness_penalty
    
    # Use differential evolution for global search
    rng = np.random.default_rng(42)
    
    # Define bounds for each parameter
    # x, y in [-5, 5], angle in [0, 360)
    bounds = []
    for i in range(11):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 360.0))  # angle
    
    # Start with the seed as initial population
    best_inner = inner_seed.copy()
    best_side = _min_enclosing_hex_side(inner_seed, 0.0)
    
    # Run differential evolution
    if time.time() < deadline - 120:
        try:
            from scipy.optimize import differential_evolution
            
            # Use a smaller population for speed
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                mut=(0.3, 0.8),
                recomb=0.5,
                polish=False,  # Skip polish to save time
                init='sobol'
            )
            
            if result.fun < best_side:
                candidate = result.x.reshape(11, 3)
                candidate[:, 2] = candidate[:, 2] % 360.0
                cand_side = _min_enclosing_hex_side(candidate, 0.0)
                if _check_validity(candidate, np.array([0.0, 0.0, 0.0]), cand_side):
                    best_inner = candidate
                    best_side = cand_side
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 60:
        x0 = best_inner.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_side:
                candidate = res.x.reshape(11, 3)
                candidate[:, 2] = candidate[:, 2] % 360.0
                cand_side = _min_enclosing_hex_side(candidate, 0.0)
                if _check_validity(candidate, np.array([0.0, 0.0, 0.0]), cand_side):
                    best_inner = candidate
                    best_side = cand_side
        except Exception:
            pass
    
    # Phase 3: Optimize the outer hexagon angle
    if time.time() < deadline - 30:
        def obj_outer_angle(outer_ang):
            return _min_enclosing_hex_side(best_inner, outer_ang)
        
        # Grid search over outer angles
        best_outer_ang = 0.0
        best_side_with_outer = best_side
        
        for outer_ang in np.arange(0, 60, 1.0):
            if time.time() > deadline - 30:
                break
            side = obj_outer_angle(outer_ang)
            if side < best_side_with_outer:
                best_side_with_outer = side
                best_outer_ang = outer_ang
        
        # Refine
        if time.time() < deadline - 30:
            try:
                res = minimize(obj_outer_angle, best_outer_ang, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_side_with_outer:
                    best_outer_ang = res.x[0]
                    best_side_with_outer = res.fun
            except Exception:
                pass
        
        best_side = best_side_with_outer
        best_outer = np.array([0.0, 0.0, best_outer_ang])
    
    # Final validity check
    if not _check_validity(best_inner, best_outer, best_side):
        # Fallback to seed
        best_inner = inner_seed.copy()
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = _min_enclosing_hex_side(inner_seed, 0.0)
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
