# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


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


def _min_gap_between(v1, v2):
    """
    Compute the minimum gap between two hexagons using their 12 edge normals.
    Returns the minimum gap (positive if separated, negative if overlapping).
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
    return min_gap


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
    
    Structural change: Exploit 6-fold symmetry of the outer hexagon.
    Parameterize the configuration with only the coordinates and orientation of
    the central hexagon and one of the six surrounding hexagons, then generate
    the remaining ten via rotational symmetries. This reduces the search space
    from 33 degrees of freedom to just 5 parameters.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s limit
    
    # Define the symmetric packing:
    # - 1 central hexagon at (0, 0) with angle a0
    # - 6 hexagons in first ring at distance d from center, angles 60*k + a1
    #   (where k=0..5, and a1 is the rotation of the ring)
    # - 4 hexagons in second ring at the "corner" positions
    #   These are at positions that touch two first-ring hexes each
    
    # For a 6-fold symmetric arrangement with 11 hexes:
    # Center: (0, 0)
    # First ring (6): at distance d, angles 0, 60, 120, 180, 240, 300 (plus global rotation)
    # Second ring (4): We need 4 more to total 11.
    #   In a hexagonal lattice, the next positions after the first ring are at
    #   distance 2d from center, at angles 30, 90, 150, 210, 270, 330.
    #   But we only need 4. For maximum symmetry, we can't pick just 4 of 6
    #   while maintaining 6-fold symmetry. So the 6-fold symmetry is broken
    #   by the second ring.
    
    # Alternative: Use a different topology.
    # Actually, for 11 hexes with 6-fold symmetry, we could have:
    # - 1 center
    # - 6 in first ring
    # - 4 in second ring, but arranged with 2-fold symmetry (pairs)
    
    # Let me reconsider: The optimal packing of 11 hexes might not have full
    # 6-fold symmetry. But we can still use a reduced parameterization.
    
    # Strategy: Start from the known good seed and optimize with a better method.
    # The seed has a honeycomb-like arrangement.
    
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
    
    best_inner = inner_seed.copy()
    best_outer = np.array([0.0, 0.0, 0.0])
    best_side = _min_enclosing_hex_side(inner_seed, 0.0)
    
    # Phase 1: Try a 6-fold symmetric parameterization
    # Parameters: [d1, a_ring, d2, a2, outer_angle]
    # where d1 = distance from center to first ring
    #       a_ring = rotation of first ring
    #       d2 = distance from center to second ring
    #       a2 = rotation of second ring
    #       outer_angle = rotation of outer hexagon
    
    def symmetric_config(params):
        """Generate 11 hex positions from symmetric parameters."""
        d1, a_ring, d2, a2, outer_angle = params
        
        hexes = np.zeros((11, 3))
        
        # Center hex
        hexes[0] = [0.0, 0.0, 0.0]
        
        # First ring: 6 hexes
        for k in range(6):
            ang = np.radians(a_ring + 60 * k)
            hexes[1 + k] = [d1 * np.cos(ang), d1 * np.sin(ang), a_ring + 60 * k]
        
        # Second ring: 4 hexes at angles 30, 150, 210, 330 (relative to a2)
        # These are the "corner" positions in a hexagonal lattice
        second_angles = [30, 150, 210, 330]
        for k, base_ang in enumerate(second_angles):
            ang = np.radians(a2 + base_ang)
            hexes[7 + k] = [d2 * np.cos(ang), d2 * np.sin(ang), a2 + base_ang]
        
        return hexes, outer_angle
    
    def symmetric_objective(params):
        """Objective for symmetric configuration."""
        hexes, outer_angle = symmetric_config(params)
        
        # Check for overlaps
        verts_list = []
        for i in range(11):
            x, y, a_deg = hexes[i]
            verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
        
        total_penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                gap = _min_gap_between(verts_list[i], verts_list[j])
                if gap < 1e-6:
                    total_penalty += (1e-6 - gap)
        
        side = _min_enclosing_hex_side(hexes, outer_angle)
        
        # Compactness penalty
        max_dist = 0
        for i in range(11):
            for j in range(i + 1, 11):
                dist = np.linalg.norm(hexes[i, :2] - hexes[j, :2])
                max_dist = max(max_dist, dist)
        compactness = max(0, max_dist - 8.0) * 0.1
        
        return side + 100.0 * total_penalty + compactness
    
    # Optimize the symmetric configuration
    if time.time() < deadline - 180:
        try:
            # Initial guess based on touching hexes
            d1_init = np.sqrt(3)  # touching distance for unit hexes
            a_ring_init = 0.0
            d2_init = 2 * np.sqrt(3)  # second ring distance
            a2_init = 0.0
            outer_angle_init = 0.0
            
            x0 = [d1_init, a_ring_init, d2_init, a2_init, outer_angle_init]
            
            res = minimize(
                symmetric_objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < best_side:
                hexes, outer_angle = symmetric_config(res.x)
                if _check_validity(hexes, np.array([0.0, 0.0, outer_angle]), 
                                   _min_enclosing_hex_side(hexes, outer_angle)):
                    best_inner = hexes
                    best_outer = np.array([0.0, 0.0, outer_angle])
                    best_side = _min_enclosing_hex_side(hexes, outer_angle)
        except Exception:
            pass
    
    # Phase 2: Full 33-parameter optimization from the best symmetric solution
    if time.time() < deadline - 120:
        try:
            def full_objective(params):
                inner = params.reshape(11, 3)
                
                side = _min_enclosing_hex_side(inner, best_outer[2])
                
                verts_list = []
                for i in range(11):
                    x, y, a_deg = inner[i]
                    verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
                
                total_penalty = 0.0
                for i in range(11):
                    for j in range(i + 1, 11):
                        gap = _min_gap_between(verts_list[i], verts_list[j])
                        if gap < 1e-6:
                            total_penalty += (1e-6 - gap)
                
                max_dist = 0
                for i in range(11):
                    for j in range(i + 1, 11):
                        dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
                        max_dist = max(max_dist, dist)
                compactness = max(0, max_dist - 8.0) * 0.1
                
                return side + 100.0 * total_penalty + compactness
            
            x0 = best_inner.flatten()
            
            res = minimize(
                full_objective, x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < best_side:
                candidate = res.x.reshape(11, 3)
                candidate[:, 2] = candidate[:, 2] % 360.0
                cand_side = _min_enclosing_hex_side(candidate, best_outer[2])
                if _check_validity(candidate, best_outer, cand_side):
                    best_inner = candidate
                    best_side = cand_side
        except Exception:
            pass
    
    # Phase 3: Optimize outer hexagon angle
    if time.time() < deadline - 60:
        def obj_outer_angle(outer_ang):
            return _min_enclosing_hex_side(best_inner, outer_ang)
        
        best_outer_ang = best_outer[2]
        best_side_with_outer = best_side
        
        # Grid search
        for outer_ang in np.arange(0, 60, 2.0):
            if time.time() > deadline - 60:
                break
            side = obj_outer_angle(outer_ang)
            if side < best_side_with_outer:
                best_side_with_outer = side
                best_outer_ang = outer_ang
        
        # Refine
        if time.time() < deadline - 60:
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
    
    # Phase 4: Final local refinement with multiple restarts
    if time.time() < deadline - 30:
        try:
            def final_objective(params):
                inner = params.reshape(11, 3)
                side = _min_enclosing_hex_side(inner, best_outer[2])
                
                verts_list = []
                for i in range(11):
                    x, y, a_deg = inner[i]
                    verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
                
                total_penalty = 0.0
                for i in range(11):
                    for j in range(i + 1, 11):
                        gap = _min_gap_between(verts_list[i], verts_list[j])
                        if gap < 1e-6:
                            total_penalty += (1e-6 - gap)
                
                return side + 100.0 * total_penalty
            
            rng = np.random.default_rng(42)
            for restart in range(3):
                if time.time() > deadline - 30:
                    break
                
                # Perturb the current best
                perturbation = rng.normal(0, 0.1, size=(11, 3))
                perturbation[:, 2] = rng.normal(0, 5, size=11)  # larger angle perturbation
                x0 = (best_inner + perturbation).flatten()
                
                res = minimize(
                    final_objective, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                if res.fun < best_side:
                    candidate = res.x.reshape(11, 3)
                    candidate[:, 2] = candidate[:, 2] % 360.0
                    cand_side = _min_enclosing_hex_side(candidate, best_outer[2])
                    if _check_validity(candidate, best_outer, cand_side):
                        best_inner = candidate
                        best_side = cand_side
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_inner, best_outer, best_side):
        # Fallback to seed
        best_inner = inner_seed.copy()
        best_outer = np.array([0.0, 0.0, 0.0])
        best_side = _min_enclosing_hex_side(inner_seed, 0.0)
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
