# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, fsolve, least_squares

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

def _generate_symmetric_graphs(n=11):
    """Generate contact graphs consistent with the 1+6+4 topology.
    Returns list of edge lists (each edge is (i,j) with i<j)."""
    graphs = []
    # Center is 0, ring is 1-6, outer is 7-10
    # Ring angles: 30+60k for k=0..5 -> indices 1..6
    # Outer angles: 30+120k for k=0..3 -> indices 7..10
    # Ring hex k touches center and ring hex (k-1)%6 and (k+1)%6
    
    # Base ring edges
    ring_edges = [(0, k+1) for k in range(6)]  # center to ring
    for k in range(6):
        ring_edges.append((k+1, (k+1)%6 + 1))  # ring ring
    
    # Outer hex k (index 7+k) sits in valley between ring hex (2k) and (2k+1)
    # So outer k touches ring (2k+1) and ring (2k+2)%6
    # Actually let's think: outer at angle 30+120k is between ring at 30+60*(2k) and 30+60*(2k+1)
    # Ring index for angle 30+60*m is m+1
    # So outer k touches ring (2k)%6+1 and (2k+1)%6+1
    
    outer_ring_edges = []
    for k in range(4):
        r1 = (2*k) % 6 + 1
        r2 = (2*k+1) % 6 + 1
        outer_ring_edges.append((7+k, r1))
        outer_ring_edges.append((7+k, r2))
    
    base_edges = ring_edges + outer_ring_edges
    
    # Now add optional outer-outer edges
    # Outer hex k and k+1 (mod 4) may or may not touch
    # Also outer k may touch outer k+2 (opposite) - unlikely
    
    # Generate all subsets of possible outer-outer edges
    possible_outer_outer = [(7+k, 7+(k+1)%4) for k in range(4)]  # adjacent outers
    # Also consider outer touching non-adjacent ring
    # For now, just vary outer-outer connections
    
    for mask in range(1 << 4):
        edges = list(base_edges)
        for k in range(4):
            if mask & (1 << k):
                edges.append(possible_outer_outer[k])
        # Deduplicate and sort
        edge_set = set()
        for e in edges:
            edge_set.add((min(e), max(e)))
        graphs.append(sorted(edge_set))
    
    # Also try graphs where outer hexes touch non-adjacent ring hexes
    # This is getting complex; stick with the main family
    return graphs

def _solve_contact_graph(edges, init_positions, init_angles):
    """Solve for positions where all edges in the contact graph are exactly at distance sqrt(3)."""
    n = 11
    d_target = np.sqrt(3)
    
    def residuals(vars):
        positions = vars[:2*n].reshape(n, 2)
        # Fix first two positions to break translational/rotational symmetry
        # Actually, we'll fix position 0 at origin and position 1 on x-axis
        # But that's 3 constraints. Let's just solve and then center.
        
        res = []
        for (i, j) in edges:
            d = np.linalg.norm(positions[i] - positions[j])
            res.append(d - d_target)
        return np.array(res)
    
    # Initial guess
    init_vars = np.concatenate([init_positions.flatten(), init_angles])
    
    try:
        result = least_squares(
            residuals,
            init_vars[:2*n],
            method='lm',
            max_nfev=10000,
            xtol=1e-12,
            ftol=1e-12
        )
        if result.success or result.cost < 1e-8:
            positions = result.x.reshape(n, 2)
            # Center the configuration
            centroid = np.mean(positions, axis=0)
            positions -= centroid
            angles = init_angles.copy()
            
            if _check_separation_fast(positions, angles):
                return positions, angles
    except Exception:
        pass
    return None, None

def _delaunay_edges(positions):
    """Compute Delaunay triangulation edges using scipy."""
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(positions)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return sorted(edges)
    except Exception:
        return None

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    d = np.sqrt(3)
    
    # Phase 1: Generate diverse initial configurations
    def make_init_config(ring_r, outer_r, angle_offset, outer_pattern=0):
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k + angle_offset)
            positions[k + 1] = [ring_r * np.cos(a), ring_r * np.sin(a)]
        
        if outer_pattern == 0:
            outer_angles = [30, 90, 150, 210]
        elif outer_pattern == 1:
            outer_angles = [0, 60, 120, 180]
        elif outer_pattern == 2:
            outer_angles = [30, 150, 270, 90]
        else:
            outer_angles = [60, 120, 180, 240]
        
        for k, va in enumerate(outer_angles):
            a = np.radians(va + angle_offset)
            positions[7 + k] = [outer_r * np.cos(a), outer_r * np.sin(a)]
        return positions, angles
    
    # Phase 2: For each initial config, try contact graph solving
    ring_r_values = [1.732, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
    outer_r_values = [2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2]
    angle_offsets = [0, 5, 10, 15, 20, 25, 30]
    outer_patterns = [0, 1, 2, 3]
    
    # Pre-generate symmetric graphs
    symmetric_graphs = _generate_symmetric_graphs()
    
    tried_configs = set()
    
    for outer_pattern in outer_patterns:
        if time.time() > deadline - 120:
            break
        for ao in angle_offsets:
            if time.time() > deadline - 120:
                break
            for ar in ring_r_values:
                if time.time() > deadline - 120:
                    break
                for vr in outer_r_values:
                    if time.time() > deadline - 120:
                        break
                    
                    config_key = (outer_pattern, ao, round(ar, 2), round(vr, 2))
                    if config_key in tried_configs:
                        continue
                    tried_configs.add(config_key)
                    
                    init_pos, init_ang = make_init_config(ar, vr, ao, outer_pattern)
                    
                    # Check if this config is already valid
                    if _check_separation_fast(init_pos, init_ang):
                        side = _min_enclosing_side(init_pos, init_ang)
                        if side < best_side:
                            best_side = side
                            best_positions = init_pos.copy()
                            best_angles = init_ang.copy()
                    
                    # Try contact graph solving with symmetric graphs
                    for edges in symmetric_graphs:
                        if time.time() > deadline - 120:
                            break
                        sol_pos, sol_ang = _solve_contact_graph(edges, init_pos, init_ang)
                        if sol_pos is not None:
                            side = _min_enclosing_side(sol_pos, sol_ang)
                            if side < best_side:
                                best_side = side
                                best_positions = sol_pos.copy()
                                best_angles = sol_ang.copy()
                    
                    # Also try Delaunay-based graph
                    delaunay_edges = _delaunay_edges(init_pos)
                    if delaunay_edges is not None:
                        sol_pos, sol_ang = _solve_contact_graph(delaunay_edges, init_pos, init_ang)
                        if sol_pos is not None:
                            side = _min_enclosing_side(sol_pos, sol_ang)
                            if side < best_side:
                                best_side = side
                                best_positions = sol_pos.copy()
                                best_angles = sol_ang.copy()
    
    # Phase 3: Local optimization to polish
    if best_positions is not None and time.time() < deadline - 60:
        def objective(pos_flat):
            pos = pos_flat.reshape(11, 2)
            if not _check_separation_fast(pos, best_angles):
                return 1e10
            return _min_enclosing_side(pos, best_angles)
        
        try:
            res = minimize(
                objective,
                best_positions.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res.fun < best_side and res.fun < 1e9:
                trial_pos = res.x.reshape(11, 2)
                if _check_separation_fast(trial_pos, best_angles):
                    side = _min_enclosing_side(trial_pos, best_angles)
                    if side < best_side:
                        best_side = side
                        best_positions = trial_pos
        except Exception:
            pass
    
    # Phase 4: Fine coordinate descent
    if best_positions is not None and time.time() < deadline - 15:
        for cd_iter in range(500):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(11):
                for coord in range(2):
                    for step in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                        trial = best_positions.copy()
                        trial[i, coord] += step
                        if not _check_separation_fast(trial, best_angles):
                            continue
                        trial_side = _min_enclosing_side(trial, best_angles)
                        if trial_side < best_side - 1e-12:
                            best_positions = trial
                            best_side = trial_side
                            improved = True
                            break
            if not improved:
                break
    
    # Fallback
    if best_positions is None:
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(4):
            a = np.radians(30 + 120 * k)
            positions[7 + k] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        best_side = _min_enclosing_side(best_positions, best_angles)
    
    # Final verification
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        positions[0] = [0.0, 0.0]
        for k in range(6):
            a = np.radians(30 + 60 * k)
            positions[k + 1] = [d * np.cos(a), d * np.sin(a)]
        for k in range(4):
            a = np.radians(30 + 120 * k)
            positions[7 + k] = [2 * d * np.cos(a), 2 * d * np.sin(a)]
        best_positions = positions
        best_angles = angles
        final_side = _min_enclosing_side(best_positions, best_angles)
    
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001
    
    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
