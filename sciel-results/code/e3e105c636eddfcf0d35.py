# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side length s."""
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(inner_hexes):
    """Get all vertices of all inner hexagons. inner_hexes shape (n, 3): (x, y, angle_deg)."""
    n = len(inner_hexes)
    all_v = np.zeros((n * 6, 2))
    for i in range(n):
        all_v[i*6:(i+1)*6] = _hex_vertices(inner_hexes[i, 0], inner_hexes[i, 1], 1.0, inner_hexes[i, 2])
    return all_v

def _check_separation_fast(inner_hexes, tol=1e-6):
    """Check that all pairs of inner hexagons are separated using edge normals."""
    n = len(inner_hexes)
    # Precompute vertices for each hex
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(inner_hexes[i, 0], inner_hexes[i, 1], 1.0, inner_hexes[i, 2]))
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts_list[i]
            vj = verts_list[j]
            
            # 12 edge normals: 6 from each hex
            separated = False
            for k in range(6):
                # Normal for edge k of hex i
                a_i = np.radians(inner_hexes[i, 2]) + np.radians(30 + 60 * k)
                normal_i = np.array([np.cos(a_i), np.sin(a_i)])
                ni = vi @ normal_i
                nj = vj @ normal_i
                if max(ni) < min(nj) - tol or max(nj) < min(ni) - tol:
                    separated = True
                    break
                # Normal for edge k of hex j
                a_j = np.radians(inner_hexes[j, 2]) + np.radians(30 + 60 * k)
                normal_j = np.array([np.cos(a_j), np.sin(a_j)])
                ni = vi @ normal_j
                nj = vj @ normal_j
                if max(ni) < min(nj) - tol or max(nj) < min(ni) - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    return True

def _soft_max(x, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max > 1e6:
        return x_max
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _min_enclosing_side_smooth(inner_hexes, beta=50.0):
    """Smooth version of min enclosing hexagon side length using soft-max."""
    all_v = _all_vertices(inner_hexes)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    
    # For each vertex and each normal, compute projection
    projs = all_v @ normals.T  # shape (66, 6)
    # For each normal direction, we need the max projection over all vertices
    # Use soft-max for smoothness
    max_projs = np.array([_soft_max(projs[:, i], beta) for i in range(6)])
    return np.max(max_projs) / apothem_factor

def _min_enclosing_side(inner_hexes):
    """Exact min enclosing hexagon side length."""
    all_v = _all_vertices(inner_hexes)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    
    projs = all_v @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(inner_hexes, outer_side, outer_center, tol=1e-6):
    """Check all inner vertices are inside the outer hexagon."""
    all_v = _all_vertices(inner_hexes) - outer_center
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem = outer_side * np.sqrt(3) / 2
    
    projs = all_v @ normals.T
    return np.all(projs <= apothem + tol)

def _build_symmetric_positions(r1, r2, r3, theta1, theta2, theta3):
    """
    Build positions with 6-fold symmetry.
    - 1 center hex at origin
    - 6 hexes at radius r1, angles theta1 + 60k
    - 3 hexes at radius r2, angles theta2 + 120k  (wait, need 6-fold or 3-fold)
    
    Actually for 11 hexagons with symmetry:
    - 1 center
    - 6 at radius r1 (ring 1)
    - 4 at radius r2 (ring 2) -- but 4 doesn't fit 6-fold symmetry
    
    Let me reconsider. 11 = 1 + 6 + 4. The 4 can't have 6-fold symmetry.
    Maybe: 1 + 6 + 4 where the 4 are at specific positions.
    
    Alternative: use lower symmetry. Let's parameterize by:
    - center at (0,0)
    - 6 surrounding at distance d1, angles 30+60k (for pointy-top touching)
    - 4 more at distance d2, angles 0, 90, 180, 270 (or some symmetric pattern)
    
    Actually, let me just use a general parameterization with some symmetry hints.
    """
    positions = np.zeros((11, 2))
    # Center
    positions[0] = [0.0, 0.0]
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        positions[1 + k] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    # Ring 2: 4 hexagons at angles 0, 90, 180, 270 (or 30, 120, 210, 300)
    # Let's try 30, 120, 210, 300 to align with hex grid
    for k in range(4):
        angle = np.radians(30 + 90 * k)
        positions[7 + k] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    return positions

def _solve_topology():
    """
    Solve using a contact graph approach.
    For unit hexagons, touching means center-to-center distance = sqrt(3) (for parallel hexes)
    or other distances depending on orientation.
    
    Let's use a penalty-based optimization that minimizes:
    1. The enclosing hexagon side length
    2. Penalties for overlap (negative separation)
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Initial guess: hexagonal cluster
    d = np.sqrt(3)
    positions = np.zeros((11, 2))
    positions[0] = [0.0, 0.0]
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        positions[1 + k] = [d * np.cos(angle), d * np.sin(angle)]
    # 4 more at distance 2*d in 4 directions
    for k in range(4):
        angle = np.radians(30 + 90 * k)
        positions[7 + k] = [2 * d * np.cos(angle), 2 * d * np.sin(angle)]
    
    # Angles: all 0 initially (pointy-top)
    angles = np.zeros(11)
    
    def make_inner(params):
        pos = params[:22].reshape(11, 2)
        ang = params[22:33]
        return np.column_stack([pos, ang])
    
    def objective(params):
        inner = make_inner(params)
        # Smooth enclosing side
        side = _min_enclosing_side_smooth(inner, beta=30.0)
        
        # Overlap penalty: for each pair, compute min separation over all 12 normals
        # If any pair has negative separation (overlap), add penalty
        n = 11
        verts_list = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(n)]
        
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                vi = verts_list[i]
                vj = verts_list[j]
                min_sep = np.inf
                for k in range(6):
                    a_i = np.radians(inner[i, 2]) + np.radians(30 + 60 * k)
                    normal_i = np.array([np.cos(a_i), np.sin(a_i)])
                    ni = vi @ normal_i
                    nj = vj @ normal_i
                    sep = max(min(nj) - max(ni), min(ni) - max(nj))
                    min_sep = min(min_sep, sep)
                    
                    a_j = np.radians(inner[j, 2]) + np.radians(30 + 60 * k)
                    normal_j = np.array([np.cos(a_j), np.sin(a_j)])
                    ni = vi @ normal_j
                    nj = vj @ normal_j
                    sep = max(min(nj) - max(ni), min(ni) - max(nj))
                    min_sep = min(min_sep, sep)
                
                if min_sep < 1e-6:
                    overlap_penalty += (1e-6 - min_sep) ** 2
        
        return side + 100.0 * overlap_penalty
    
    # Optimize
    best_inner = None
    best_side = np.inf
    
    # Try Nelder-Mead
    x0 = np.concatenate([positions.flatten(), angles])
    
    try:
        res = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if time.time() < deadline - 30.0:
            inner = make_inner(res.x)
            if _check_separation_fast(inner):
                side = _min_enclosing_side(inner)
                if side < best_side:
                    best_side = side
                    best_inner = inner.copy()
    except Exception:
        pass
    
    # Try SLSQP with bounds
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-5, 5)] * 22 + [(-180, 180)] * 11
            res2 = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            
            if time.time() < deadline - 30.0:
                inner = make_inner(res2.x)
                if _check_separation_fast(inner):
                    side = _min_enclosing_side(inner)
                    if side < best_side:
                        best_side = side
                        best_inner = inner.copy()
        except Exception:
            pass
    
    # Local refinement from best found
    if best_inner is not None and time.time() < deadline - 60.0:
        # Fine-tune with smaller perturbations
        np.random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 30.0:
                break
            pert = np.random.randn(11, 2) * 0.05
            trial_pos = best_inner[:, :2] + pert
            trial_inner = np.column_stack([trial_pos, best_inner[:, 2]])
            
            if _check_separation_fast(trial_inner):
                trial_side = _min_enclosing_side(trial_inner)
                if trial_side < best_side:
                    best_side = trial_side
                    best_inner = trial_inner.copy()
    
    if best_inner is None:
        # Fallback: use the initial configuration
        inner = np.column_stack([positions, angles])
        # Scale to ensure separation
        scale = 1.0
        while not _check_separation_fast(inner) and scale < 3.0:
            positions = positions * (1 + 0.05)
            inner = np.column_stack([positions, angles])
            scale += 0.05
        best_inner = inner
        best_side = _min_enclosing_side(inner)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return best_inner, outer_hex_data, float(best_side)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 620.0
    
    # Strategy 1: Symmetry-based parameterization with optimization
    # Use a reduced parameter space exploiting approximate symmetry
    
    d = np.sqrt(3)
    
    # Parameterize: center at origin, ring1 radius r1, ring2 radius r2
    # Ring 1: 6 hexes at 30+60k degrees
    # Ring 2: 4 hexes at 0, 90, 180, 270 degrees (or other pattern)
    
    def build_from_params(params):
        r1 = params[0]
        r2 = params[1]
        angle_offset = params[2]  # global rotation
        # Small perturbations for breaking symmetry
        pos = np.zeros((11, 2))
        pos[0] = [0.0, 0.0]
        for k in range(6):
            angle = np.radians(30 + 60 * k + angle_offset)
            pos[1 + k] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        for k in range(4):
            angle = np.radians(30 + 90 * k + angle_offset)
            pos[7 + k] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        angles = np.zeros(11)
        return np.column_stack([pos, angles])
    
    def objective_sym(params):
        inner = build_from_params(params)
        side = _min_enclosing_side_smooth(inner, beta=30.0)
        
        # Overlap penalty
        n = 11
        verts_list = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(n)]
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                vi = verts_list[i]
                vj = verts_list[j]
                min_sep = np.inf
                for k in range(6):
                    a_i = np.radians(inner[i, 2]) + np.radians(30 + 60 * k)
                    normal_i = np.array([np.cos(a_i), np.sin(a_i)])
                    ni = vi @ normal_i
                    nj = vj @ normal_i
                    sep = max(min(nj) - max(ni), min(ni) - max(nj))
                    min_sep = min(min_sep, sep)
                    
                    a_j = np.radians(inner[j, 2]) + np.radians(30 + 60 * k)
                    normal_j = np.array([np.cos(a_j), np.sin(a_j)])
                    ni = vi @ normal_j
                    nj = vj @ normal_j
                    sep = max(min(nj) - max(ni), min(ni) - max(nj))
                    min_sep = min(min_sep, sep)
                
                if min_sep < 1e-6:
                    overlap_penalty += (1e-6 - min_sep) ** 2
        
        return side + 100.0 * overlap_penalty
    
    # Initial guess
    x0 = np.array([d, 2 * d, 0.0])
    
    best_inner = None
    best_side = np.inf
    
    # Optimize the symmetric parameterization
    try:
        res = minimize(
            objective_sym, x0, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if time.time() < deadline - 60.0:
            inner = build_from_params(res.x)
            if _check_separation_fast(inner):
                side = _min_enclosing_side(inner)
                if side < best_side:
                    best_side = side
                    best_inner = inner.copy()
    except Exception:
        pass
    
    # Now do full 22+11 DOF refinement from the symmetric solution
    if best_inner is not None and time.time() < deadline - 90.0:
        x0_full = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2]])
        
        def objective_full(params):
            pos = params[:22].reshape(11, 2)
            ang = params[22:33]
            inner = np.column_stack([pos, ang])
            side = _min_enclosing_side_smooth(inner, beta=30.0)
            
            n = 11
            verts_list = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(n)]
            overlap_penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    vi = verts_list[i]
                    vj = verts_list[j]
                    min_sep = np.inf
                    for k in range(6):
                        a_i = np.radians(inner[i, 2]) + np.radians(30 + 60 * k)
                        normal_i = np.array([np.cos(a_i), np.sin(a_i)])
                        ni = vi @ normal_i
                        nj = vj @ normal_i
                        sep = max(min(nj) - max(ni), min(ni) - max(nj))
                        min_sep = min(min_sep, sep)
                        
                        a_j = np.radians(inner[j, 2]) + np.radians(30 + 60 * k)
                        normal_j = np.array([np.cos(a_j), np.sin(a_j)])
                        ni = vi @ normal_j
                        nj = vj @ normal_j
                        sep = max(min(nj) - max(ni), min(ni) - max(nj))
                        min_sep = min(min_sep, sep)
                    
                    if min_sep < 1e-6:
                        overlap_penalty += (1e-6 - min_sep) ** 2
            
            return side + 100.0 * overlap_penalty
        
        try:
            res2 = minimize(
                objective_full, x0_full, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            if time.time() < deadline - 30.0:
                pos = res2.x[:22].reshape(11, 2)
                ang = res2.x[22:33]
                inner = np.column_stack([pos, ang])
                if _check_separation_fast(inner):
                    side = _min_enclosing_side(inner)
                    if side < best_side:
                        best_side = side
                        best_inner = inner.copy()
        except Exception:
            pass
    
    # Fallback if nothing found
    if best_inner is None:
        positions = np.zeros((11, 2))
        positions[0] = [0.0, 0.0]
        for k in range(6):
            angle = np.radians(30 + 60 * k)
            positions[1 + k] = [d * np.cos(angle), d * np.sin(angle)]
        for k in range(4):
            angle = np.radians(30 + 90 * k)
            positions[7 + k] = [2 * d * np.cos(angle), 2 * d * np.sin(angle)]
        angles = np.zeros(11)
        best_inner = np.column_stack([positions, angles])
        best_side = _min_enclosing_side(best_inner)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return best_inner, outer_hex_data, float(best_side)

# EVOLVE-BLOCK-END
