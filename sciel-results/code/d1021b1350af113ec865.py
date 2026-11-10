import numpy as np
from scipy.optimize import linprog, minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        if L < 1e-10:
            n[i] = [1.0, 0.0]
        else:
            n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    """Return (best_margin, active_normal). best_margin>0 means separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def _min_side_for_positions(positions, outer_angle=0.0):
    """
    Calculate the minimal side length of a regular hexagon (fixed orientation)
    that encloses the given positions (centers of unit hexagons).
    The outer hexagon is centered at the centroid of the positions.
    """
    # Normals for a hexagon with angle 0
    normals = np.array([
        [np.cos(np.deg2rad(30 + 60*i)), np.sin(np.deg2rad(30 + 60*i))]
        for i in range(6)
    ])
    
    # Shift positions so centroid is at origin
    centroid = positions.mean(axis=0)
    shifted = positions - centroid
    
    proj = shifted @ normals.T
    ranges = proj.max(axis=0) - proj.min(axis=0)
    max_range = ranges.max()
    return max_range / np.sqrt(3.0)

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 590.0  # Safety margin

    best_val = -np.inf
    best_result = None

    # Initial guess: A compact cluster
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
    
    # A standard hexagonal flower of 7 hexagons
    coords_A = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    ]
    # Add 4 more to make 11. A common compact shape is a 3-3-3-2 or similar.
    # Let's try a "plus" shape or extending the flower.
    # (1,1), (-1,1) are already there. 
    # Let's add (2,0), (0,2), (-2,2), (-2,0)? No, that's 11.
    # Let's stick to the seed's initial guess which was reasonable.
    coords_A_full = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (1,1), (-1,1), (0,2), (0,-2)
    ]
    
    init_positions = np.array([axial_to_cartesian(q, r) for q, r in coords_A_full])
    init_positions -= init_positions.mean(axis=0)
    
    # Strategy: 
    # 1. Use LinProg to maximize the minimum gap between hexagons, given a fixed bounding box size? 
    #    No, we want to minimize the bounding box.
    # 2. Better: Use a penalty method with SLSQP or trust-constr to minimize side length 
    #    subject to gap constraints. But the gap constraint is non-smooth (max of linear functions).
    # 3. Alternative: Fixed orientation. The gap between two hexagons with same orientation 
    #    is a convex function of the relative position? 
    #    The gap is the minimum distance between two convex polygons. 
    #    Maximizing the minimum gap is a max-min problem.
    
    # Let's try a different structural approach: 
    # Treat the 11 hexagons as a rigid cluster? No, they can move relative to each other.
    # But maybe we can fix the topology (which hexagons touch which) and optimize the geometry?
    # That's hard to guess.
    
    # Let's try a global optimization with differential_evolution on the positions, 
    # but with a much better objective function that accurately computes the gap.
    # The current seed used a crude circle approximation.
    # We will use the exact polygon gap in the objective, but it's expensive.
    # To speed it up, we can pre-compute the edge normals.
    
    # Actually, for fixed orientation hexagons, the gap is determined by the relative position.
    # Let's use a smarter local search: 
    # Start from a good initial guess (e.g. the hexagonal lattice packing).
    # Use Nelder-Mead to minimize the side length, with a hard constraint on gap.
    # If gap < 0, return a large penalty.
    
    # The key improvement: Use the EXACT gap calculation in the objective, 
    # not the circle approximation. This allows the optimizer to find 
    # configurations where hexagons interlock in ways that circles wouldn't allow, 
    # potentially leading to a tighter packing.
    
    def exact_gap(p1, p2):
        """Calculate the exact gap between two unit hexagons at p1 and p2 (fixed orientation 0)."""
        V1 = _hex_vertices(p1[0], p1[1], 1.0, 0.0)
        V2 = _hex_vertices(p2[0], p2[1], 1.0, 0.0)
        gap, _ = _gap_and_margin(V1, V2)
        return gap

    def objective(params):
        positions = params.reshape(11, 2)
        
        # 1. Overlap penalty (exact)
        min_gap = np.inf
        for i in range(11):
            for j in range(i + 1, 11):
                g = exact_gap(positions[i], positions[j])
                if g < min_gap:
                    min_gap = g
        
        # Strong penalty for overlap
        penalty = 0.0
        if min_gap < 0:
            penalty = -min_gap * 1000.0
        
        # 2. Size penalty
        side = _min_side_for_positions(positions)
        
        return side + penalty

    # Run multiple restarts
    num_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            pos = init_positions.copy()
        else:
            # Add noise to the initial positions
            # The noise should be small enough to stay in the same "basin" but large enough to escape local minima
            pos = init_positions + rng.normal(0, 0.3, size=(11, 2))
        
        # Use Nelder-Mead
        res = minimize(objective, pos.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if res.fun < 1000: # Check if it's not just a huge penalty
            final_pos = res.x.reshape(11, 2)
            
            # Reconstruct inner
            inner = np.column_stack((final_pos[:, 0], final_pos[:, 1], np.zeros(11)))
            
            # Determine outer
            cx, cy = final_pos.mean(axis=0)
            side = _min_side_for_positions(final_pos)
            outer = np.array([cx, cy, 0.0])
            
            # Validate
            if _all_valid(inner, outer, side):
                val = 1.0 / side
                if val > best_val:
                    best_val = val
                    best_result = (inner, outer, side)
                    # Update current best positions for next restart? 
                    # Maybe keep the best found so far as the new initial guess for subsequent restarts?
                    # This is a form of "memetic" algorithm.
                    init_positions = final_pos.copy()
                    # But we also want to explore other regions, so we keep the original too?
                    # Let's just keep the best found as the new center for noise.

    if best_result is None:
        # Fallback
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0

    return best_result

# EVOLVE-BLOCK-END
