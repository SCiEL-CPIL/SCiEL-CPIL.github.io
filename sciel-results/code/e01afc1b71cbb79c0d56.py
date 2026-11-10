# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
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

def _outer_side_for_points(pts):
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 361):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T
        maxproj = proj.max(axis=0)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_a = a_deg
    return best_s, best_a

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

def _compute_min_s_for_positions(positions, angles):
    """
    positions: (N, 2) array of centers relative to centroid.
    angles: (N,) array of angles.
    Returns (s, angle_deg) of minimal enclosing hexagon.
    """
    # We need to find the rotation of the container relative to the points.
    # The points are fixed in this function's frame, but we can rotate the points
    # or the container. Rotating the container by alpha is equivalent to rotating
    # the points by -alpha.
    # s(alpha) = 2/sqrt(3) * max_k (max_i (p_i rotated by -alpha . n_k(alpha)))
    # Actually, simpler: 
    # Container normal k is at angle a + 60k.
    # Projection of p_i on normal k is p_i . (cos(a+60k), sin(a+60k)).
    # We want to minimize max_k (max_i proj).
    
    def calc_s(a_deg):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = positions @ normals.T
        maxproj = proj.max(axis=0)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        return s
    
    res = minimize_scalar(calc_s, bounds=(0, 60), method='bounded', options={'xatol': 1e-4})
    best_a = res.x
    best_s = res.fun
    return best_s, best_a

def _generate_topology_13_minus_2():
    """
    Generates positions for the 13-hex flower minus 2 opposite outer hexes.
    13-hex flower:
    Center (0,0)
    Ring 1: 6 hexes at dist sqrt(3)
    Ring 2: 6 hexes at dist 2*sqrt(3) (corners)
    Total 1+6+6=13.
    We remove 2 opposite from Ring 2.
    """
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1
    for i in range(6):
        a = np.deg2rad(60 * i)
        x = np.sqrt(3) * np.cos(a)
        y = np.sqrt(3) * np.sin(a)
        centers.append([x, y])
        angles.append(0.0)
        
    # Ring 2 (Corners)
    # These are at distance 2*sqrt(3) from center, same angles as Ring 1?
    # No, the "corners" of the hexagonal grid are at (2, 0), (1, sqrt(3)), etc in axial coords?
    # Let's use cartesian.
    # Ring 1 centers: (sqrt(3), 0), (sqrt(3)/2, 3/2), ...
    # The hexes in Ring 2 touch Ring 1 hexes.
    # Distance between Ring 1 center and Ring 2 center is sqrt(3).
    # Ring 2 centers are at distance 2*sqrt(3) from origin?
    # Let's check: (sqrt(3), 0) is Ring 1.
    # Neighbor in direction (1, 0) is at (sqrt(3)+sqrt(3), 0) = (2*sqrt(3), 0).
    # Yes.
    # So Ring 2 centers are at distance 2*sqrt(3) at angles 0, 60, 120, 180, 240, 300.
    
    for i in range(6):
        a = np.deg2rad(60 * i)
        x = 2.0 * np.sqrt(3) * np.cos(a)
        y = 2.0 * np.sqrt(3) * np.sin(a)
        centers.append([x, y])
        angles.append(0.0)
        
    # We have 13 centers.
    # Remove 2 opposite ones from Ring 2 (indices 7 to 12).
    # Let's remove indices 7 and 10 (0 and 180 degrees).
    # Indices: 0:Center, 1-6:Ring1, 7-12:Ring2.
    # Remove 7 (angle 0) and 10 (angle 180).
    keep = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12]
    
    centers = [centers[i] for i in keep]
    angles = [angles[i] for i in keep]
    
    return np.array(centers), np.array(angles)

def _generate_topology_7_plus_4_gaps():
    """
    7 hex flower + 4 hexes in 4 of the 6 gaps between Ring 1 hexes.
    """
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1
    r1_pos = []
    for i in range(6):
        a = np.deg2rad(60 * i)
        x = np.sqrt(3) * np.cos(a)
        y = np.sqrt(3) * np.sin(a)
        r1_pos.append([x, y])
        centers.append([x, y])
        angles.append(0.0)
        
    # Gaps are between Ring 1 hexes.
    # Gap i is between Ring 1[i] and Ring 1[(i+1)%6].
    # The center of the hex in the gap is at distance 2*sqrt(3) from origin?
    # No, the gap is "outside" the ring.
    # Let's calculate the position of a hex touching Ring 1[i] and Ring 1[i+1].
    # Let C1 = r1_pos[i], C2 = r1_pos[i+1].
    # The new center C3 must be at distance sqrt(3) from C1 and C2.
    # C1 and C2 are at distance sqrt(3) from origin. Angle between them is 60 deg.
    # Triangle O-C1-C2 is equilateral (side sqrt(3)).
    # So C3 is at distance 2*sqrt(3) from origin?
    # Vector C1 + C2 has magnitude sqrt(3)*sqrt(3) = 3? No.
    # |C1| = sqrt(3), |C2| = sqrt(3), angle 60.
    # |C1+C2| = sqrt(3+3+2*3*0.5) = sqrt(9) = 3.
    # Direction is bisector.
    # C3 is the other vertex of the equilateral triangle formed by C1, C2, and... ?
    # No, C1, C2, C3 form a triangle where all sides are sqrt(3).
    # So C3 is at distance sqrt(3) from C1 and C2.
    # The origin O is also at distance sqrt(3) from C1 and C2.
    # So O, C1, C3, C2 form a rhombus?
    # Yes, O-C1-C3-C2 is a rhombus with side sqrt(3).
    # So C3 = C1 + C2 - O = C1 + C2.
    # So C3 is at C1 + C2.
    # Magnitude of C3 is 3.
    # Angle is bisector of C1 and C2.
    
    # We want to place 4 hexes in 4 of the 6 gaps.
    # To maintain symmetry, let's pick gaps 0, 1, 3, 4? Or 0, 2, 4, ?
    # Let's try picking 4 consecutive gaps? Or alternating?
    # Let's try all combinations of 4 gaps out of 6.
    # But for now, let's just generate one specific topology: Gaps 0, 1, 3, 4.
    # Actually, let's just return the 7 hexes and let the caller add the others?
    # No, let's generate the full set.
    
    # Let's pick gaps at indices 0, 1, 3, 4. (Leaving 2 and 5 empty).
    # Gap 0: between R1[0] and R1[1]. Center = R1[0] + R1[1].
    # Gap 1: between R1[1] and R1[2]. Center = R1[1] + R1[2].
    # Gap 3: between R1[3] and R1[4]. Center = R1[3] + R1[4].
    # Gap 4: between R1[4] and R1[5]. Center = R1[4] + R1[5].
    
    gap_indices = [0, 1, 3, 4]
    for i in gap_indices:
        c1 = r1_pos[i]
        c2 = r1_pos[(i+1)%6]
        c3 = [c1[0] + c2[0], c1[1] + c2[1]]
        centers.append(c3)
        angles.append(0.0)
        
    return np.array(centers), np.array(angles)

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 600.0

    best_val = -np.inf
    best_inner = None
    best_outer = None
    best_side = None

    topologies = []
    
    # Topology 1: 13-2
    try:
        c, a = _generate_topology_13_minus_2()
        topologies.append((c, a))
    except Exception:
        pass
        
    # Topology 2: 7+4 gaps (one specific combination)
    try:
        c, a = _generate_topology_7_plus_4_gaps()
        topologies.append((c, a))
    except Exception:
        pass
        
    # Try other combinations of 4 gaps for Topology 2
    import itertools
    for combo in itertools.combinations(range(6), 4):
        # Skip the one already added if it matches
        if list(combo) == [0, 1, 3, 4]:
            continue
        try:
            centers = []
            angles = []
            centers.append([0.0, 0.0])
            angles.append(0.0)
            r1_pos = []
            for i in range(6):
                ang = np.deg2rad(60 * i)
                x = np.sqrt(3) * np.cos(ang)
                y = np.sqrt(3) * np.sin(ang)
                r1_pos.append([x, y])
                centers.append([x, y])
                angles.append(0.0)
                
            for i in combo:
                c1 = r1_pos[i]
                c2 = r1_pos[(i+1)%6]
                c3 = [c1[0] + c2[0], c1[1] + c2[1]]
                centers.append(c3)
                angles.append(0.0)
                
            topologies.append((np.array(centers), np.array(angles)))
        except Exception:
            pass

    for centers, angles in topologies:
        if time.time() > deadline - 10.0:
            break
            
        # Center the positions
        mean = centers.mean(axis=0)
        centers_rel = centers - mean
        
        s, a_deg = _compute_min_s_for_positions(centers_rel, angles)
        
        # Construct inner and outer
        inner = np.column_stack((centers_rel[:, 0], centers_rel[:, 1], angles))
        # Outer center is at 0,0 in this frame
        outer = np.array([0.0, 0.0, a_deg])
        
        if _all_valid(inner, outer, s):
            val = 1.0 / s
            if val > best_val:
                best_val = val
                best_inner = inner
                best_outer = outer
                best_side = s

    if best_inner is not None:
        return best_inner, best_outer, float(best_side)
        
    # Fallback to seed program logic if topology search fails
    # (Reusing the simple seed from the original code)
    base = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [-2.0, 0.0],
        [1.0, 1.732],
        [-1.0, 1.732],
        [1.0, -1.732],
        [-1.0, -1.732],
        [3.0, 1.732],
        [-3.0, 1.732],
        [3.0, -1.732],
        [-3.0, -1.732],
    ])
    inner = np.column_stack((base[:, 0], base[:, 1], np.zeros(11)))
    s, a_out = _outer_side_for_points(base)
    outer = np.array([0.0, 0.0, a_out])
    return inner, outer, float(s)

# EVOLVE-BLOCK-END
