import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
DEADLINE = time.time() + 145.0  # Safety margin for 150s limit

# Precompute base vertices for a unit hexagon (side 1)
# Vertices at angles 0, 60, ..., 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_vertices(cx, cy, angle_rad):
    """
    Returns (6, 2) vertices for a unit hexagon.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ BASE_VERTS.T).T + np.array([cx, cy])

def required_outer_side(verts):
    """
    Compute the minimum side length of a regular hexagon (centered at origin, 
    aligned with axes) that encloses the given vertices.
    """
    if len(verts) == 0:
        return 0.0
    
    x_coords = verts[:, 0]
    y_coords = verts[:, 1]
    
    # The regular hexagon with side length S and rotation 0 has support function:
    # h(theta) = S * max(cos(theta), cos(theta-60), cos(theta-120), ...)
    # Actually, it's easier to use the 6 linear inequalities defining the hexagon.
    # A regular hexagon centered at (0,0) with side length S and rotation 0 is defined by:
    # |x| <= S * sqrt(3)/2
    # |y| <= S
    # |x/sqrt(3) + y/2| <= S * sqrt(3)/2 ? No.
    
    # Let's use the standard form.
    # Vertices of outer hexagon (side S, rot 0):
    # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    
    # The half-widths in the 6 normal directions are:
    # n1 = (1, 0) -> dist = S
    # n2 = (1/2, sqrt(3)/2) -> dist = S * sqrt(3)/2
    # n3 = (-1/2, sqrt(3)/2) -> dist = S * sqrt(3)/2
    # n4 = (-1, 0) -> dist = S
    # n5 = (-1/2, -sqrt(3)/2) -> dist = S * sqrt(3)/2
    # n6 = (1/2, -sqrt(3)/2) -> dist = S * sqrt(3)/2
    
    # For a point (x,y) to be inside, we need:
    # x <= S
    # x/2 + y*sqrt(3)/2 <= S * sqrt(3)/2  => x + y*sqrt(3) <= 3S
    # -x/2 + y*sqrt(3)/2 <= S * sqrt(3)/2 => -x + y*sqrt(3) <= 3S
    # -x <= S => x >= -S
    # -x/2 - y*sqrt(3)/2 <= S * sqrt(3)/2 => -x - y*sqrt(3) <= 3S
    # x/2 - y*sqrt(3)/2 <= S * sqrt(3)/2 => x - y*sqrt(3) <= 3S
    
    # So S must satisfy:
    # S >= x
    # S >= -x
    # S >= (x + y*sqrt(3)) / 3
    # S >= (-x + y*sqrt(3)) / 3
    # S >= (-x - y*sqrt(3)) / 3
    # S >= (x - y*sqrt(3)) / 3
    
    # Taking max over all vertices.
    s1 = np.max(x_coords)
    s2 = np.max(-x_coords)
    s3 = np.max((x_coords + y_coords * SQRT3) / 3.0)
    s4 = np.max((-x_coords + y_coords * SQRT3) / 3.0)
    s5 = np.max((-x_coords - y_coords * SQRT3) / 3.0)
    s6 = np.max((x_coords - y_coords * SQRT3) / 3.0)
    
    return max(s1, s2, s3, s4, s5, s6)

def check_validity(verts_list):
    """
    Check if the list of hexagons (each a (6,2) array of vertices) are disjoint.
    Uses SAT for distance.
    """
    n_hex = len(verts_list)
    for i in range(n_hex):
        for j in range(i+1, n_hex):
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # Early exit: if center distance > 4, skip
            c1 = v1.mean(axis=0)
            c2 = v2.mean(axis=0)
            if np.linalg.norm(c1 - c2) > 4.0:
                continue
                
            # SAT
            min_gap = np.inf
            axes = []
            for k in range(6):
                e1 = v1[(k+1)%6] - v1[k]
                n1 = np.array([-e1[1], e1[0]])
                norm = np.linalg.norm(n1)
                if norm > 1e-9:
                    axes.append(n1 / norm)
                e2 = v2[(k+1)%6] - v2[k]
                n2 = np.array([-e2[1], e2[0]])
                norm = np.linalg.norm(n2)
                if norm > 1e-9:
                    axes.append(n2 / norm)
            
            for n in axes:
                proj1 = v1.dot(n)
                proj2 = v2.dot(n)
                max1, min1 = np.max(proj1), np.min(proj1)
                max2, min2 = np.max(proj2), np.min(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = 0.0
                    
                if gap < min_gap:
                    min_gap = gap
                    
            # If min_gap < 0, they overlap. 
            # Note: SAT min_gap is 0 if they touch or overlap. 
            # We need to distinguish. 
            # If min_gap == 0, check if any vertex is strictly inside.
            if min_gap <= 1e-9:
                # Check overlap
                # Simple check: if any vertex of v1 is inside v2
                # Or just use a small negative threshold for "overlap" in penalty?
                # For validity check, we want strict disjointness.
                # If min_gap is 0, they touch. Is touching allowed? "disjoint" usually means interiors don't intersect.
                # The task says "disjoint". Usually in packing, touching is allowed.
                # However, the current program's penalty was for overlap.
                # Let's assume touching is valid (distance >= 0).
                # But to be safe against numerical noise, we might want a small buffer?
                # No, let's stick to min_gap >= -1e-9.
                if min_gap < -1e-9:
                    return False
                    
    return True

def objective_symmetric(params):
    """
    params: [r1, r2, r3]
    r1: distance from center to ring 1 (6 hexagons)
    r2: distance from center to ring 2 (5 hexagons)
    r3: distance from center to ring 3? No, we have 1+6+5=12.
    Wait, 1 center + 6 ring 1 + 5 ring 2 = 12.
    So params are [r1, r2].
    But the 5 hexagons in ring 2 are not symmetrically placed like ring 1.
    In a hexagonal packing, the second ring has 12 spots. We pick 5 of them?
    Or is the structure 1 + 6 + 5?
    The 5 hexagons in the second ring are placed at angles 0, 72, 144, 216, 288? No, that breaks hex symmetry.
    
    Let's look at the inspiration: "1-center + 6-ring-1 + 5-ring-2".
    The 5 hexagons in ring 2 are likely placed to maintain as much symmetry as possible or just packed tightly.
    Actually, a common pattern for 12 is 1 center, 6 around, and 5 more around the outside.
    The 5 outer ones are placed at 5 of the 6 "valleys" or "corners" of the inner ring?
    Or maybe they are placed at angles 0, 60, 120, 180, 240? That would be 5 of the 6 directions.
    Let's assume the 5 hexagons are at angles 0, 60, 120, 180, 240 relative to the center, at distance r2.
    The 6 hexagons in ring 1 are at angles 0, 60, ..., 300, at distance r1.
    
    This configuration has 5-fold symmetry? No, it has reflection symmetry but not 6-fold.
    Wait, if we put 5 hexagons at 0, 60, 120, 180, 240, we break the 6-fold symmetry.
    Is there a better symmetric arrangement?
    Maybe the 12 hexagons are arranged in a 3x4 grid? No.
    
    Let's stick to the structural search idea:
    We will define the positions of all 12 hexagons based on a few parameters.
    Structure:
    - Hex 0: Center (0,0)
    - Hex 1-6: Ring 1, distance r1, angles 0, 60, ..., 300.
    - Hex 7-11: Ring 2, distance r2, angles 0, 60, 120, 180, 240. (One spot missing, say at 300).
    
    This creates a "C" shape or almost a full ring.
    To optimize, we can also allow the ring 2 angles to vary slightly?
    Or just optimize r1 and r2.
    
    Let's try optimizing r1 and r2.
    """
    r1, r2 = params[0], params[1]
    
    verts_list = []
    all_verts = []
    
    # Center
    v0 = get_vertices(0, 0, 0)
    verts_list.append(v0)
    all_verts.append(v0)
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        # Rotate hexagon to align with radial direction?
        # Usually, in hex packing, hexagons are aligned with the lattice.
        # Lattice angle is 0. So all hexagons have angle 0.
        v = get_vertices(x, y, 0)
        verts_list.append(v)
        all_verts.append(v)
        
    # Ring 2 (5 hexagons)
    for i in range(5):
        angle = i * np.pi / 3
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        v = get_vertices(x, y, 0)
        verts_list.append(v)
        all_verts.append(v)
        
    # Check validity
    if not check_validity(verts_list):
        # Penalty for overlap
        # Calculate overlap depth
        penalty = 0.0
        for i in range(len(verts_list)):
            for j in range(i+1, len(verts_list)):
                v1 = verts_list[i]
                v2 = verts_list[j]
                c1 = v1.mean(axis=0)
                c2 = v2.mean(axis=0)
                if np.linalg.norm(c1 - c2) > 4.0:
                    continue
                # SAT
                min_gap = np.inf
                axes = []
                for k in range(6):
                    e1 = v1[(k+1)%6] - v1[k]
                    n1 = np.array([-e1[1], e1[0]])
                    norm = np.linalg.norm(n1)
                    if norm > 1e-9: axes.append(n1 / norm)
                    e2 = v2[(k+1)%6] - v2[k]
                    n2 = np.array([-e2[1], e2[0]])
                    norm = np.linalg.norm(n2)
                    if norm > 1e-9: axes.append(n2 / norm)
                for n in axes:
                    proj1 = v1.dot(n)
                    proj2 = v2.dot(n)
                    max1, min1 = np.max(proj1), np.min(proj1)
                    max2, min2 = np.max(proj2), np.min(proj2)
                    if max1 < min2: gap = min2 - max1
                    elif max2 < min1: gap = min1 - max2
                    else: gap = 0.0
                    if gap < min_gap: min_gap = gap
                if min_gap < 0:
                    penalty += (-min_gap)**2
        return 100.0 * penalty
        
    # Compute outer side length
    stacked_verts = np.vstack(all_verts)
    side = required_outer_side(stacked_verts)
    return side

def construct():
    start_time = time.time()
    
    # Initial guess
    # r1 should be around sqrt(3) ~ 1.732 for touching ring 1
    # r2 should be around 2*sqrt(3) ~ 3.464 for touching ring 2
    # But we want to minimize the outer hexagon, so we pack them tighter.
    # The optimal r1 is likely slightly less than sqrt(3) if we allow rotation?
    # But we fixed angle to 0.
    
    # Let's run Powell optimization
    try:
        res = minimize(objective_symmetric, [1.7, 3.4], method='Powell', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if time.time() > DEADLINE:
            # Fallback to initial guess if timeout
            best_params = [1.7, 3.4]
        else:
            best_params = res.x
    except Exception:
        best_params = [1.7, 3.4]
        
    # Finalize with best_params
    r1, r2 = best_params
    
    inner = []
    all_verts = []
    
    # Center
    v0 = get_vertices(0, 0, 0)
    inner.append([0.0, 0.0, 0.0])
    all_verts.append(v0)
    
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        x = r1 * np.cos(angle)
        y = r1 * np.sin(angle)
        inner.append([float(x), float(y), 0.0])
        v = get_vertices(x, y, 0)
        all_verts.append(v)
        
    # Ring 2
    for i in range(5):
        angle = i * np.pi / 3
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        inner.append([float(x), float(y), 0.0])
        v = get_vertices(x, y, 0)
        all_verts.append(v)
        
    # Compute final side length
    stacked_verts = np.vstack(all_verts)
    side_len = required_outer_side(stacked_verts)
    
    # Center of outer hexagon is (0,0)
    # Rotation is 0
    
    return inner, (0.0, 0.0), float(side_len), 0.0

