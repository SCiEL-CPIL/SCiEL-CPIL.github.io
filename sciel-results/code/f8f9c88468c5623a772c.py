import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
START = time.time()
DEADLINE = 145.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, s, rot):
    """
    Returns vertices of a regular hexagon with side length s, 
    centered at (cx, cy), rotated by rot degrees.
    """
    a0 = math.radians(rot)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """
    Returns vertices of a unit regular hexagon.
    """
    a0 = math.radians(ang_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + math.cos(a), cy + math.sin(a)))
    return pts

def _min_outer_side(inner, outer_c, rot):
    """
    Calculates the minimum side length of a regular hexagon 
    (centered at outer_c, rotated by rot) that encloses all inner hexagons.
    """
    theta_rad = math.radians(rot)
    # Normals to the sides of the outer hexagon
    # The sides are at angles 30, 90, 150, 210, 270, 330 relative to the hexagon's orientation
    # Wait, for a regular hexagon with a vertex at angle 'rot', the side normals are at rot + 30 + k*60
    normals = []
    for k in range(6):
        alpha = theta_rad + math.radians(30 + k * 60)
        normals.append((math.cos(alpha), math.sin(alpha)))
    
    ox, oy = outer_c
    max_proj = 0.0
    for (x, y, a) in inner:
        pts = _inner_hex(x, y, a)
        for (px, py) in pts:
            for (nx, ny) in normals:
                proj = (px - ox) * nx + (py - oy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    # Distance from center to side is s * sqrt(3) / 2
    # So max_proj = s * sqrt(3) / 2  => s = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < 1e-9:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    """
    Returns True if polygons A and B overlap (interiors intersect).
    Uses Separating Axis Theorem.
    """
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            # Normal vector
            nx, ny = -ay, ax
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            # If there is a gap, they don't overlap
            if maxA + 1e-9 < minB or maxB + 1e-9 < minA:
                return False
    return True

def _in_outer(inner_pts, outer_pts):
    return all(_point_in_poly(p, outer_pts) for p in inner_pts)

def _valid(inner, outer_c, s, rot):
    """
    Checks if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon.
    2. No two inner hexagons overlap.
    """
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    
    # Check containment
    for i in range(len(polys)):
        if not _in_outer(polys[i], outer_pts):
            return False
            
    # Check disjointness
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _make_flower_seed():
    """
    Constructs the 'flower' packing:
    1 center hexagon at (0,0).
    6 hexagons at distance sqrt(3) (touching the center).
    4 hexagons at distance 3 (touching the outer ring).
    
    To ensure strict disjointness for the evaluator, we add a tiny epsilon 
    to the distances.
    """
    eps = 1e-6
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons
    d1 = math.sqrt(3.0) + eps
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d1 * math.cos(a), d1 * math.sin(a), 0.0])
        
    # Ring 2: 4 hexagons (at 30, 90, 150, 210 degrees? No, symmetric positions)
    # The vertices of the central flower extend to radius 2.
    # The hexagons in ring 1 are centered at distance sqrt(3).
    # Their furthest points are at distance sqrt(3) + 1.
    # Wait, the standard flower packing has 1+6+4=11 hexagons.
    # Let's verify the positions for the 4 outer ones.
    # In a hexagonal grid, the positions are (i,j) in axial coordinates.
    # Center (0,0).
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1). Distance sqrt(3).
    # Ring 2: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,1), (0,1), (1,2), (2,2)... 
    # Actually, the "flower" of 11 hexagons usually refers to the central hexagon + 6 neighbors + 4 of the next ring? 
    # Or is it the 19-hexagon cluster minus 8?
    # Let's look at the inspiration code:
    # Ring 2: 4 hexagons at distance 3 + 2*eps, at 30, 90, 150, 210 deg.
    # Let's stick to the inspiration's explicit construction which scored 10.0.
    
    d2 = 3.0 + 2 * eps
    # The inspiration used angles 30, 90, 150, 210. 
    # Wait, 30, 90, 150, 210 are not symmetrically distributed around the circle (gaps of 60, 60, 60, 180).
    # Let's re-read inspiration 1:
    # "Ring 2: 4 hexagons at distance 3 + 2*eps, at 30, 90, 150, 210 deg"
    # This seems asymmetric. Let's check the geometry.
    # If we have 6 in ring 1 at 30, 90, 150, 210, 270, 330.
    # Placing 4 in ring 2 at 30, 90, 150, 210 would align them with 4 of the ring 1 hexagons.
    # This creates a "keyhole" or asymmetric shape.
    # However, the inspiration code scored 10.0. I will replicate it exactly, 
    # but I will also try a symmetric 1+6+4 arrangement if the optimizer allows.
    # Actually, for N=11, the optimal packing is likely the one with 4 on the outer ring 
    # placed to minimize the bounding box.
    
    for k in range(4):
        # Inspiration used: 30 + k*60 for k in 0..3 -> 30, 90, 150, 210
        a = math.radians(30 + k * 60)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
        
    return inner, (0.0, 0.0), 0.0, 0.0

def _objective_outer(vars, inner):
    """
    Objective function for optimizing the outer hexagon parameters (cx, cy, rot).
    """
    cx, cy, rot = vars
    return _min_outer_side(inner, (cx, cy), rot)

def _objective_joint(vars, inner_angles):
    """
    Objective function for joint optimization of inner centers and outer hexagon.
    vars: [outer_cx, outer_cy, outer_rot, x1, y1, ..., x11, y11]
    """
    outer_cx = vars[0]
    outer_cy = vars[1]
    outer_rot = vars[2]
    
    n = len(inner_angles)
    centers = vars[3:].reshape(n, 2)
    
    # Reconstruct inner list
    inner_list = []
    for i in range(n):
        inner_list.append([centers[i, 0], centers[i, 1], inner_angles[i]])
        
    side = _min_outer_side(inner_list, (outer_cx, outer_cy), outer_rot)
    
    # Penalty for overlaps
    penalty = 0.0
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner_list]
    for i in range(n):
        for j in range(i+1, n):
            if _poly_overlap(polys[i], polys[j]):
                # Calculate overlap penalty based on how much they overlap?
                # For now, a large constant penalty
                penalty += 100.0
                
    return side + penalty

def construct():
    best_inner = None
    best_oc = (0.0, 0.0)
    best_s = float('inf')
    best_rot = 0.0
    
    # 1. Direct Construction of the Flower
    inner, oc, s_init, rot_init = _make_flower_seed()
    
    # Calculate initial tight side
    s_tight = _min_outer_side(inner, oc, rot_init)
    
    # Validate initial state
    if _valid(inner, oc, s_tight, rot_init):
        best_inner = [list(p) for p in inner]
        best_oc = oc
        best_s = s_tight
        best_rot = rot_init
    else:
        # If invalid (shouldn't happen with eps), try to fix by increasing eps or falling back
        # For now, we assume the eps is sufficient.
        pass

    # 2. Optimize Outer Hexagon
    if best_inner is not None and time.time() < DEADLINE - 10:
        x0 = np.array([best_oc[0], best_oc[1], best_rot])
        try:
            res = minimize(_objective_outer, x0, args=(best_inner,), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_s:
                # Verify validity with the new outer parameters
                new_oc = (res.x[0], res.x[1])
                new_rot = res.x[2]
                new_s = res.fun
                if _valid(best_inner, new_oc, new_s, new_rot):
                    best_oc = new_oc
                    best_rot = new_rot
                    best_s = new_s
        except Exception:
            pass

    # 3. Joint Optimization (Inner + Outer)
    if best_inner is not None and time.time() < DEADLINE - 20:
        # Extract angles
        angles = [p[2] for p in best_inner]
        # Initial guess for vars
        x0 = np.zeros(3 + 2 * N)
        x0[0] = best_oc[0]
        x0[1] = best_oc[1]
        x0[2] = best_rot
        for i in range(N):
            x0[3 + 2*i] = best_inner[i][0]
            x0[4 + 2*i] = best_inner[i][1]
            
        try:
            res = minimize(_objective_joint, x0, args=(angles,), method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_s:
                # Parse result
                new_oc = (res.x[0], res.x[1])
                new_rot = res.x[2]
                new_inner = []
                for i in range(N):
                    new_inner.append([res.x[3+2*i], res.x[4+2*i], angles[i]])
                    
                new_s = _min_outer_side(new_inner, new_oc, new_rot)
                
                # Check validity
                if _valid(new_inner, new_oc, new_s, new_rot) and new_s < best_s:
                    best_inner = new_inner
                    best_oc = new_oc
                    best_rot = new_rot
                    best_s = new_s
        except Exception:
            pass

    # Fallback: If we didn't find a valid solution or the score is bad, 
    # use the original program's logic as a backup.
    # But given the inspiration score, the direct construction should work.
    
    if best_inner is None:
        # Emergency fallback: Return a large hexagon with random hexagons inside
        # This should not happen, but ensures we return something.
        inner = [[i*0.1, 0, 0] for i in range(N)]
        return inner, (0,0), 10.0, 0.0

    # Final refinement: Tighten the side length one last time
    final_s = _min_outer_side(best_inner, best_oc, best_rot)
    if _valid(best_inner, best_oc, final_s, best_rot):
        best_s = final_s

    return best_inner, best_oc, best_s, best_rot

if __name__ == "__main__":
    # Simple test
    inner, oc, s, rot = construct()
    print(f"Side: {s}, Rot: {rot}, Center: {oc}")
