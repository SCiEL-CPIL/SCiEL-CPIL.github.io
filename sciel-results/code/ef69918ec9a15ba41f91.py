import numpy as np
import math
import time
import random
from itertools import combinations

N = 11
EPS = 1e-9
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin for 150s limit

def _hex_vertices(cx, cy, s, rot_deg):
    """
    Get vertices of a regular hexagon with center (cx, cy), side length s, 
    rotated by rot_deg degrees.
    """
    a0 = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _inner_hex(cx, cy, ang_deg):
    """
    Unit regular hexagon (side 1) with center (cx, cy) and rotation ang_deg.
    """
    return _hex_vertices(cx, cy, 1.0, ang_deg)

def _min_outer_side(inner, outer_c, rot_deg):
    """
    Compute the minimum side length S of a regular hexagon centered at outer_c
    with rotation rot_deg that contains all inner hexagons.
    """
    theta_rad = math.radians(rot_deg)
    # Normals to the sides of the outer hexagon
    # The sides are perpendicular to the normals.
    # For a hexagon rotated by theta, the normals are at theta + 30 + k*60.
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
    
    # Distance from center to side is S * sqrt(3) / 2
    # So S * sqrt(3) / 2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3.0)

def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _point_in_poly(pt, poly):
    """
    Check if point pt is inside convex polygon poly.
    """
    n = len(poly)
    sign = 0
    for i in range(n):
        c = _cross(poly[i], poly[(i+1) % n], pt)
        if abs(c) < EPS:
            continue
        s = 1 if c > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return True

def _poly_overlap(A, B):
    """
    SAT: return True if convex polygons A and B overlap.
    """
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            ax = poly[(i+1) % n][0] - poly[i][0]
            ay = poly[(i+1) % n][1] - poly[i][1]
            nx, ny = -ay, ax
            # Normalize normal? Not strictly needed for separation check if consistent
            minA = min(p[0]*nx + p[1]*ny for p in A)
            maxA = max(p[0]*nx + p[1]*ny for p in A)
            minB = min(p[0]*nx + p[1]*ny for p in B)
            maxB = max(p[0]*nx + p[1]*ny for p in B)
            if maxA + EPS < minB or maxB + EPS < minA:
                return False
    return True

def _valid(inner, outer_c, s, rot_deg):
    """
    Check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon.
    2. No inner hexagons overlap.
    """
    outer_pts = _hex_vertices(outer_c[0], outer_c[1], s, rot_deg)
    polys = [_inner_hex(x, y, a) for (x, y, a) in inner]
    
    # Check containment
    for i in range(len(polys)):
        for p in polys[i]:
            if not _point_in_poly(p, outer_pts):
                return False
                
    # Check overlap
    for i in range(len(polys)):
        for j in range(i+1, len(polys)):
            if _poly_overlap(polys[i], polys[j]):
                return False
    return True

def _generate_lattice_seed(rng):
    """
    Generate a seed based on the hexagonal lattice structure.
    Center + 6 neighbors + 4 from next shell.
    """
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons at distance sqrt(3)
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(30 + k * 60)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
        
    # Second ring candidates
    # Distance 2*sqrt(3) in directions 30, 90, ...
    # Or distance sqrt(3) in directions 0, 60, ... (which are the "corners" of the hex grid)
    # Let's pick 4 from the 6 positions at distance 2*sqrt(3)
    d2 = 2 * d
    candidates = []
    for k in range(6):
        a = math.radians(30 + k * 60)
        candidates.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
        
    # Pick 4 random candidates
    selected = rng.sample(candidates, 4)
    inner.extend(selected)
    
    # Add small random noise to escape local minima if lattice is not optimal
    # But start with exact lattice
    return inner

def _make_random_seed(rng):
    """
    Generate a random seed.
    """
    inner = []
    for _ in range(N):
        x = rng.uniform(-4, 4)
        y = rng.uniform(-4, 4)
        a = rng.uniform(0, 60)
        inner.append([x, y, a])
    return inner

def _try_refine(inner, outer_c, rot_deg, rng, iters, budget):
    """
    Local search to minimize outer side length.
    """
    # Initial tight side
    s = _min_outer_side(inner, outer_c, rot_deg)
    if not _valid(inner, outer_c, s, rot_deg):
        return None
    
    best_inner = [list(h) for h in inner]
    best_s = s
    best_oc = outer_c
    best_rot = rot_deg
    
    cur_inner = [list(h) for h in inner]
    cur_oc = outer_c
    cur_rot = rot_deg
    cur_s = s
    
    it = 0
    while it < iters:
        if time.time() > budget:
            break
        it += 1
        
        # Determine perturbation size based on iteration
        # Start large, get smaller
        scale = 0.5 * (1.0 - it / float(iters)) + 0.1
        
        # Perturb positions and angles
        cand_inner = []
        for (x, y, a) in cur_inner:
            nx = x + rng.uniform(-scale, scale)
            ny = y + rng.uniform(-scale, scale)
            na = a + rng.uniform(-15 * scale, 15 * scale)
            cand_inner.append([nx, ny, na])
            
        # Perturb outer center and rotation slightly
        cand_oc = (cur_oc[0] + rng.uniform(-0.2 * scale, 0.2 * scale), 
                   cur_oc[1] + rng.uniform(-0.2 * scale, 0.2 * scale))
        cand_rot = cur_rot + rng.uniform(-5 * scale, 5 * scale)
        
        # Calculate tight side for candidate
        cand_s = _min_outer_side(cand_inner, cand_oc, cand_rot)
        
        # Check validity
        if _valid(cand_inner, cand_oc, cand_s, cand_rot):
            # Accept if it improves the best
            if cand_s < best_s:
                best_inner = cand_inner
                best_s = cand_s
                best_oc = cand_oc
                best_rot = cand_rot
                
            # Accept as current if it's valid (even if not better than best, 
            # to allow exploring the landscape)
            # Simple hill climbing: only accept if better than current? 
            # Or simulated annealing? Let's stick to simple: 
            # Accept if better than current, or with some probability if worse (annealing).
            # For speed, let's try: accept if better than current.
            if cand_s < cur_s:
                cur_inner = cand_inner
                cur_oc = cand_oc
                cur_rot = cand_rot
                cur_s = cand_s
            else:
                # Occasionally accept worse moves to escape local minima
                if rng.random() < 0.1:
                    cur_inner = cand_inner
                    cur_oc = cand_oc
                    cur_rot = cand_rot
                    cur_s = cand_s

    return best_inner, best_oc, best_s, best_rot

def construct():
    start_time = time.time()
    budget = START_TIME + DEADLINE
    
    rng = random.Random(42)
    
    best_result = None
    best_score = float('inf')
    
    # 1. Try lattice-based seeds
    for i in range(10):
        if time.time() > budget - 10:
            break
        seed_inner = _generate_lattice_seed(rng)
        # Tighten side
        s = _min_outer_side(seed_inner, (0,0), 0.0)
        if _valid(seed_inner, (0,0), s, 0.0):
            res = _try_refine(seed_inner, (0,0), 0.0, rng, iters=200, budget=budget - 5)
            if res:
                inner, oc, s_val, rot = res
                if s_val < best_score:
                    best_score = s_val
                    best_result = (inner, oc, s_val, rot)
        
    # 2. Try random seeds
    for i in range(20):
        if time.time() > budget - 5:
            break
        seed_inner = _make_random_seed(rng)
        s = _min_outer_side(seed_inner, (0,0), 0.0)
        if _valid(seed_inner, (0,0), s, 0.0):
            res = _try_refine(seed_inner, (0,0), 0.0, rng, iters=100, budget=budget - 2)
            if res:
                inner, oc, s_val, rot = res
                if s_val < best_score:
                    best_score = s_val
                    best_result = (inner, oc, s_val, rot)
    
    # 3. Final polish on the best result found
    if best_result:
        inner, oc, s_val, rot = best_result
        # Run a finer search with smaller perturbations
        res = _try_refine(inner, oc, rot, rng, iters=500, budget=budget)
        if res:
            inner, oc, s_val, rot = res
    
    if best_result is None:
        # Fallback: return a valid configuration, even if not optimal
        # Use the first lattice seed
        inner = _generate_lattice_seed(rng)
        s = _min_outer_side(inner, (0,0), 0.0)
        # Ensure validity, if not, expand
        while not _valid(inner, (0,0), s, 0.0):
            s *= 1.1
        return inner, (0.0, 0.0), s, 0.0

    inner, oc, s_val, rot = best_result
    # Final tight check
    s_tight = _min_outer_side(inner, oc, rot)
    if s_tight < s_val and _valid(inner, oc, s_tight, rot):
        s_val = s_tight
        
    return inner, (oc[0], oc[1]), s_val, rot

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
