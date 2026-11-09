# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 630.0 - 30.0  # safety margin

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    return np.array([
        [cx + d * np.cos(a + np.deg2rad(60 * i)),
         cy + d * np.sin(a + np.deg2rad(60 * i))] for i in range(6)
    ])

def _edge_normals(vertices):
    n = np.empty_like(vertices)
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        e = p2 - p1
        n[i] = np.array([e[1], -e[0]])
    return n

def _separation(v1, v2):
    # Returns minimum gap (positive if separated) using SAT on 12 edge normals
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    best = np.inf
    for normals in (n1, n2):
        for axis in normals:
            p1 = v1 @ axis
            p2 = v2 @ axis
            lo1, hi1 = p1.min(), p1.max()
            lo2, hi2 = p2.min(), p2.max()
            gap = min(hi1, hi2) - max(lo1, lo2)
            if gap < best:
                best = gap
    return best

def _inside_outer(v, outer_v):
    # For every outer edge (p1,p2) in vertex order, cross(p2-p1, v-p1) >= -1e-6
    for i in range(6):
        p1 = outer_v[i]
        p2 = outer_v[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _check_packing(inner, outer, s):
    # inner: (12,3) (x,y,angle_deg), outer: (3,), s: float
    # Returns (valid: bool, penalty: float)
    outer_v = _hex_vertices(outer[0], outer[1], s, outer[2])
    # distinct outer vertices
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_v[i] - outer_v[j]) < 1e-9:
                return False, 1.0
    # unit edge length check for each inner hexagon
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False, 1.0
    # containment
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        for vert in v:
            if not _inside_outer(vert, outer_v):
                return False, 1.0
    # separation
    for i in range(12):
        for j in range(i + 1, 12):
            vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _separation(vi, vj)
            if gap <= 1e-6:
                return False, 1.0
    return True, 0.0

def _min_outer_side(inner):
    # Compute minimal side length of a regular hexagon (centered at origin, angle 0)
    # that contains all inner hexagons.
    # A regular hexagon with side s, center (0,0), angle 0 has inradius = s * sqrt(3)/2.
    # The distance from center to any point must be <= s * sqrt(3)/2? No, that's for a circle.
    # For a hexagon, the constraint is that for each edge, the point must be on the inside.
    # Edge normals for hexagon with side s, angle 0:
    # Vertices: s*(cos(60i), sin(60i)) for i=0..5
    # Actually, the problem says vertices are (cx + d cos(a + 60i), ...).
    # So for d=s, a=0, vertices are at angles 0, 60, 120, 180, 240, 300.
    # The edges are between these vertices.
    # The normal to the edge between vertex 0 and 1 (angles 0 and 60) points in direction 30 degrees.
    # The distance from center to this edge is the inradius: s * cos(30) = s * sqrt(3)/2.
    # So the constraint for a point p to be inside is: p . n <= inradius for all 6 normals.
    # The 6 normals are at angles 30, 90, 150, 210, 270, 330.
    # So max over all inner vertices of (v . n) for all 6 normals must be <= s * sqrt(3)/2.
    # Thus s >= 2/sqrt(3) * max(v . n).
    
    normals = []
    for i in range(6):
        angle = np.deg2rad(30 + 60 * i)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        for vert in v:
            projs = vert @ normals.T
            max_proj = max(max_proj, projs.max())
    return 2.0 / np.sqrt(3.0) * max_proj

def hexagon_packing_12():
    # 1. Initial layout: Nested rings
    # Center: 1
    # Ring 1: 6
    # Ring 2: 5 (to make 12 total)
    # In a hexagonal grid, the distance between centers of adjacent hexagons is sqrt(3).
    # Let's place them in a hexagonal grid pattern.
    
    inner = np.zeros((12, 3))
    inner[0] = [0, 0, 0]
    
    # Ring 1
    r1 = np.sqrt(3)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        inner[i + 1, 0] = r1 * np.cos(angle)
        inner[i + 1, 1] = r1 * np.sin(angle)
        inner[i + 1, 2] = 0
    
    # Ring 2: 5 hexagons
    # In a perfect hex grid, ring 2 has 12 hexagons. We pick 5 of them.
    # To be symmetric, we might pick 5 that are spread out.
    # Or just pick the first 5 in angular order.
    # Let's try to pick them to be as symmetric as possible.
    # Maybe 0, 72, 144, 216, 288?
    # But the grid is 6-fold symmetric.
    # Let's just pick 5 adjacent ones or spread them.
    # Actually, a "flower" of 7 is 1 center + 6 ring.
    # Adding 5 more to make 12.
    # Let's place them in ring 2, skipping 7 of the 12 positions.
    # Positions in ring 2:
    # (2, 0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3)) are ring 1.
    # Ring 2 positions (distance 2*sqrt(3) from center? No, next shell).
    # In hex grid, neighbors of neighbors.
    # Let's just use the standard hex grid coordinates.
    # q, r -> x = sqrt(3)*(q + r/2), y = 1.5*r
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2) ... wait, that's 12.
    # Let's pick 5 from Ring 2.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) -> 5 hexagons.
    # This is not symmetric.
    # Maybe pick (2,0), (0,2), (-2,1), (-2,-1), (0,-2)?
    # Let's just pick 5 and let the optimizer fix it.
    # For simplicity, let's pick 5 spread around.
    # Indices 0, 2, 4, 6, 8 of the 12 ring-2 positions.
    
    ring2_coords = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
        (-1, -1), (0, -2), (1, -2), (2, -1), (2, 1), (1, 2)
    ]
    # Pick 0, 2, 4, 6, 8
    for i, idx in enumerate([0, 2, 4, 6, 8]):
        q, r = ring2_coords[idx]
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner[7 + i, 0] = x
        inner[7 + i, 1] = y
        inner[7 + i, 2] = 0

    outer = np.array([0.0, 0.0, 0.0])
    s = _min_outer_side(inner)
    
    valid, _ = _check_packing(inner, outer, s)
    best_score = 1.0 / s if valid else 0.0
    best_inner = inner.copy()
    best_outer = outer.copy()
    best_s = s

    if _time_left() < 10:
        return best_inner, best_outer, best_s

    # 2. Optimization
    # Variables: inner (12*3=36), outer (3), s (1) -> 40 vars
    # Objective: minimize s
    # Constraints: separation >= 1e-6, containment >= -1e-6
    
    def objective(x):
        # x: [inner_flat (36), outer (3), s (1)]
        inner_flat = x[:36].reshape(12, 3)
        outer_flat = x[36:39]
        s_val = x[39]
        
        # Penalize if s is too small (invalid)
        valid, _ = _check_packing(inner_flat, outer_flat, s_val)
        if valid:
            return s_val
        else:
            return 1e6  # Large penalty
    
    # This is a non-convex problem. Let's use a simple local search or Nelder-Mead.
    # Nelder-Mead might be slow for 40 vars.
    # Let's try to optimize only s, keeping inner fixed? No, that's just _min_outer_side.
    # We need to move inner hexagons to reduce s.
    
    # Let's use a simple gradient-free optimizer or just try a few random perturbations.
    # Given the time limit, let's try a short run of Nelder-Mead on a subset of variables?
    # Or just accept the initial layout if it's good enough.
    
    # The initial layout (1 center, 6 ring, 5 ring2) has s = ?
    # Max proj for ring 2 (0,2) -> y=3. x=0.
    # Vertices of hex at (0,3): (0, 3+1), etc.
    # Max y is 4.
    # Normal at 90 deg is (0,1). Proj is 4.
    # s = 2/sqrt(3) * 4 = 8/sqrt(3) ~ 4.618.
    # 1/s ~ 0.216.
    # Seed score was 1.0 with s=8. 1/8 = 0.125.
    # So 0.216 is better than 0.125.
    # Wait, the score is 1/s.
    # Seed: s=8, score=0.125.
    # My layout: s=4.618, score=0.216.
    # This is already an improvement!
    
    # Let's verify validity.
    # Separation:
    # Center (0,0) and (0, sqrt(3)) are adjacent. Distance sqrt(3).
    # Hexagons of side 1 in hex grid touch if distance is sqrt(3).
    # So they are tangent. Gap is 0.
    # The check requires gap > 1e-6.
    # So they are INVALID (touching).
    # We need to separate them slightly or scale up.
    # If we scale up, s increases.
    # If we move them apart, s might increase.
    # The minimal packing of 12 hexagons likely requires them to touch.
    # But the contract says "two inner hexagons not separated: ... projection intervals more than 1e-6 apart".
    # So they must be strictly separated.
    # This means we can't have a perfect tiling. We must have gaps.
    # So we need to expand the layout slightly.
    
    # Let's add a small epsilon to the scale of the inner layout.
    # Or just increase s to satisfy the gap.
    
    # Let's try to optimize s and positions to minimize s while keeping gap >= 1e-6.
    
    # Since Nelder-Mead on 40 vars is risky, let's try a simpler approach:
    # Scale the inner layout by a factor (1 + eps) until valid.
    
    for eps in np.linspace(0.0, 0.1, 100):
        inner_scaled = inner.copy()
        inner_scaled[:, :2] *= (1 + eps)
        s_candidate = _min_outer_side(inner_scaled)
        valid, _ = _check_packing(inner_scaled, outer, s_candidate)
        if valid:
            best_inner = inner_scaled
            best_s = s_candidate
            best_score = 1.0 / best_s
            break

    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
