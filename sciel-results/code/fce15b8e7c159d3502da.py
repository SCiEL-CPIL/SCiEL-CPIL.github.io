import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 12
DEADLINE = time.time() + 130.0  # Safety margin

def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def compute_min_side(positions, angle_inner, angle_outer):
    """
    positions: list of (x, y) centers for inner hexagons.
    angle_inner: rotation of inner hexagons.
    angle_outer: rotation of outer hexagon.
    Returns min side length S of outer hexagon centered at (0,0).
    """
    # Vertices of all inner hexagons
    all_verts = []
    for x, y in positions:
        verts = hex_vertices(x, y, 1.0, angle_inner)
        all_verts.extend(verts)
    
    # Normals of outer hexagon
    a = math.radians(angle_outer)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    for px, py in all_verts:
        for nx, ny in normals:
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
                
    # Apothem = S * sqrt(3) / 2 => S = 2 * max_proj / sqrt(3)
    return (2.0 / math.sqrt(3.0)) * max_proj

def generate_topology_3x4():
    """3x4 Hex Grid"""
    positions = []
    # Lattice vectors u=(2,0), v=(1,sqrt(3))
    # 3 rows (j=0,1,2), 4 cols (i=0,1,2,3)
    for j in range(3):
        for i in range(4):
            x = 2*i + j
            y = j * math.sqrt(3)
            positions.append((x, y))
    # Center
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    return [(x-cx, y-cy) for x, y in positions]

def generate_topology_2x6():
    """2x6 Hex Grid"""
    positions = []
    for j in range(2):
        for i in range(6):
            x = 2*i + j
            y = j * math.sqrt(3)
            positions.append((x, y))
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    return [(x-cx, y-cy) for x, y in positions]

def generate_topology_flower5():
    """1 Center + 6 Ring 1 + 5 Ring 2 (consecutive)"""
    positions = [(0.0, 0.0)]
    # Ring 1
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * math.cos(theta), 2.0 * math.sin(theta)))
    # Ring 2
    # The 12 positions of Ring 2 are at distance 4 from center?
    # No, the centers are at distance:
    # Corner-ish: (4, 0) etc?
    # Let's use the lattice generation for Ring 2.
    # Ring 2 consists of hexes at (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) in axial?
    # Cartesian for Ring 2:
    # (4, 0), (2, 2*sqrt(3)), (-2, 2*sqrt(3)), (-4, 0), (-2, -2*sqrt(3)), (2, -2*sqrt(3))
    # AND
    # (3, sqrt(3)), (1, 3*sqrt(3)), (-1, 3*sqrt(3)), (-3, sqrt(3)), (-3, -sqrt(3)), (-1, -3*sqrt(3)), (1, -3*sqrt(3)), (3, -sqrt(3))
    # Wait, the "Edge" ones are at distance 2*sqrt(3)?
    # Distance from (0,0) to (3, sqrt(3)) is sqrt(9+3)=sqrt(12)=2*sqrt(3).
    # Distance from (0,0) to (4, 0) is 4.
    # So Ring 2 has two "shells" of distances? No, it's one ring.
    # The 12 hexes are at:
    # 6 "corners" at dist 4? No.
    # Let's just generate the 12 neighbors of the 6 Ring 1 hexes, excluding Ring 1 and Center.
    # Simpler: Generate all lattice points within dist 4.5, filter by dist > 3.5.
    pts = []
    for j in range(-2, 3):
        for i in range(-2, 3):
            x = 2*i + j
            y = j * math.sqrt(3)
            d = math.sqrt(x*x + y*y)
            if 3.5 < d < 4.5:
                pts.append((x, y))
    # pts should have 12 items.
    # Sort by angle
    pts.sort(key=lambda p: math.atan2(p[1], p[0]))
    # Take 5 consecutive
    for k in range(5):
        positions.append(pts[k])
        
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    return [(x-cx, y-cy) for x, y in positions]

def construct():
    t0 = time.time()
    best_side = 1e6
    best_result = None
    
    topologies = [
        generate_topology_3x4(),
        generate_topology_2x6(),
        generate_topology_flower5()
    ]
    
    # Angle ranges
    # Due to symmetry, we only need to search relative angle in [0, 60)
    # We fix inner angle to 0.
    # We search outer angle in [0, 60].
    
    for topo in topologies:
        if time.time() > DEADLINE - 10:
            break
            
        def objective(beta_deg):
            return compute_min_side(topo, 0.0, beta_deg)
        
        # Coarse grid search
        best_beta = 0.0
        best_val = 1e6
        for b in np.linspace(0, 60, 121): # 0.5 deg steps
            val = objective(b)
            if val < best_val:
                best_val = val
                best_beta = b
                
        # Refine
        try:
            res = minimize_scalar(objective, bounds=(max(0, best_beta-2), min(60, best_beta+2)), method='bounded', options={'xatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_beta = res.x
        except Exception:
            pass
            
        if best_val < best_side:
            best_side = best_val
            # Construct the result
            inner = [(x, y, 0.0) for x, y in topo]
            outer_center = (0.0, 0.0)
            outer_side = best_val
            outer_angle = best_beta
            best_result = (inner, outer_center, outer_side, outer_angle)
            
    if best_result is None:
        # Fallback
        topo = generate_topology_3x4()
        inner = [(x, y, 0.0) for x, y in topo]
        outer_center = (0.0, 0.0)
        outer_side = compute_min_side(topo, 0.0, 0.0)
        best_result = (inner, outer_center, outer_side, 0.0)
        
    return best_result
