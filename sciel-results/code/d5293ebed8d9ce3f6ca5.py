"""Packing 11 unit regular hexagons inside a regular hexagon (minimize outer side)."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 11
DEADLINE = 115.0  # safety margin before 120s kill
START = time.time()

# Precompute hexagon geometry
# A unit regular hexagon has vertices at distance 1 from center.
# Vertex i at angle 60*i + offset.
# For a hexagon with "flat-top" orientation (angle=0), vertices are at:
#   (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
# That's the pointy-top... let's be precise.
# Standard: vertex at angle theta_i = 60*i degrees from x-axis.
# v_i = (cos(60i), sin(60i)) for i=0..5 gives flat-top hexagon (vertices on x-axis).
# Actually cos(0)=1, sin(0)=0 -> (1,0). cos(60)=0.5, sin(60)=0.866 -> (0.5, 0.866).
# This is a hexagon with a vertex pointing right.

HEX_VERTS = np.array([[math.cos(math.pi/3 * i), math.sin(math.pi/3 * i)] for i in range(6)])

def hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return R @ HEX_VERTS.T + np.array([cx, cy])

def outer_hex_support(cx, cy, side, angle_deg):
    """
    Compute the support function of the outer hexagon.
    The outer hexagon has center (cx,cy), side length 'side', rotation angle_deg.
    A regular hexagon with side s has inradius = s*sqrt(3)/2, circumradius = s.
    Vertices are at distance s from center.
    The support function h(theta) = max over vertices of dot(vertex - center, dir).
    For a regular hexagon with vertices at angles 60*i + angle_deg, distance s:
    h(theta) = s * max_i cos(theta - (60i + angle_deg))
    But we need the minimal enclosing hexagon, so we need to find the smallest side
    such that all inner hexagon vertices are inside.
    
    Actually, the problem says the outer is a regular hexagon. We need to find the
    minimal side length S and rotation/position such that all 11 inner hexagons
    fit inside.
    
    Approach: For a given configuration of inner hexagons, find the minimal enclosing
    regular hexagon. The outer hexagon has 6 directions. For a regular hexagon with
    normal directions at angles phi + 60*i (i=0..5), the constraint is:
      For each inner vertex v_j: dot(v_j - outer_center, n_i) <= inradius = S*sqrt(3)/2
    where n_i = (cos(phi+60i), sin(phi+60i)).
    
    So S >= max_j max_i dot(v_j - C, n_i) / (sqrt(3)/2)
    
    We can optimize over C and phi to minimize S.
    """
    pass

def compute_outer_side(inner_hexes):
    """
    Given inner hexagons (list of (cx, cy, angle)), compute minimal enclosing regular hexagon side.
    We optimize over outer center (Cx, Cy) and outer rotation phi.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in inner_hexes:
        verts = hex_vertices(cx, cy, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # shape (66, 2)
    
    def objective(params):
        Cx, Cy, phi = params
        C = np.array([Cx, Cy])
        # Normal directions of outer hexagon
        normals = np.array([[math.cos(math.radians(phi + 60*i)), math.sin(math.radians(phi + 60*i))] for i in range(6)])
        # For each vertex, compute max dot with normals
        # dot(v - C, n_i) for all v, all i
        rel = all_verts - C  # (66, 2)
        dots = rel @ normals.T  # (66, 6)
        max_proj = np.max(dots, axis=0)  # (6,)
        S = np.max(max_proj) / (math.sqrt(3) / 2)
        return max(0.0, S)
    
    # Initial guess: center of mass of all vertices
    cx0 = np.mean(all_verts[:, 0])
    cy0 = np.mean(all_verts[:, 1])
    
    best_S = float('inf')
    best_params = None
    
    # Try a few random inits for phi and center
    for trial in range(5):
        if time.time() - START > DEADLINE:
            break
        phi0 = np.random.uniform(0, 60) if trial > 0 else 0.0
        Cx0 = cx0 + (np.random.uniform(-1, 1) if trial > 0 else 0)
        Cy0 = cy0 + (np.random.uniform(-1, 1) if trial > 0 else 0)
        res = minimize(objective, [Cx0, Cy0, phi0], method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        S = res.fun
        if S < best_S:
            best_S = S
            best_params = res.x
    
    return best_S

def construct():
    global START, DEADLINE
    START = time.time()
    DEADLINE = START + 115.0
    
    # Initial configuration: hexagonal packing
    # 11 hexagons: center + 6 ring + 4 more? 
    # Actually for 11: 1 center, 6 in first ring, 4 in second ring partially?
    # Standard hex packing: 1, 7, 19, 37...
    # 11 = 1 + 6 + 4. Let's place them in a hexagonal pattern.
    
    # Distance between centers of adjacent unit hexagons (edge-to-edge) = sqrt(3)
    d = math.sqrt(3)
    
    # Hexagonal lattice positions for 11 hexagons
    # Use axial coordinates
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance d
    # Ring 2: 12 hexagons at distance 2d, but we only need 4 more
    
    positions = [(0.0, 0.0)]
    # Ring 1
    for i in range(6):
        ang = math.radians(60 * i)
        positions.append((d * math.cos(ang), d * math.sin(ang)))
    # Ring 2: pick 4 from the 12
    for i in range(12):
        if len(positions) >= 11:
            break
        ang = math.radians(60 * (i // 2) + (30 if i % 2 == 0 else 90))
        # Actually ring 2 positions are at distance 2d, angles offset
        # Let me just use a simple pattern
        pass
    
    # Simpler: use a known good initial layout
    # 11 hexagons in a roughly circular arrangement
    positions = [(0.0, 0.0)]
    for i in range(6):
        ang = math.radians(60 * i)
        positions.append((d * math.cos(ang), d * math.sin(ang)))
    # Add 4 more at distance ~2d, at angles 0, 120, 240, and one more
    for i in range(4):
        ang = math.radians(60 * i + 30)  # offset to avoid overlap
        positions.append((2.2 * d * math.cos(ang), 2.2 * d * math.sin(ang)))
    
    # Ensure we have exactly 11
    positions = positions[:11]
    
    # Initial angles: all 0
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Optimize
    def fitness(params):
        if time.time() > DEADLINE:
            return 1e6
        inner_hexes = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = params[3*i+2]
            inner_hexes.append((cx, cy, ang))
        return compute_outer_side(inner_hexes)
    
    # Bounds
    bounds = [(-10, 10)] * (3 * N)
    
    # Use differential evolution for global search
    best_S = float('inf')
    best_params = None
    
    try:
        if time.time() < DEADLINE - 60:
            de_result = differential_evolution(fitness, bounds, maxiter=50, tol=1e-4,
                                                seed=42, polish=False, workers=-1,
                                                init='latinhypercube')
            if de_result.fun < best_S:
                best_S = de_result.fun
                best_params = de_result.x
    except Exception:
        pass
    
    # Polish with Nelder-Mead
    if best_params is not None and time.time() < DEADLINE - 10:
        try:
            res = minimize(fitness, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract final configuration
    if best_params is not None:
        final_inner = []
        for i in range(N):
            final_inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    else:
        final_inner = inner
    
    # Compute final outer side
    final_S = compute_outer_side([(x, y, a) for x, y, a in final_inner])
    
    # Find optimal outer center and rotation
    all_verts = np.vstack([hex_vertices(x, y, a) for x, y, a in final_inner])
    cx0 = np.mean(all_verts[:, 0])
    cy0 = np.mean(all_verts[:, 1])
    
    def outer_obj(params):
        Cx, Cy, phi = params
        C = np.array([Cx, Cy])
        normals = np.array([[math.cos(math.radians(phi + 60*i)), math.sin(math.radians(phi + 60*i))] for i in range(6)])
        rel = all_verts - C
        dots = rel @ normals.T
        max_proj = np.max(dots, axis=0)
        return np.max(max_proj) / (math.sqrt(3) / 2)
    
    best_outer_S = final_S
    best_outer_params = [cx0, cy0, 0.0]
    
    if time.time() < DEADLINE - 5:
        try:
            res = minimize(outer_obj, [cx0, cy0, 0.0], method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_outer_S:
                best_outer_S = res.fun
                best_outer_params = res.x
        except Exception:
            pass
    
    outer_cx, outer_cy, outer_phi = best_outer_params
    
    return final_inner, (outer_cx, outer_cy), best_outer_S, outer_phi
