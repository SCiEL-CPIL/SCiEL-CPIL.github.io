"""Optimize packing of 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = time.time() + 135.0  # safety margin before 150s kill

def hex_vertices(cx, cy, r, angle_deg):
    """Vertices of regular hexagon (circumradius r) centered at (cx,cy) rotated angle_deg."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    base = r * np.array([np.cos(np.radians(60*k)), np.sin(np.radians(60*k))] for k in range(6))
    return (R @ base.T).T + np.array([cx, cy])

def outer_side_from_vertices(verts):
    """Minimum enclosing regular hexagon side length for a set of vertices.
    Returns (side, center, angle_deg)."""
    # 6 outward normals (angle 0 => hexagon with flat top? use pointy-top base)
    # For a regular hexagon with circumradius S, vertex at angle 0 is (S,0).
    # Outward normals at angles 0,60,...,300 for the 6 sides.
    normals = np.array([[np.cos(np.radians(60*k)), np.sin(np.radians(60*k))] for k in range(6)])
    # For a given center c and rotation theta, support in direction n_k (rotated by theta):
    # side = max over v of max_k ( (v-c) . n_k(theta) )
    # We'll optimize over c and theta too. But for objective during inner optimization,
    # we can fix theta=0 and just find c, then the enclosing hexagon has that orientation.
    # Actually the outer hexagon can rotate; we should optimize its orientation too.
    # For simplicity in the inner objective, fix outer orientation = 0 and optimize center.
    # The final polish will also optimize outer center+angle.
    #
    # With outer orientation fixed at 0:
    # side(c) = max_k max_v (v - c) . n_k
    #         = max_k ( max_v (v . n_k) - c . n_k )
    # For each k, the best c . n_k is unbounded... no: c is shared across all k.
    # We want min_c max_k ( M_k - c.n_k ) where M_k = max_v (v.n_k).
    # This is a convex problem in c. We can solve it or just include c in the optimizer.
    # For the inner objective, I'll just compute with c = 0 for speed and let the
    # outer optimizer adjust. Actually that's wrong - the inner positions depend on c.
    #
    # Better: include outer center in the full optimization. But that's 2 + 3N = 38 vars.
    # For the inner topology optimization, I'll fix outer center at origin and outer angle=0,
    # and the objective is side = max_k max_v (v . n_k) (since c=0).
    # Then in the final polish, I optimize outer center and angle jointly.
    
    # For now, c=0, theta=0:
    vals = verts @ normals.T  # (nv, 6)
    M = vals.max(axis=0)  # (6,)
    side = M.max()
    return side

def objective(params, n_inner):
    """params: [x1,y1,a1, ..., xn,yn,an] (degrees). Returns enclosing hex side (c=0, theta=0)."""
    verts_all = []
    for i in range(n_inner):
        x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
        v = hex_vertices(x, y, 1.0, ang)
        verts_all.append(v)
    verts = np.vstack(verts_all)
    return outer_side_from_vertices(verts)

def objective_full(params):
    """params: [cx, cy, cang, x1,y1,a1, ..., xn,yn,an]. Full objective."""
    cx, cy, cang = params[0], params[1], params[2]
    n_inner = (len(params) - 3) // 3
    verts_all = []
    for i in range(n_inner):
        x, y, ang = params[3+3*i], params[3+3*i+1], params[3+3*i+2]
        v = hex_vertices(x, y, 1.0, ang)
        verts_all.append(v)
    verts = np.vstack(verts_all)
    # Outer hexagon center (cx,cy), rotation cang
    a = np.radians(cang)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    normals = R @ np.array([[np.cos(np.radians(60*k)), np.sin(np.radians(60*k))] for k in range(6)]).T
    # side = max over v of max_k ( (v - c) . n_k )
    rel = verts - np.array([cx, cy])
    vals = rel @ normals
    side = vals.max()
    return side

def make_topology_lattice():
    """1+6+5 hexagonal lattice pattern (with symmetry breaking)."""
    # Center hex at origin
    # 6 around it at distance 2 (touching), angles 0,60,...,300
    # 5 more in a ring
    pts = []
    pts.append((0.0, 0.0, 0.0))  # center
    for k in range(6):
        ang = 60.0 * k
        pts.append((2.0*np.cos(np.radians(ang)), 2.0*np.sin(np.radians(ang)), 0.0))
    # 5 more: extend one arm and add to fill
    # Put 5 in a second ring at distance 4, angles 0,60,120,180,240
    for k in range(5):
        ang = 60.0 * k
        pts.append((4.0*np.cos(np.radians(ang)), 4.0*np.sin(np.radians(ang)), 0.0))
    return pts

def make_topology_chain():
    """Linear chain with slight zigzag."""
    pts = []
    for k in range(N):
        x = 2.0 * k
        y = 0.0 if k % 2 == 0 else 0.1
        pts.append((x, y, 0.0))
    return pts

def make_topology_cluster():
    """Tight cluster: 2x3 grid-like in hexagonal coords."""
    # Hexagonal grid positions (axial coords to cartesian)
    # Using pointy-top hex grid
    def hex_to_cart(q, r):
        x = 1.5 * q
        y = np.sqrt(3)/2 * q + np.sqrt(3) * r
        return x, y
    positions = [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0),
                 (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)]
    pts = []
    for q, r in positions:
        x, y = hex_to_cart(q, r)
        pts.append((x, y, 0.0))
    return pts

def make_topology_symmetric_ring():
    """1 center + 6 ring + 5 outer ring (different from lattice)."""
    pts = []
    pts.append((0.0, 0.0, 0.0))
    for k in range(6):
        ang = 60.0 * k + 30.0  # offset
        pts.append((2.0*np.cos(np.radians(ang)), 2.0*np.sin(np.radians(ang)), 0.0))
    for k in range(5):
        ang = 60.0 * k + 6.0
        pts.append((4.0*np.cos(np.radians(ang)), 4.0*np.sin(np.radians(ang)), 0.0))
    return pts

def make_topology_packed_square():
    """3x4 rectangular arrangement in hex grid."""
    def hex_to_cart(q, r):
        x = 1.5 * q
        y = np.sqrt(3)/2 * q + np.sqrt(3) * r
        return x, y
    positions = [(q, r) for r in range(4) for q in range(3)]
    pts = []
    for q, r in positions:
        x, y = hex_to_cart(q, r)
        pts.append((x, y, 0.0))
    return pts

def make_topology_hexagon12():
    """12 hexagons forming a larger hexagonal shape (3-2-1-2-3 pattern won't give 12).
    Use 1+6+5 but with the 5 forming a partial ring."""
    pts = []
    pts.append((0.0, 0.0, 0.0))
    for k in range(6):
        ang = 60.0 * k
        pts.append((2.0*np.cos(np.radians(ang)), 2.0*np.sin(np.radians(ang)), 0.0))
    # 5 more at distance ~3.5, spread
    for k in range(5):
        ang = 60.0 * k + 30.0
        pts.append((3.8*np.cos(np.radians(ang)), 3.8*np.sin(np.radians(ang)), 0.0))
    return pts

def optimize_topology(topology, max_iter=3000, deadline=None):
    """Optimize a given topology. Returns (best_params, best_val)."""
    n_inner = len(topology)
    x0 = np.array([p[0] for p in topology] + [p[1] for p in topology] + [p[2] for p in topology])
    
    best_val = objective(x0, n_inner)
    best_params = x0.copy()
    
    # Nelder-Mead
    try:
        res = minimize(objective, x0, args=(n_inner,), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Check deadline
    if deadline is not None and time.time() > deadline:
        return best_params, best_val
    
    # Second pass with tighter settings
    try:
        res2 = minimize(objective, best_params, args=(n_inner,), method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass
    
    return best_params, best_val

def polish_full(inner_params, max_iter=2000, deadline=None):
    """Optimize outer center, angle, and all inner positions jointly."""
    n_inner = len(inner_params) // 3
    # x0: [cx, cy, cang, x1, y1, a1, ...]
    x0 = np.concatenate([[0.0, 0.0, 0.0], inner_params])
    
    best_val = objective_full(x0)
    best_params = x0.copy()
    
    try:
        res = minimize(objective_full, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline:
        return best_params, best_val
    
    # Tighter polish
    try:
        res2 = minimize(objective_full, best_params, method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass
    
    return best_params, best_val

def construct():
    t_start = time.time()
    deadline = time.time() + 135.0
    
    topologies = [
        make_topology_lattice,
        make_topology_chain,
        make_topology_cluster,
        make_topology_symmetric_ring,
        make_topology_packed_square,
        make_topology_hexagon12,
    ]
    
    best_overall_val = float('inf')
    best_overall_params = None
    
    # Optimize each topology
    for topo_func in topologies:
        if time.time() > deadline - 30.0:
            break
        topo = topo_func()
        params, val = optimize_topology(topo, max_iter=2000, deadline=deadline)
        if val < best_overall_val:
            best_overall_val = val
            best_overall_params = params
    
    if best_overall_params is None:
        # Fallback
        topo = make_topology_lattice()
        best_overall_params = np.array([p[0] for p in topo] + [p[1] for p in topo] + [p[2] for p in topo])
        best_overall_val = objective(best_overall_params, N)
    
    # Final polish with outer hexagon optimization
    if time.time() < deadline - 15.0:
        full_params, full_val = polish_full(best_overall_params, max_iter=2000, deadline=deadline)
        if full_val < best_overall_val:
            best_overall_val = full_val
            best_overall_params = full_params
    
    # Extract result
    if len(best_overall_params) > 3:
        cx, cy, cang = best_overall_params[0], best_overall_params[1], best_overall_params[2]
        inner_params = best_overall_params[3:]
    else:
        cx, cy, cang = 0.0, 0.0, 0.0
        inner_params = best_overall_params
    
    inner = []
    for i in range(N):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        ang = inner_params[3*i+2]
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (float(cx), float(cy)), float(best_overall_val), float(cang)
