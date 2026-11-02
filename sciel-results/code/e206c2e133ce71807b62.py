"""Kinematic closure: 12 unit hexagons as a fixed honeycomb cluster.
1D search over relative rotation between cluster and outer hexagon."""
import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def _hex_centers_shape(shape):
    """Generate 12 hexagon centers on a honeycomb lattice for a given shape.
    Returns list of (x,y)."""
    e1 = np.array([math.sqrt(3), 0.0])
    e2 = np.array([math.sqrt(3)/2.0, 1.5])
    centers = []
    if shape == "2x6":
        for i in range(6):
            for j in range(2):
                centers.append(i * e1 + j * e2)
    elif shape == "hex1":
        # Hexagon of radius 1: 7 hexes (center + 6 neighbors), plus 5 more around
        # Axial coords for hex radius 1: all (q,r) with max(|q|,|r|,|q+r|)<=1 -> 7 hexes
        # Add 5 more at distance 2 in some directions
        axial = [(0,0)]
        for q in range(-1, 2):
            for r in range(-1, 2):
                if max(abs(q), abs(r), abs(q+r)) <= 1 and (q,r) != (0,0):
                    axial.append((q,r))
        # that's 7. Add 5 more: pick 5 of the 12 at distance 2
        # distance-2 neighbors: (2,0),(0,2),(-2,2),(-2,0),(0,-2),(2,-2) and (1,1)? no
        # Let's just add (2,0),(1,1),(-1,2),(-2,0),(-1,-1) -> 5 more, total 12
        extra = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1)]
        axial.extend(extra)
        for (q,r) in axial:
            # axial to cartesian: x = e1*q + e2*r? Use: x = sqrt(3)*(q + r/2), y = 1.5*r
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            centers.append(np.array([x, y]))
    elif shape == "3x4":
        for i in range(4):
            for j in range(3):
                centers.append(i * e1 + j * e2)
    else:
        for i in range(6):
            for j in range(2):
                centers.append(i * e1 + j * e2)
    centers = np.array(centers)
    centers = centers - centers.mean(axis=0)
    return centers

def _S_for_outer_rot(beta_deg, centers):
    """Compute min outer hexagon side S for cluster (fixed at 0 deg) and outer rotated by beta_deg."""
    all_corners = []
    for c in centers:
        for m in range(6):
            th = math.radians(60 * m)
            corner = c + np.array([math.cos(th), math.sin(th)])
            all_corners.append(corner)
    all_corners = np.array(all_corners)
    phi = math.radians(beta_deg)
    u = []
    for k in range(6):
        th = phi + math.radians(60 * k)
        u.append(np.array([math.cos(th), math.sin(th)]))
    u = np.array(u)
    proj = all_corners @ u.T
    max_proj = proj.max()
    S = (2.0 / math.sqrt(3.0)) * max_proj
    return S

def _best_for_shape(shape):
    centers = _hex_centers_shape(shape)
    lo, hi = 0.0, 60.0
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = _S_for_outer_rot(m1, centers)
        f2 = _S_for_outer_rot(m2, centers)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
    beta_opt = (lo + hi) / 2.0
    S_opt = _S_for_outer_rot(beta_opt, centers)
    return S_opt, beta_opt, centers

def construct():
    best_S = float('inf')
    best_beta = 0.0
    best_centers = None
    for shape in ["2x6", "3x4", "hex1"]:
        try:
            S, beta, centers = _best_for_shape(shape)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_centers = centers
        except Exception:
            continue
    if best_centers is None:
        best_centers = _hex_centers_shape("2x6")
        best_S = _S_for_outer_rot(0.0, best_centers)
        best_beta = 0.0
    inner = []
    for c in best_centers:
        inner.append([float(c[0]), float(c[1]), 0.0])
    return inner, (0.0, 0.0), float(best_S), float(best_beta)
# EVOLVE-BLOCK-END
