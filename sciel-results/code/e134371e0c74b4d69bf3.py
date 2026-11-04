"""Optimized packing of 11 unit hexagons in a regular hexagon."""
import numpy as np
import time

N = 11

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def enclosing_hex_side_and_center(centers, angles, rot_deg):
    """Find minimal side and center of enclosing hexagon (rotated by rot_deg)."""
    # Collect all vertices of all inner hexagons
    all_pts = []
    for (cx, cy), ang in zip(centers, angles):
        verts = hex_vertices(cx, cy, ang)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    r = np.radians(rot_deg)
    max_range = 0
    for k in range(6):
        theta = r + k * np.pi / 3
        n = np.array([np.cos(theta), np.sin(theta)])
        projs = all_pts @ n
        rng = np.max(projs) - np.min(projs)
        max_range = max(max_range, rng)
    
    s = max_range / np.sqrt(3.0)
    
    # Find a valid center: must satisfy for each k:
    # max_p(p·n_k) - r_in <= c·n_k <= min_p(p·n_k) + r_in
    # where r_in = s * sqrt(3) / 2 = max_range / 2
    r_in = max_range / 2.0
    
    # Use the mean of the midpoints as center
    c = np.zeros(2)
    for k in range(6):
        theta = r + k * np.pi / 3
        n = np.array([np.cos(theta), np.sin(theta)])
        projs = all_pts @ n
        lo_k = np.max(projs) - r_in
        hi_k = np.min(projs) + r_in
        mid = (lo_k + hi_k) / 2.0
        c += mid * n
    c /= 6.0
    
    return s, c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    d = np.sqrt(3)
    
    centers = []
    centers.append((0.0, 0.0))
    for i in range(6):
        theta = i * np.pi / 3
        centers.append((d * np.cos(theta), d * np.sin(theta)))
    d2 = 2 * d
    for i in range(4):
        theta = i * np.pi / 3
        centers.append((d2 * np.cos(theta), d2 * np.sin(theta)))
    
    angles = [0.0] * 11
    
    best_s = float('inf')
    best_rot = 0.0
    
    gr = (np.sqrt(5) - 1) / 2
    lo = 0.0
    hi = 60.0
    
    x1 = lo + (1 - gr) * (hi - lo)
    x2 = lo + gr * (hi - lo)
    
    f1, _ = enclosing_hex_side_and_center(centers, angles, x1)
    f2, _ = enclosing_hex_side_and_center(centers, angles, x2)
    
    max_iters = 300
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if (hi - lo) < 1e-8:
            break
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = lo + (1 - gr) * (hi - lo)
            f1, _ = enclosing_hex_side_and_center(centers, angles, x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2, _ = enclosing_hex_side_and_center(centers, angles, x2)
    
    if f1 < f2:
        best_rot = x1
        best_s = f1
    else:
        best_rot = x2
        best_s = f2
    
    _, enc_center = enclosing_hex_side_and_center(centers, angles, best_rot)
    
    inner = [[cx, cy, ang] for (cx, cy), ang in zip(centers, angles)]
    return inner, (enc_center[0], enc_center[1]), best_s, best_rot

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
