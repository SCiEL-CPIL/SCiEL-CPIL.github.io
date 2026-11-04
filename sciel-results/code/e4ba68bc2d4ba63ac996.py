"""Packing 11 unit hexagons inside a minimal regular hexagon.

Strategy:
- Place 11 unit hexagons in a hexagonal cluster (one center + 6 in first ring + 4 in second ring
  arranged to be compact, or use the natural hex lattice of 11 hexes).
- The cluster is rigid (fixed relative positions and orientations).
- We optimize:
  1. The global rotation angle of the cluster.
  2. The outer hexagon's rotation angle.
  3. The outer hexagon's center offset (x, y) — though by symmetry we can fix it to origin
     and just rotate, but allowing translation can help.
  4. The outer side length is determined by the support function: for each of the 6 facets of the
     outer hexagon, the maximum projection of any inner hexagon vertex onto that facet's normal
     determines the required distance from center to that facet, which relates to side length.

For a regular hexagon of side s, the distance from center to each side (apothem) is s * sqrt(3)/2.
The 6 facet normals point at angles 0, 60, 120, 180, 240, 300 degrees (if the hexagon is oriented
with a flat side at the top, the normals are at 30, 90, 150, 210, 270, 330 — depends on convention).

Let's define: outer hexagon centered at (cx, cy), side length S, rotation theta_out.
The six facet normals are at angles theta_out + k*60 for k=0..5 (if we define the normal of
facet 0 at angle theta_out). Actually, let's be precise:

A regular hexagon with side S, centered at origin, with one vertex pointing up (pointy-top):
Vertices at angles 90, 150, 210, 270, 330, 30 degrees, radius S.
The facets (edges) are between consecutive vertices. The outward normals of the facets are at
angles 60, 120, 180, 240, 300, 0 degrees... 

Let me use a cleaner formulation. A regular hexagon can be defined by its 6 support lines.
For a hexagon with apothem a (distance from center to each side), the side length is S = 2a/sqrt(3).
The 6 sides have outward normals at angles alpha, alpha+60, ..., alpha+300.

For the inner hexagons to be inside, for each facet normal direction n_k, the maximum projection
of all inner hexagon vertices onto n_k must be <= a (the apothem), and the minimum projection
must be >= -a. By symmetry of the outer hexagon, if we center the cluster at the outer center,
we need max_k |max_proj_k - min_proj_k|... no, we need for each k: max_proj_k <= a AND min_proj_k >= -a.
Since we can translate the cluster, we can center it so that for each k, (max_proj_k + min_proj_k)/2 is the center offset in that direction. But we have 2D translation (2 DOF) and 6 constraints. 

Simpler: fix the outer hexagon center at origin. Allow the inner cluster to be translated by (tx, ty) and rotated by theta_in. The outer hexagon is rotated by theta_out. For each of the 6 facet normals n_k of the outer hexagon, compute the range of projections of all inner vertices onto n_k. The required apothem a must satisfy a >= max_k (max_proj_k - min_proj_k)/2 ... no. 

Actually: for the inner shape to fit in the outer hexagon centered at origin with apothem a and facet normals n_k:
For each k: max_{v in inner} dot(v, n_k) <= a AND min_{v in inner} dot(v, n_k) >= -a.
This is equivalent to: for each k, (max_proj_k + a) >= 0 and (a - min_proj_k) >= 0, i.e., a >= max_proj_k and a >= -min_proj_k, so a >= max(max_proj_k, -min_proj_k) for each k.
Overall: a >= max_k max(max_proj_k, -min_proj_k).

Then S = 2a/sqrt(3).

We want to minimize S, which means minimizing a, which means minimizing max_k max(max_proj_k, -min_proj_k).

The inner cluster has fixed relative geometry. We parameterize by:
- theta_in: rotation of the cluster
- tx, ty: translation of the cluster
- theta_out: rotation of the outer hexagon

That's 4 parameters. We can do a grid search over these.

Let me define the 11 hexagons. Unit hexagon (side 1) has vertices at distance 1 from center (circumradius = side for regular hexagon). The 6 vertices of a unit hexagon centered at origin, rotated by phi:
At angles phi + k*60 for k=0..5, radius 1.

For the cluster, I'll use the standard hex packing:
- Center hex at (0,0)
- 6 hexagons in the first ring at distance 2*sqrt(3)/... 

For two unit hexagons to touch, the distance between centers is 2 * apothem = 2 * sqrt(3)/2 * ... 
Unit hexagon apothem = sqrt(3)/2. Two touching hexagons have center distance = 2 * apothem = sqrt(3)? No.
If two regular hexagons of side s are placed with parallel sides and touching, the distance between centers is 2 * apothem = 2 * (s*sqrt(3)/2) = s*sqrt(3). For s=1, that's sqrt(3).

Wait, let me reconsider. A regular hexagon with side s has:
- Circumradius (center to vertex) = s
- Apothem (center to side midpoint) = s * sqrt(3)/2

Two hexagons touching along a full edge: the distance between centers is 2 * apothem = 2 * s * sqrt(3)/2 = s * sqrt(3). For s=1, distance = sqrt(3).

But in a hexagonal tiling, the hexagons are arranged so that each hexagon touches 6 neighbors. The neighbor centers are at distance 2 * apothem = sqrt(3) (for unit hexagons) in 6 directions separated by 60 degrees.

So the 11 hexagons in a hex cluster:
- 1 at center (0,0)
- 6 at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300 degrees
- 4 more? For 11 total, we need 4 more in the second ring. The second ring has 12 positions, but we only need 4. To make it compact, we can pick 4 adjacent ones.

Actually, let's just use the natural hex lattice positions for 11 hexagons. The hex lattice positions for a hexagonal number arrangement:
Row 0: 1 hex
Row 1: 3 hexes
Row 2: 5 hexes
Row 3: 2 hexes? No, that's 11 but not symmetric.

Let me just place them:
Center: (0,0)
Ring 1 (6 hexes): at angles 0,60,...,300, distance sqrt(3)
Ring 2: we need 4 more. Place them at angles 30, 90, 150, 210? No, let's pick 4 consecutive from the 12 available second-ring positions.

Actually, for simplicity and symmetry, let me just use 11 hexagons in a roughly circular arrangement. Or better: use the fact that the problem allows each hex to rotate independently, but for a rigid cluster approach, I'll fix all inner hexagons to have the same orientation as the cluster.

Let me just implement the rigid cluster with 11 hexagons in a hex lattice, optimize the 4 parameters via grid search, and return the best.

I'll use numpy for speed.
"""

import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _make_cluster_positions():
    """Return list of (x, y) for 11 hexagons in a hex lattice cluster."""
    d = np.sqrt(3)  # distance between touching hex centers
    positions = [(0.0, 0.0)]
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = k * np.pi / 3
        positions.append((d * np.cos(ang), d * np.sin(ang)))
    # Ring 2: 4 more hexagons, pick 4 consecutive from the 12 available
    # Second ring positions are at distance 2*d in 12 directions (every 30 deg)
    # Let's pick 4 that make a compact shape
    for k in [1, 2, 3, 4]:  # angles 30, 60, 90, 120 degrees
        ang = k * np.pi / 6
        positions.append((2 * d * np.cos(ang), 2 * d * np.sin(ang)))
    return positions

def _compute_min_side(positions, theta_in_deg, tx, ty, theta_out_deg):
    """Compute the minimum outer hexagon side length for given parameters."""
    theta_in = np.radians(theta_in_deg)
    theta_out = np.radians(theta_out_deg)
    
    # All vertices of all inner hexagons
    all_verts = []
    for (px, py) in positions:
        # Rotate position by theta_in, then translate
        rx = px * np.cos(theta_in) - py * np.sin(theta_in) + tx
        ry = px * np.sin(theta_in) + py * np.cos(theta_in) + ty
        # Hex vertices at this center, rotated by theta_in
        angles = theta_in + np.arange(6) * np.pi / 3
        vx = rx + np.cos(angles)
        vy = ry + np.sin(angles)
        all_verts.append(np.column_stack([vx, vy]))
    
    all_verts = np.vstack(all_verts)
    
    # Outer hexagon facet normals
    normals = np.column_stack([
        np.cos(theta_out + np.arange(6) * np.pi / 3),
        np.sin(theta_out + np.arange(6) * np.pi / 3)
    ])
    
    # Projections
    projs = all_verts @ normals.T  # shape (n_verts, 6)
    
    max_projs = np.max(projs, axis=0)
    min_projs = np.min(projs, axis=0)
    
    # Required apothem
    a = np.max(np.maximum(max_projs, -min_projs))
    
    # Side length
    S = 2 * a / np.sqrt(3)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    positions = _make_cluster_positions()
    
    best_S = float('inf')
    best_params = (0.0, 0.0, 0.0, 0.0)  # theta_in, tx, ty, theta_out
    
    # Grid search
    # theta_in: 0 to 30 degrees (symmetry of hexagon means 60 deg period, but cluster breaks symmetry)
    # Let's search 0 to 60 degrees for theta_in
    # theta_out: 0 to 60 degrees
    # tx, ty: small range, say -2 to 2
    
    # Coarse grid
    theta_in_vals = np.arange(0, 60, 5.0)
    theta_out_vals = np.arange(0, 60, 5.0)
    tx_vals = np.arange(-2, 2.5, 1.0)
    ty_vals = np.arange(-2, 2.5, 1.0)
    
    for ti in theta_in_vals:
        for to in theta_out_vals:
            for tx in tx_vals:
                for ty in ty_vals:
                    S = _compute_min_side(positions, ti, tx, ty, to)
                    if S < best_S:
                        best_S = S
                        best_params = (ti, tx, ty, to)
        
        if time.time() > deadline:
            break
    
    # Refine around best
    if time.time() < deadline:
        ti0, tx0, ty0, to0 = best_params
        for _ in range(3):
            step = 0.5
            ti_vals = np.arange(ti0 - 5, ti0 + 5.1, step)
            to_vals = np.arange(to0 - 5, to0 + 5.1, step)
            tx_vals = np.arange(tx0 - 1, tx0 + 1.1, step)
            ty_vals = np.arange(ty0 - 1, ty0 + 1.1, step)
            
            for ti in ti_vals:
                for to in to_vals:
                    for tx in tx_vals:
                        for ty in ty_vals:
                            S = _compute_min_side(positions, ti, tx, ty, to)
                            if S < best_S:
                                best_S = S
                                best_params = (ti, tx, ty, to)
            
            if time.time() > deadline:
                break
            step *= 0.5
    
    ti, tx, ty, to = best_params
    
    # Build inner hexagons
    inner = []
    theta_in = np.radians(ti)
    for (px, py) in positions:
        rx = px * np.cos(theta_in) - py * np.sin(theta_in) + tx
        ry = px * np.sin(theta_in) + py * np.cos(theta_in) + ty
        inner.append([rx, ry, ti])
    
    return inner, (0.0, 0.0), best_S, to

