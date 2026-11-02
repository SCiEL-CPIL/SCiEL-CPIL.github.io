"""
Hexagon packing: place N unit regular hexagons inside a regular hexagon,
minimizing the outer side length.

Strategy:
- Use a hexagonal lattice arrangement for the inner hexagons (all aligned, no rotation).
- For a given arrangement, compute the minimal enclosing regular hexagon.
- The key insight: for a set of points, the minimal enclosing regular hexagon of a given
  orientation can be found by projecting onto 3 axes (the 3 pairs of parallel sides).
- We optimize the arrangement by trying different lattice parameters and orientations.

For N hexagons in a hexagonal lattice:
- A hexagonal number arrangement: 1, 7, 19, 37, 61, 91, 127, ...
- For N=11, we can use a 3x3 grid with 2 removed, or a linear arrangement, etc.

Actually, let me think more carefully. The optimal packing of small regular hexagons
inside a large regular hexagon is a known problem. For small N, the optimal arrangements
are often based on hexagonal lattices.

For the minimal enclosing hexagon of a set of points:
A regular hexagon with center (cx, cy), side length s, and rotation angle theta has
6 sides. Each side is a line at distance s*sqrt(3)/2 from the center.
The 6 normal directions are at angles theta + k*60 degrees for k=0..5.

For all points to be inside, for each of the 6 directions n_k:
  max over all points p of dot(p - c, n_k) <= s * sqrt(3) / 2

So the minimal s for a given center c and orientation theta is:
  s = max over k=0..5 of [ max over p of dot(p - c, n_k) ] / (sqrt(3)/2)

We can optimize over c and theta.

For a hexagonal lattice of N hexagons, the vertices of all hexagons form a specific set.
Let me compute the vertices for a hexagonal lattice arrangement.

A regular hexagon with side length 1 and center at (x, y) and rotation angle a has
vertices at (x + cos(a + k*60°), y + sin(a + k*60°)) for k=0..5.

For a hexagonal lattice with all hexagons aligned (same rotation), the centers are at
positions on a triangular lattice with spacing 2*sqrt(3) (distance between centers of
adjacent hexagons in a honeycomb is 2*sqrt(3) for side-1 hexagons... wait, let me recalculate.

For regular hexagons of side length 1:
- The distance from center to vertex (circumradius) is 1.
- The distance from center to edge midpoint (apothem) is sqrt(3)/2.
- Two adjacent hexagons in a honeycomb: the distance between centers is 2 * apothem = sqrt(3).

Wait, no. In a honeycomb tiling, adjacent hexagons share an edge. The distance between
centers of two hexagons sharing an edge is 2 * (apothem) = 2 * sqrt(3)/2 = sqrt(3).

So for a hexagonal lattice of hexagon centers, the lattice vectors have length sqrt(3)
and are at 60 degrees to each other.

Let me define the lattice:
- v1 = (sqrt(3), 0)
- v2 = (sqrt(3)/2, 3/2)

These have length sqrt(3) and angle 60 degrees between them.

For N=11, a natural arrangement is a hexagonal cluster. The hexagonal numbers are:
1, 7, 19, 37, ...
So 11 doesn't fit exactly. We could use a 3x3 hexagonal arrangement (which has 19 points)
and remove 8, or use a different arrangement.

Actually, for 11 hexagons, a good arrangement might be:
- A central hexagon
- 6 around it (total 7)
- 4 more in the next ring (total 11)

Or we could try a 2x2x2 rectangular-ish arrangement.

Let me just try several arrangements and find the best one.

Actually, let me think about this differently. The problem is to minimize the outer hexagon
side length. Let me try a few specific arrangements:

1. Linear: all in a row. Side length would be large.
2. Hexagonal cluster: best for small N.
3. 2-row arrangement.

For the hexagonal cluster of 7 (1 center + 6 around), the minimal enclosing hexagon
has side length 3 (since the outer hexagons extend 1 unit from their centers, and the
centers are at distance sqrt(3) from the center, so the vertices are at distance sqrt(3)+1
from the center... but we need to project onto the hexagon normals).

Let me just code up the optimization for a given set of centers and find the best
enclosing hexagon. Then try different center arrangements.

For efficiency, I'll:
1. Try several candidate arrangements of N hexagon centers.
2. For each, compute the minimal enclosing hexagon by optimizing over center and rotation.
3. Return the best result.

The optimization over center and rotation for a fixed set of points:
- For a given rotation theta, the minimal center can be found analytically.
- The constraint is that for each of the 6 normals n_k(theta), the projection of all points
  onto n_k must fit within [-s*sqrt(3)/2, s*sqrt(3)/2] after centering.
- For a given theta, the optimal center c minimizes the max over k of (max_p dot(p, n_k) - min_p dot(p, n_k)) / (sqrt(3)).
- Actually, for a given theta, the width in direction n_k is max_p dot(p, n_k) - min_p dot(p, n_k).
- The center should be at the midpoint: c_k = (max_p dot(p, n_k) + min_p dot(p, n_k)) / 2 for each k.
- But we need a single center that works for all 6 directions. The optimal center minimizes
  the maximum over k of the half-width, which is a convex problem in c.

Actually, for a fixed theta, the problem of finding the minimal s and center c is:
  minimize s
  subject to: for all k, for all p: dot(p - c, n_k) <= s * sqrt(3)/2
             for all k, for all p: dot(c - p, n_k) <= s * sqrt(3)/2

This is a linear program in (c, s). We can solve it with scipy linprog.

But actually, there's a simpler approach. For a fixed theta, the minimal enclosing hexagon
can be found by noting that the hexagon is the intersection of 6 half-planes. The minimal
s is determined by the maximum "width" in the 3 directions (since opposite sides give the
same constraint).

Let me just sweep theta from 0 to 60 degrees (by symmetry, only 0-60 matters) and for
each theta, compute the minimal s.

For a given theta and set of points P:
  For each of the 3 axis directions (0, 60, 120 degrees offset by theta):
    Compute the range of projections: r_k = max_p dot(p, n_k) - min_p dot(p, n_k)
  The minimal side length is max(r_0, r_1, r_2) / sqrt(3)
  
  Wait, that's not quite right. Let me think again.

A regular hexagon with "radius" R (distance from center to vertex) has side length R.
The distance from center to each side (apothem) is R * sqrt(3)/2.

So if the half-width in direction n_k is w_k = (max_p dot(p - c, n_k) - min_p dot(p - c, n_k)) / 2,
then we need w_k <= R * sqrt(3)/2 for all k.

So R >= 2 * w_k / sqrt(3) for all k, meaning R >= max_k(2 * w_k / sqrt(3)).

But w_k depends on c. For a fixed theta, we want to choose c to minimize max_k(2*w_k/sqrt(3)).

This is equivalent to minimizing the maximum over k of (max_p dot(p, n_k) - min_p dot(p, n_k)),
which is independent of c! Because max_p dot(p - c, n_k) - min_p dot(p - c, n_k) = 
max_p dot(p, n_k) - min_p dot(p, n_k).

So for a fixed theta, the minimal R is:
  R = max over k=0,1,2 of [max_p dot(p, n_k(theta)) - min_p dot(p, n_k(theta))] / sqrt(3)

where n_k(theta) are the 3 unit vectors at angles theta, theta+60, theta+120.

And the center c can be chosen as the centroid or any point that centers the projections.

Great, so the algorithm is:
1. Generate candidate sets of hexagon centers.
2. For each set, compute all vertices of all hexagons.
3. Sweep theta from 0 to 60 degrees.
4. For each theta, compute R = max over 3 axes of (max_proj - min_proj) / sqrt(3).
5. The best theta gives the minimal R.
6. The center c is the average of the midpoints for each axis.

Let me implement this.
"""

import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """Return the 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _min_enclosing_hex_side(points, theta_sweep=None):
    """
    Find the minimal side length of a regular hexagon enclosing all points.
    Returns (side_length, center_x, center_y, theta_deg).
    """
    if theta_sweep is None:
        theta_sweep = np.linspace(0, np.pi / 3, 121)  # 0 to 60 degrees, 121 points
    
    best_side = np.inf
    best_theta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    for theta in theta_sweep:
        # 3 axis directions
        angles = theta + np.pi / 3 * np.arange(3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project all points onto each normal
        projections = points @ normals.T  # shape (n_points, 3)
        
        # For each axis, compute range
        ranges = projections.max(axis=0) - projections.min(axis=0)
        
        # Side length = max range / sqrt(3)
        side = np.max(ranges) / np.sqrt(3)
        
        if side < best_side:
            best_side = side
            best_theta = theta
            # Center: midpoint of projections for each axis, then average
            # For each axis k, the center projection is (max + min) / 2
            midpoints = (projections.max(axis=0) + projections.min(axis=0)) / 2.0
            # Solve for center: normals @ [cx, cy]^T = midpoints
            # This is an overdetermined system; use least squares
            try:
                c, _, _, _ = np.linalg.lstsq(normals.T, midpoints, rcond=None)
                best_cx, best_cy = c[0], c[1]
            except:
                best_cx, best_cy = points[:, 0].mean(), points[:, 1].mean()
    
    return best_side, best_cx, best_cy, np.degrees(best_theta)

def _make_hex_cluster(n):
    """
    Create a hexagonal cluster of n hexagon centers.
    Uses a hexagonal lattice with spacing sqrt(3).
    """
    if n == 1:
        return np.array([[0.0, 0.0]])
    
    # Generate hexagonal lattice points in order of distance from origin
    v1 = np.array([np.sqrt(3), 0.0])
    v2 = np.array([np.sqrt(3) / 2, 1.5])
    
    points = []
    # Search in a large enough region
    max_coord = int(np.ceil(np.sqrt(n) * 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            p = i * v1 + j * v2
            points.append((p[0], p[1], i**2 + i*j + j**2))  # hex distance squared
    
    # Sort by hex distance
    points.sort(key=lambda x: x[2])
    
    # Take first n
    centers = np.array([[p[0], p[1]] for p in points[:n]])
    
    # Center the cluster
    centers -= centers.mean(axis=0)
    
    return centers

def _make_linear(n):
    """Linear arrangement."""
    spacing = np.sqrt(3)
    xs = spacing * (np.arange(n) - (n - 1) / 2.0)
    return np.column_stack([xs, np.zeros(n)])

def _make_2row(n):
    """Two-row arrangement."""
    n1 = n // 2
    n2 = n - n1
    spacing = np.sqrt(3)
    x1 = spacing * (np.arange(n1) - (n1 - 1) / 2.0)
    x2 = spacing * (np.arange(n2) - (n2 - 1) / 2.0) + spacing / 2
    y1 = np.zeros(n1)
    y2 = np.full(n2, 1.5)  # vertical spacing for hex lattice
    return np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])

def _make_rectangular(n):
    """Rectangular grid arrangement."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    spacing = np.sqrt(3)
    xs = spacing * (np.arange(cols) - (cols - 1) / 2.0)
    ys = 1.5 * (np.arange(rows) - (rows - 1) / 2.0)
    grid = np.array([[x, y] for y in ys for x in xs])
    return grid[:n]

def _all_vertices(centers, angle_deg=0.0):
    """Get all vertices of all hexagons."""
    verts = []
    for cx, cy in centers:
        v = _hex_vertices(cx, cy, 1.0, angle_deg)
        verts.append(v)
    return np.vstack(verts)

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    best_side = np.inf
    best_result = None
    
    # Generate candidate center arrangements
    arrangements = []
    
    # Hexagonal cluster
    try:
        arrangements.append(('hex_cluster', _make_hex_cluster(N)))
    except:
        pass
    
    # Linear
    try:
        arrangements.append(('linear', _make_linear(N)))
    except:
        pass
    
    # Two-row
    try:
        arrangements.append(('2row', _make_2row(N)))
    except:
        pass
    
    # Rectangular
    try:
        arrangements.append(('rect', _make_rectangular(N)))
    except:
        pass
    
    # Try different rotations of the inner hexagons
    inner_rotations = [0.0, 30.0]
    
    for arr_name, centers in arrangements:
        if time.time() - start_time > deadline - 5:
            break
        
        for rot in inner_rotations:
            if time.time() - start_time > deadline - 5:
                break
            
            # Get all vertices
            verts = _all_vertices(centers, rot)
            
            # Find minimal enclosing hexagon
            side, cx, cy, theta = _min_enclosing_hex_side(verts)
            
            if side < best_side:
                best_side = side
                # Build inner list
                inner = []
                for p in centers:
                    inner.append([float(p[0]), float(p[1]), float(rot)])
                best_result = (inner, (float(cx), float(cy)), float(side), float(theta))
    
    # If no valid result, return a fallback
    if best_result is None:
        # Fallback: linear arrangement with large outer hexagon
        spacing = np.sqrt(3)
        inner = []
        for k in range(N):
            x = spacing * (k - (N - 1) / 2.0)
            inner.append([x, 0.0, 0.0])
        return inner, (0.0, 0.0), 50.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-START
# (construct function is defined above)
# EVOLVE-BLOCK-END
