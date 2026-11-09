"""Packing unit hexagons inside a regular hexagon using a hexagonal lattice cluster.

Approach family: hexagonal (triangular) lattice packing — the known densest
packing of congruent hexagons. Place N side-1 hexagons on a triangular lattice
with edge-to-edge spacing sqrt(3), in a compact blob ordered by hex distance
from the origin, then compute the minimal enclosing regular hexagon (axis
aligned) by projecting every inner vertex onto the six outer normals.
"""
import math

N = 11

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (circumradius = side) centered at (cx, cy)."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + math.radians(60 * k)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def construct():
    # Triangular lattice basis for edge-to-edge side-1 hexagons (spacing = sqrt(3)).
    s = math.sqrt(3.0)
    e1x, e1y = s, 0.0
    e2x, e2y = s / 2.0, 1.5  # s*sqrt(3)/2 == 3/2

    # Hex distance in axial coordinates.
    def hdist(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2

    # Compact blob: axial coords sorted by distance from origin.
    coords = []
    for q in range(-N, N + 1):
        for r in range(-N, N + 1):
            coords.append((q, r))
    coords.sort(key=lambda cr: (hdist(cr[0], cr[1]), cr[0], cr[1]))
    axial = coords[:N]

    # Cartesian centers.
    centers = []
    for q, r in axial:
        x = e1x * q + e2x * r
        y = e1y * q + e2y * r
        centers.append((x, y))

    # Outer hexagon (flat-top, angle 0) unit normals.
    normals = []
    for j in range(6):
        th = math.radians(60 * j)
        normals.append((math.cos(th), math.sin(th)))

    inradius_factor = math.sqrt(3.0) / 2.0  # inradius = S * factor

    # Minimal S such that every inner vertex is inside the outer hexagon.
    max_proj = 0.0
    for (px, py) in centers:
        for (vx, vy) in _hex_vertices(px, py, 0.0, 1.0):
            for (nx, ny) in normals:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj

    S = max_proj / inradius_factor

    inner = [[c[0], c[1], 0.0] for c in centers]
    return inner, (0.0, 0.0), S, 0.0
