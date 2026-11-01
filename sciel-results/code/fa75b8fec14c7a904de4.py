"""Rigid polyhedral-mesh optimiser for the 100-point logarithmic energy on S^2."""
import math
import time
import numpy as np
from scipy.spatial import Delaunay, cKDTree

N = 100

# EVOLVE-BLOCK-START

def _subdivide(face, pts):
    """Subdivide a triangle into 4; returns new vertex indices and 4 sub-triangles."""
    a, b, c = face
    new = len(pts)
    ab = (pts[a] + pts[b]) / 2.0
    bc = (pts[b] + pts[c]) / 2.0
    ca = (pts[c] + pts[a]) / 2.0
    for v in (ab, bc, ca):
        pts.append(v / np.linalg.norm(v))
    ab_i, bc_i, ca_i = new, new + 1, new + 2
    return ab_i, bc_i, ca_i, [(a, ab_i, ca_i), (ab_i, b, bc_i), (ca_i, bc_i, c), (ab_i, bc_i, ca_i)]


def _icosahedron():
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
           (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
           (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    pts = [np.array(p, dtype=float) / np.linalg.norm(p) for p in raw]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    return pts, faces


def _build_mesh():
    """Subdivide the icosahedron until we have exactly 100 vertices, returning faces."""
    pts, faces = _icosahedron()
    while len(pts) < N:
        new_faces = []
        for f in faces:
            ab_i, bc_i, ca_i, subs = _subdivide(f, pts)
            new_faces.extend(subs)
        faces = new_faces
    return pts, faces[:N]


def _energy_and_grad(pts, faces):
    """Log energy sum_{i<j} log(1/d) and gradient, with edges from the fixed mesh."""
    n = len(pts)
    tree = cKDTree(pts)
    # candidate edge set from mesh faces
    edge_set = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (a, c)):
            edge_set.add((min(u, v), max(u, v)))
    # add k-nearest to guarantee a connected, low-energy edge graph
    k = 12
    pairs = tree.query_pairs(r=10.0, k=k)
    for u, v in pairs:
        edge_set.add((min(u, v), max(u, v)))

    edges = list(edge_set)
    e = np.array(edges, dtype=int)
    d = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
    d = np.maximum(d, 1e-9)
    L = -np.log(d)
    energy = float(np.sum(L))

    # gradient: d/dx_i sum_{j} log(1/|x_i-x_j|) = - sum_j (x_i - x_j)/|x_i-x_j|^2
    g = np.zeros_like(pts)
    for u, v in edges:
        diff = pts[u] - pts[v]
        inv_d2 = 1.0 / (d2 := float(np.dot(diff, diff)))
        g[u] += diff * inv_d2
        g[v] -= diff * inv_d2
    # project onto tangent plane and normalise
    g = g - np.sum(g * pts, axis=1, keepdims=True) * pts
    return energy, g


def _project(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return pts / n


def construct():
    start = time.time()
    pts, faces = _build_mesh()
    pts = np.array(pts[:N], dtype=float)
    pts = _project(pts)

    # L-BFGS style line search via Adam-style momentum
    best = pts.copy()
    best_e, _ = _energy_and_grad(best, faces)
    v = np.zeros_like(pts)
    lr = 0.02
    for step in range(40000):
        if time.time() - start > 100.0:
            break
        e, g = _energy_and_grad(pts, faces)
        v = 0.9 * v - lr * g
        cand = _project(pts + v)
        ce, _ = _energy_and_grad(cand, faces)
        if ce < e:
            pts = cand
            if ce < best_e:
                best_e = ce
                best = cand.copy()
        else:
            lr *= 0.7
            v *= 0.5
    return best
# EVOLVE-BLOCK-END
