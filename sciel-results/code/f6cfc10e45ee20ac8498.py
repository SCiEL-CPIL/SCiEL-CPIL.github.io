# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side(inner_hex_data, outer_center=None, outer_angle=0.0):
    if outer_center is None:
        outer_center = np.array([0.0, 0.0])
    all_verts = np.vstack([_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner_hex_data])
    all_verts -= outer_center
    oa = np.radians(outer_angle)
    ov = np.column_stack([np.cos(oa + np.radians(60)*np.arange(6)),
                          np.sin(oa + np.radians(60)*np.arange(6))])
    max_s = 0.0
    for i in range(6):
        edge = ov[(i+1) % 6] - ov[i]
        normal = np.array([edge[1], -edge[0]])
        normal /= np.linalg.norm(normal)
        max_proj = np.max(all_verts @ normal)
        max_s = max(max_s, max_proj / (np.sqrt(3)/2))
    return max_s


def _hexagons_overlap(vi, vj, tol=1e-6):
    """Return True if two hexagons overlap using SAT."""
    for verts in [vi, vj]:
        for k in range(6):
            edge = verts[(k+1) % 6] - verts[k]
            normal = np.array([edge[1], -edge[0]])
            normal /= np.linalg.norm(normal)
            pi = vi @ normal
            pj = vj @ normal
            if np.max(pi) < np.min(pj) - tol or np.max(pj) < np.min(pi) - tol:
                return False  # Separating axis found
    return True  # No separating axis -> overlap


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590
    sqrt3 = np.sqrt(3)
    
    inner1 = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-3.0, 0.0, 0.0],
        [1.5, sqrt3, 0.0], [-1.5, sqrt3, 0.0], [4.5, sqrt3, 0.0], [-4.5, sqrt3, 0.0],
        [1.5, -sqrt3, 0.0], [-1.5, -sqrt3, 0.0], [4.5, -sqrt3, 0.0], [-4.5, -sqrt3, 0.0],
        [0.0, 2*sqrt3, 0.0],
    ])
    s1 = _min_outer_side(inner1)
    best_inner, best_s = inner1.copy(), s1
    
    inner2 = np.array([[0.0, 0.0, 0.0]])
    for i in range(6):
        ang = np.radians(30 + 60*i)
        inner2 = np.vstack([inner2, [sqrt3*np.cos(ang), sqrt3*np.sin(ang), 0.0]])
    for i in range(5):
        ang = np.radians(60*i)
        inner2 = np.vstack([inner2, [2*sqrt3*np.cos(ang), 2*sqrt3*np.sin(ang), 0.0]])
    s2 = _min_outer_side(inner2)
    if s2 < best_s:
        best_inner, best_s = inner2.copy(), s2
    
    return best_inner, np.array([0.0, 0.0, 0.0]), best_s
# EVOLVE-BLOCK-END
