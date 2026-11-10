# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(ang), cy + s * np.sin(ang)])

def _cluster_positions():
    # Rigid cluster: center + 6 around + 6 second-ring (all same orientation)
    # Neighbor distance for touching unit hexagons (same orientation): 3/sqrt(3)
    d = 3.0 / np.sqrt(3.0)
    h = np.sqrt(3.0)
    pos = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [-d, 0.0],
        [d/2, d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        [-d/2, -d*np.sqrt(3)/2],
        [0.0, 2*d*np.sqrt(3)/2],
        [0.0, -2*d*np.sqrt(3)/2],
        [1.5*d, d*np.sqrt(3)/2],
        [-1.5*d, d*np.sqrt(3)/2],
    ])
    return pos

def _required_side(pos_rot, a_deg):
    # Required enclosing hexagon side (centered at origin, angle a_deg)
    # containing all cluster vertices (each cluster hex has circumradius 1)
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    normals = np.column_stack([np.cos(ang), np.sin(ang)])  # (6,2)
    # project all vertices
    proj = pos_rot @ normals.T  # (N,6)
    maxproj = np.max(proj, axis=0)  # (6,)
    # side = max over axes j of (maxproj[j] + maxproj[j+3] + 2)
    s = np.max(maxproj + np.roll(maxproj, 3) + 2.0)
    return s

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 580.0
    pos = _cluster_positions()

    # All inner hexagons share orientation 0; outer also orientation 0.
    # We rotate the cluster by theta (equivalently, outer stays at 0).
    # Try many theta values to find min enclosing side.
    best_theta = 0.0
    best_side = np.inf

    # Coarse scan
    thetas = np.linspace(0.0, 60.0, 601)
    for th in thetas:
        if time.time() > deadline - 10:
            break
        c, s_ = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
        R = np.array([[c, -s_], [s_, c]])
        pos_rot = pos @ R.T
        side = _required_side(pos_rot, 0.0)
        if side < best_side:
            best_side = side
            best_theta = th

    # Refine around best
    for _ in range(3):
        lo, hi = max(0.0, best_theta - 0.05), min(60.0, best_theta + 0.05)
        thetas = np.linspace(lo, hi, 2001)
        for th in thetas:
            if time.time() > deadline - 5:
                break
            c, s_ = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
            R = np.array([[c, -s_], [s_, c]])
            pos_rot = pos @ R.T
            side = _required_side(pos_rot, 0.0)
            if side < best_side:
                best_side = side
                best_theta = th

    # Final position: rotate cluster by best_theta, all angles 0
    th = best_theta
    c, s_ = np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))
    R = np.array([[c, -s_], [s_, c]])
    pos_rot = pos @ R.T

    inner = np.column_stack([pos_rot[:, 0], pos_rot[:, 1], np.zeros(len(pos_rot))])
    outer = np.array([0.0, 0.0, 0.0])
    side = float(best_side)

    return inner, outer, side

# EVOLVE-BLOCK-END
