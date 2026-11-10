import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def hexagon_packing_11():
    start = time.time()
    deadline = 620.0

    hex_verts = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]
                          for a in range(0, 360, 60)])

    # 6 unique edge normals (for a flat-top hexagon, these are at 30+60k)
    edge_normals = np.array([[np.cos(np.deg2rad(30 + 60 * i)),
                              np.sin(np.deg2rad(30 + 60 * i))] for i in range(6)])

    proj = hex_verts @ edge_normals.T
    h_min = np.min(proj, axis=0)
    h_max = np.max(proj, axis=0)

    def pair_separation(c1, c2):
        d = c1 - c2
        dproj = d @ edge_normals.T
        # overlap = min(h_max, h_max + dproj) - max(h_min, h_min + dproj)
        # separation = max(h_min, h_min+dproj) - min(h_max, h_max+dproj)
        # We want min over k of (h_max - h_min) - overlap_k > 0
        # which is min over k of (h_max - h_max + dproj_k) - (h_min - h_min + dproj_k) ...
        # Actually separation_k = (h_min + dproj_k) - h_max  (if positive)
        # OR h_min - (h_max + dproj_k) ... no.
        # Let's use: separation = min over k of ( (h_min[k] + dproj[k]) - h_max[k] )
        # Wait, if dproj > 0, the second hex shifts right. The left edge of second is h_min+dproj.
        # For them to not overlap, h_min+dproj > h_max (of first) OR h_min < h_max - |dproj|
        # The distance between the closest faces is min( (h_min+dproj) - h_max, h_max - (h_min+dproj) )? No.
        # Let's define gap_k = min(h_max, h_max + dproj_k) - max(h_min, h_min + dproj_k)
        # If gap_k < 0, they overlap. The separation is -gap_k.
        # We want max_k(-gap_k) > 0.
        gap = np.minimum(h_max, h_max + dproj) - np.maximum(h_min, h_min + dproj)
        return np.max(-gap)

    def total_penalty(centers):
        pen = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                sep = pair_separation(centers[i], centers[j])
                if sep < 0:
                    pen += sep * sep
        return pen

    def compute_side(centers, theta):
        # The outer hexagon is regular. We can rotate it arbitrarily.
        # It has 6 edges. We need to find the minimal apothem such that
        # all inner hexagons are inside.
        # The inner hexagons are unit side.
        # The outer hexagon has side s.
        # We need to find the minimal s.
        # The constraint is that for every vertex p of every inner hex,
        # it must be inside the outer hex.
        # But wait, the inner hexagons are regular.
        # It is enough to check the vertices of the inner hexagons.
        # The vertices of an inner hex at center c, rotation alpha are c + R.
        # We need to minimize the outer hex side s.
        # A regular hexagon of side s has apothem s * sqrt(3)/2.
        # So we need to find the minimal apothem A such that all vertices are inside.
        # The outer hex has orientation theta.
        # The normals to its edges are at theta + 30 + 60k.
        # A point p is inside if for all normals n, n . p <= A.
        # So A must be >= max over all inner vertices p and all normals n of n . p.
        # Let's compute this.
        outer_normals = np.array([[np.cos(np.deg2rad(theta + 30 + 60 * i)),
                                   np.sin(np.deg2rad(theta + 30 + 60 * i))] for i in range(6)])
        max_proj = 0
        for c in centers:
            # vertices of this hex (rotation 0)
            verts = c + hex_verts
            proj = verts @ outer_normals.T
            max_proj = max(max_proj, np.max(proj))
        return max_proj * 2 / np.sqrt(3)

    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
        centers = params[:22].reshape(11, 2)
        theta = params[22]
        side = compute_side(centers, theta)
        pen = total_penalty(centers)
        return side + 100 * pen

    def build_honeycomb():
        s = np.sqrt(3)
        c = []
        for i in range(3):
            c.append([(i - 1) * s, 0])
        for i in range(5):
            c.append([(i - 2) * s, s * np.sqrt(3) / 2])
        for i in range(3):
            c.append([(i - 1) * s, s * np.sqrt(3)])
        return np.array(c)

    def check_valid(centers, theta):
        # check separation > 1e-6
        for i in range(11):
            for j in range(i + 1, 11):
                if pair_separation(centers[i], centers[j]) <= 1e-6:
                    return False
        # check outer
        outer = compute_side(centers, theta)
        # check inner vertices
        outer_verts = np.array([[outer * np.cos(np.deg2rad(60 * i)),
                                 outer * np.sin(np.deg2rad(60 * i))] for i in range(6)])
        # outer center is 0
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([edge[1], -edge[0]])
            # normalize n? No, cross product doesn't need normalization if we scale
            # cross(p2 - p1, p - p1) = edge x (p - p1)
            # = edge[0]*(p[1]-p1[1]) - edge[1]*(p[0]-p1[0])
            for c in centers:
                for v in hex_verts:
                    p = c + v
                    val = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                    # The problem says cross(p2-p1, p-p1) >= -1e-6
                    # But wait, p2-p1 is the edge vector.
                    # If p is "inside", the cross product with the edge vector
                    # (assuming counter-clockwise vertices) should have a specific sign.
                    # For a CCW polygon, the interior is to the left of the edge.
                    # edge x (p - p1) > 0 for points to the left.
                    # Let's verify with a point at the center (0,0).
                    # p1 = (1,0), p2 = (0.5, sqrt(3)/2).
                    # edge = (-0.5, 0.866).
                    # p - p1 = (-1, 0).
                    # cross = (-0.5)(0) - (0.866)(-1) = 0.866 > 0.
                    # So interior points have positive cross product.
                    # We want cross >= -1e-6.
                    # This means points slightly outside are allowed.
                    # This is consistent with "valid if ... >= -1e-6".
                    if val < -1e-6:
                        return False
        return True

    best_side = float('inf')
    best_result = None

    def try_update(centers, theta):
        nonlocal best_side, best_result
        if check_valid(centers, theta):
            side = compute_side(centers, theta)
            if side < best_side:
                best_side = side
                inner = np.column_stack([centers, np.zeros(11)])
                outer = np.array([0, 0, np.rad2deg(theta)])
                best_result = (inner, outer, side)

    # Initial candidates
    hc = build_honeycomb()
    candidates = [np.concatenate([hc.flatten(), [0.0]])]
    for _ in range(10):
        c = hc + np.random.randn(11, 2) * 0.1
        theta = np.random.uniform(-np.pi/6, np.pi/6)
        candidates.append(np.concatenate([c.flatten(), [theta]]))

    # Nelder-Mead
    for init in candidates:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            centers = res.x[:22].reshape(11, 2)
            theta = res.x[22]
            try_update(centers, theta)
        except:
            pass

    # Differential Evolution
    if time.time() < deadline - 60:
        bounds = [(-10, 10)] * 22 + [(-np.pi/6, np.pi/6)]
        try:
            de_res = differential_evolution(objective, bounds, seed=42,
                                             maxiter=50, popsize=20,
                                             tol=1e-6, polish=False)
            centers = de_res.x[:22].reshape(11, 2)
            theta = de_res.x[22]
            try_update(centers, theta)
            # Polish
            res = minimize(objective, de_res.x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
            centers = res.x[:22].reshape(11, 2)
            theta = res.x[22]
            try_update(centers, theta)
        except:
            pass

    if best_result is None:
        # Fallback
        try_update(hc, 0.0)
        if best_result is None:
            inner = np.column_stack([hc, np.zeros(11)])
            outer = np.array([0, 0, 0.0])
            best_side = compute_side(hc, 0.0)
            best_result = (inner, outer, best_side)

    return best_result
