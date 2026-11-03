import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
EPS = 1e-9


def _unit_hex_normals():
    # outward normals of a unit regular hexagon (pointy-top, flat top/bottom faces)
    return np.array([
        [math.cos(math.pi / 6 + k * math.pi / 3),
         math.sin(math.pi / 6 + k * math.pi / 3)]
        for k in range(6)
    ])


UNIT_NORM = _unit_hex_normals()


def _hexagon_vertices(cx, cy, s, ang):
    # 6 vertices of a regular hexagon of side s, center (cx,cy), rotation ang (radians)
    c, si = math.cos(ang), math.sin(ang)
    R = s * math.sqrt(3) / 2.0
    verts = []
    for k in range(6):
        a = math.pi / 6 + k * math.pi / 3
        vx = R * math.cos(a)
        vy = R * math.sin(a)
        verts.append((cx + c * vx - si * vy, cy + si * vx + c * vy))
    return np.array(verts)


def _hexagon_normals(ang):
    c, si = math.cos(ang), math.sin(ang)
    return UNIT_NORM @ np.array([[c, -si], [si, c]])


def _collision_offsets(a, b, na, nb):
    # 12 pairwise edge-plane offsets; all must be >=0 for disjointness
    ab = b - a
    # 6: b's vertices vs a's faces (normal na_k), offset = dot(va_k - b, na_k)
    va = _hexagon_vertices(0, 0, 1.0, 0.0)  # unit, ang=0
    # rotate a's vertices by na's rotation? We'll build properly:
    return None  # replaced below


def _pairwise_offsets(ca, cb, na, nb):
    # ca,cb centers; na,nb are 6x2 outward normal matrices (already rotated)
    # a's vertices (unit, rotation matching na): unit verts rotated by na
    # We derive unit vertices consistent with UNIT_NORM orientation.
    # For pointy-top: vertex k at angle pi/6 + k*pi/3, radius R=apothem (unit side -> R=apothem)
    R = APOTHEM  # circumradius? No: for side=1, apothem=sqrt3/2, circumradius=1
    # Actually circumradius of unit-side regular hexagon = 1. Vertex radius = 1.
    # Vertex k direction = normal direction (since vertex lies along normal for regular hexagon)
    # So vertex_k = center + 1.0 * (unit normal direction k)
    va = ca + na.T * 1.0  # 6x2 : each vertex = center + normal_dir * 1.0 (circumradius=1)
    vb = cb + nb.T * 1.0
    # offset for face k of a: min over j of dot(va_k - vb_j, na_k)
    # = dot(va_k - cb, na_k) - max_j dot(vb_j - cb, na_k)
    # dot(vb_j - cb, na_k) = dot(nb_j, na_k)  (since vb_j - cb = nb_j * 1.0)
    # dot(va_k - cb, na_k) = dot(na_k, na_k) = 1
    # So offset_a_k = 1 - max_j dot(nb_j, na_k)
    # Similarly offset_b_k = 1 - max_j dot(na_j, nb_k)
    # Both must be >= 0.
    # max_j dot(nb_j, na_k): compute all 6x6 dot products.
    D = na @ nb.T  # 6x6, D[k,j] = dot(na_k, nb_j)
    off_a = 1.0 - D.max(axis=1)  # 6
    off_b = 1.0 - D.T.max(axis=1)  # 6
    return np.concatenate([off_a, off_b])  # 12 offsets


def _objective(params, temperature):
    # params: 11*2 + 2 + 1 + 1 = 26
    # [x1..x11, y1..y11, cx, cy, S, ang]
    centers = params[:22].reshape(11, 2)
    cx, cy = params[22], params[23]
    S = params[24]
    ang = params[25]

    # Collision barrier
    barrier = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            # rotations of inner hexagons: keep all at 0 for simplicity? 
            # The task allows rotation. But to keep 26 vars, fix inner angles=0.
            # (Inner hexagons all axis-aligned; only outer rotates.)
            na = UNIT_NORM  # identity rotation
            nb = UNIT_NORM
            offs = _pairwise_offsets(centers[i], centers[j], na, nb)
            m = offs.max()
            shifted = (offs - m) / temperature
            shifted = np.clip(shifted, -50, 50)
            logsum = np.log(np.sum(np.exp(shifted))) + m / temperature
            minoff = temperature * logsum
            barrier += minoff

    # Containment: outer hexagon side S, center (cx,cy), rotation ang
    if S < 0.1:
        return 1e10 + 1e6 * barrier
    # outer normals
    outer_norm = _hexagon_normals(ang)
    apothem_outer = S * APOTHEM
    # For each inner hexagon vertex (11*6=66), projection onto each outer normal
    # vertex of inner i = center_i + UNIT_NORM.T (circumradius 1)
    # proj on outer normal k: dot(vertex - outer_center, outer_norm_k)
    # must satisfy |proj| <= apothem_outer
    # max over all of |proj|
    # Compute all 66x6 projections
    inner_verts = centers[:, None, :] + UNIT_NORM.T[None, :, :]  # 11x6x2
    inner_verts = inner_verts - np.array([cx, cy])  # 11x6x2
    # proj[i,j,k] = dot(inner_verts[i,j], outer_norm[k])
    # shape 11x6x6
    proj = np.einsum('ijk,kl->ijl', inner_verts, outer_norm)  # 11x6x6
    proj = proj.reshape(-1, 6)  # 66x6
    absproj = np.abs(proj)
    maxproj = absproj.max()
    # smooth max via logsumexp over 66*6*2? Use max directly for objective (it's piecewise smooth)
    # For gradient, use soft-max:
    all_abs = np.concatenate([absproj.ravel(), absproj.ravel()])  # not needed; absproj already positive
    m = absproj.max()
    shifted = (absproj - m) / temperature
    shifted = np.clip(shifted, -50, 50)
    logsum = np.log(np.sum(np.exp(shifted))) + m / temperature
    smooth_max = temperature * logsum

    containment_violation = smooth_max - apothem_outer
    # objective: maximize S (we minimize -S) but penalize violations
    # Use: minimize  containment_violation^2 + barrier^2 - S  (so larger S preferred if feasible)
    # Actually we want smallest S such that feasible. 
    # Reformulate: minimize S  subject to containment_violation <= 0 and barrier >= 0.
    # Use penalty: S + lam*(max(0,cont_viol)^2 + max(0,-barrier)^2)
    cont_pen = max(0.0, containment_violation) ** 2
    # barrier: minoff should be >= 0. penalty for minoff < 0:
    # We already summed minoffs (each ~min offset). If any negative, barrier negative.
    # Penalty: max(0, -barrier)^2
    bar_pen = max(0.0, -barrier) ** 2

    return S + 100.0 * (cont_pen + bar_pen)


def construct():
    deadline = START_TIME + TIME_LIMIT - SAFETY_MARGIN

    best_S = float('inf')
    best_params = None

    # Random starts
    rng = np.random.default_rng(42)
    n_starts = 40
    temperatures = [0.5, 0.2, 0.1]

    for start_idx in range(n_starts):
        if time.time() > deadline:
            break

        # Random initial layout: hexagons in a rough cluster
        # Place 11 centers randomly in a disk of radius ~3
        r = rng.uniform(0, 3.0, size=(11, 2))
        # Convert to polar to make disk
        theta = rng.uniform(0, 2 * math.pi, size=(11, 1))
        r = r[:, 0:1] * np.sqrt(rng.uniform(0, 1, size=(11, 1)))
        cx_init = r * np.cos(theta)
        cy_init = r * np.sin(theta)
        centers = np.column_stack([cx_init, cy_init])

        cx0 = 0.0
        cy0 = 0.0
        S0 = 4.0
        ang0 = 0.0

        params0 = np.concatenate([centers.ravel(), [cx0, cy0, S0, ang0]])

        for temp in temperatures:
            if time.time() > deadline:
                break
            try:
                res = minimize(
                    lambda p: _objective(p, temp),
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12},
                    bounds=[(None, None)] * 22 + [(None, None), (None, None), (0.5, 10.0), (0.0, math.pi / 6)]
                )
                if res.fun < 1e9:
                    S_candidate = res.x[24]
                    # Verify feasibility with exact check
                    centers_c = res.x[:22].reshape(11, 2)
                    cx_c, cy_c = res.x[22], res.x[23]
                    S_c = res.x[24]
                    ang_c = res.x[25]
                    # Check collisions
                    feasible = True
                    for i in range(11):
                        for j in range(i + 1, 11):
                            offs = _pairwise_offsets(centers_c[i], centers_c[j], UNIT_NORM, UNIT_NORM)
                            if offs.min() < -1e-7:
                                feasible = False
                                break
                        if not feasible:
                            break
                    # Check containment
                    if feasible:
                        outer_norm = _hexagon_normals(ang_c)
                        apothem_outer = S_c * APOTHEM
                        inner_verts = centers_c[:, None, :] + UNIT_NORM.T[None, :, :]
                        inner_verts = inner_verts - np.array([cx_c, cy_c])
                        proj = np.einsum('ijk,kl->ijl', inner_verts, outer_norm)
                        maxproj = np.abs(proj).max()
                        if maxproj > apothem_outer + 1e-7:
                            feasible = False
                    if feasible and S_c < best_S:
                        best_S = S_c
                        best_params = res.x.copy()
                params0 = res.x.copy()
            except Exception:
                pass

    # If no feasible found, fallback: use a simple grid of 11 hexagons
    if best_params is None:
        # Fallback: 11 hexagons in a 3x4 grid-ish arrangement, all axis-aligned
        # Use axial coordinates of a compact shape
        axial = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3), (2, 3)
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        pts = np.array(pts)
        # Center
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        # Compute required S
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        best_S = S
        best_params = np.concatenate([pts.ravel(), [cx, cy, S, 0.0]])

    # Extract result
    centers = best_params[:22].reshape(11, 2)
    cx = best_params[22]
    cy = best_params[23]
    S = best_params[24]
    ang = best_params[25]

    # Slight safety scaling of centers toward center to ensure no boundary overlap
    # Actually, just return as is; the optimizer should have ensured feasibility.
    # Apply tiny inward shift of outer? No, return exact.

    inner = []
    for i in range(11):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), 0.0])

    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(math.degrees(ang))

    return inner, center, side, rotation
