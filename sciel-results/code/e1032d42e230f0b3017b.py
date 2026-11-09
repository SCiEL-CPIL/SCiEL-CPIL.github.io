"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 310.0  # safety margin under 330s kill
_t0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _t0)

# Known published energies (Doye & Wales 1998) for reference; not used in scoring
# but useful for sanity checks.
_E_PUB = {
    ("12-6", 38): -568.54,
    ("12-6", 75): -1991.9,
    ("9-6", 59): -1679.6,
    ("9-6", 80): -2543.2,
    ("10-8", 55): -1383.8,
    ("10-8", 79): -2413.5,
}

def _spacing(kind):
    return {"12-6": 0.72, "9-6": 0.70, "10-8": 0.71}.get(kind, 0.72)

def _generate_fcc(n, rng, s):
    """n closest FCC lattice points to origin."""
    basis = np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]], dtype=float)
    L = int(np.ceil((n/4)**(1/3))) + 3
    pts = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = np.array([i,j,k]) + b
                    pts.append(p)
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)[:n]
    return pts[order] * s

def _generate_hcp(n, rng, s):
    """n closest HCP lattice points to origin."""
    a = 1.0
    c = np.sqrt(8/3)
    basis = np.array([
        [0,0,0],[1,0,0],[0.5,np.sqrt(3)/2,0],
        [1/3, np.sqrt(3)/9, c/3],
        [2/3, -np.sqrt(3)/9, c/3],
        [0.5, np.sqrt(3)/2, c/3],
    ])
    L = int(np.ceil((n/6)**(1/3))) + 3
    pts = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for b in basis:
                    p = np.array([i,j,k]) + b
                    pts.append(p)
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)[:n]
    return pts[order] * s

def _generate_icosahedral(n, rng, s):
    """Icosahedral shells (Mackay) up to n atoms."""
    # shell radii: 1, sqrt(10)/2 (~1.581), sqrt(5) (~2.236)
    # shell sizes: 1, 12, 42, 92
    shells = [
        np.array([[0,0,0]]),
        np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
                  [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],[0,1,1],[0,-1,1],
                  [0,1,-1],[0,-1,-1],[1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1]]*0.57735),
    ]
    # icosahedron 12 vertices
    phi = (1+np.sqrt(5))/2
    ico12 = np.array([
        [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
        [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
        [phi,0,1],[-phi,0,1],[phi,0,-1],[-phi,0,-1]
    ])
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
    pts = [np.array([[0,0,0]])]
    count = 1
    shell_list = [ico12]
    # second shell: 30 atoms (edges of icosahedron, midpoints)
    # approximate by scaling
    r2 = np.sqrt(10)/2
    # 30 points: icosahedron edge midpoints
    edges = []
    v = ico12
    for a in range(12):
        for b in range(a+1, 12):
            d = np.linalg.norm(v[a]-v[b])
            if d < 1.2:  # edge length
                m = (v[a]+v[b])/2
                edges.append(m)
    edges = np.array(edges)
    edges = edges / np.linalg.norm(edges, axis=1, keepdims=True) * r2
    shell_list.append(edges)
    # third shell: 60 atoms (face centers) approx
    r3 = np.sqrt(5)
    # use more points: take icosahedron vertices scaled + edge midpoints scaled
    face_pts = []
    for a in range(12):
        for b in range(a+1,12):
            for c2 in range(b+1,12):
                tri = np.linalg.norm(v[a]-v[b]) < 1.2 and np.linalg.norm(v[b]-v[c2])<1.2 and np.linalg.norm(v[a]-v[c2])<1.2
                if tri:
                    fc = (v[a]+v[b]+v[c2])/3
                    face_pts.append(fc)
    face_pts = np.array(face_pts) if face_pts else np.zeros((0,3))
    if len(face_pts) > 0:
        face_pts = face_pts / np.linalg.norm(face_pts, axis=1, keepdims=True) * r3
    shell_list.append(face_pts)
    all_pts = [np.array([[0,0,0]])]
    total = 1
    for sh in shell_list:
        if total >= n:
            break
        take = min(len(sh), n - total)
        all_pts.append(sh[:take])
        total += take
    if total < n:
        # fill with random compact
        rad = s * n**(1/3) * 0.8
        while total < n:
            p = rng.uniform(-rad, rad, 3)
            all_pts.append(p.reshape(1,3))
            total += 1
    pts = np.vstack(all_pts)
    return pts * s

def _generate_random_compact(n, rng, s):
    rad = s * n**(1/3) * 0.75
    pts = []
    min_d = 0.55 * s
    tries = 0
    while len(pts) < n and tries < n*200:
        p = rng.uniform(-rad, rad, 3)
        if np.linalg.norm(p) <= rad and all(np.linalg.norm(p-q) >= min_d for q in pts):
            pts.append(p)
        tries += 1
    while len(pts) < n:
        pts.append(rng.uniform(-rad, rad, 3))
    return np.array(pts)

def _templates(n, kind, rng):
    s = _spacing(kind)
    return [
        _generate_fcc(n, rng, s),
        _generate_hcp(n, rng, s),
        _generate_icosahedral(n, rng, s),
        _generate_random_compact(n, rng, s),
    ]

def _sa_search(pos, kind, t_budget, rng):
    """Simple simulated annealing with local relax restarts."""
    pos = pos.copy()
    e, _ = sutton_chen_energy_gradient(pos, kind)
    T = 0.5
    T_min = 0.01
    n = pos.shape[0]
    step0 = 0.1
    iters = max(50, int(t_budget / 0.005))
    for it in range(iters):
        if _remaining() < 5:
            break
        # local relax every 100 iters
        if it % 100 == 0 and it > 0:
            e, pos = relax(pos, "sutton_chen", kind)
        step = step0 * (T / 0.5)
        delta = rng.normal(0, step, 3)
        new_pos = pos + delta
        new_e, _ = sutton_chen_energy_gradient(new_pos, kind)
        if not np.isfinite(new_e):
            continue
        if new_e < e or rng.random() < np.exp(-(new_e - e)/max(T, 1e-9)):
            pos = new_pos
            e = new_e
        T *= 0.999
        if T < T_min:
            T = T_min
    e, pos = relax(pos, "sutton_chen", kind)
    return e, pos

def construct(instance):
    global _t0
    _t0 = time.time()
    n = instance["n"]
    kind = instance["kind"]
    rng = np.random.default_rng(instance.get("seed", 42) + n)

    templates = _templates(n, kind, rng)
    best_e = np.inf
    best_pos = templates[0]
    relaxed = []
    for tpl in templates:
        if _remaining() < 15:
            break
        e, pos = relax(tpl, "sutton_chen", kind)
        if np.isfinite(e) and e < best_e:
            best_e = e
            best_pos = pos
        relaxed.append((e, pos))
    relaxed.sort(key=lambda x: x[0])

    # SA on top 3 relaxed starts
    sa_budget = min(_remaining() * 0.6, 120.0)
    per_sa = sa_budget / 3.0
    for i, (e, pos) in enumerate(relaxed[:3]):
        if _remaining() < 10:
            break
        e2, pos2 = _sa_search(pos, kind, per_sa, rng)
        if np.isfinite(e2) and e2 < best_e:
            best_e = e2
            best_pos = pos2

    # final relax
    e, pos = relax(best_pos, "sutton_chen", kind)
    if np.isfinite(e):
        best_e = e
        best_pos = pos
    return best_pos
# EVOLVE-BLOCK-END
