"""Dzugutov cluster global minima: polytetrahedral seeds + divide-and-conquer assembly + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# ---------------- time management ----------------
_T0 = None
DEADLINE = 315.0  # safety margin before 330s kill

def _now():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0

def _remaining():
    return DEADLINE - _now()

# ---------------- geometry helpers ----------------
def _rotmat(rng):
    u, v, w = rng.normal(size=3)
    u /= np.linalg.norm(u)
    v = v - np.dot(v, u) * u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return np.column_stack((u, v, w))

def _icosahedron_verts():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    V = np.array([
        [-1, 0, phi], [1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, -1, 0], [phi, 1, 0], [-phi, -1, 0], [-phi, 1, 0],
    ], dtype=float)
    V /= np.linalg.norm(V[0])
    return V

_VERTS = _icosahedron_verts()
_EDGE_LEN = np.linalg.norm(_VERTS[0] - _VERTS[1])
_EDGE_MID = (_VERTS[0] + _VERTS[1]) / 2.0
_EDGE_MID /= np.linalg.norm(_EDGE_MID)
_EDGE_MID2 = (_VERTS[0] + _VERTS[2]) / 2.0
_EDGE_MID2 /= np.linalg.norm(_EDGE_MID2)

def _icosahedron_positions(rng, n):
    """Polytetrahedral Mackay icosahedron: centre + 12 vertices + 30 edge-midpoints."""
    pts = [np.zeros(3)]
    pts += [v.copy() for v in _VERTS]
    pts += [_EDGE_MID.copy() for _ in range(30)]
    pts = np.array(pts[:n], dtype=float)
    # random orientation (global rotation is irrelevant to energy)
    pts = pts @ _rotmat(rng)
    return pts

# ---------------- seed generation ----------------
def _random_cluster(rng, n):
    r = 1.15 * (n ** (1.0 / 3.0)) * 0.85
    pts = []
    for _ in range(20000):
        p = rng.uniform(-r, r, 3)
        if np.linalg.norm(p) <= r and all(np.linalg.norm(p - q) >= 0.55 for q in pts):
            pts.append(p)
        if len(pts) == n:
            break
    if len(pts) < n:  # fallback: dense random
        pts = [rng.uniform(-r, r, 3) for _ in range(n)]
    return np.array(pts, dtype=float)

def _assemble(rng, n):
    """Divide-and-conquer: relax a 13-atom icosahedron, attach a relaxed shell of the rest."""
    core_n = 13
    shell_n = n - core_n
    core = relax(_icosahedron_positions(rng, core_n), "dzugutov")[1]
    c = core.mean(axis=0)
    shell_pts = _random_cluster(rng, shell_n)
    shell_pts = shell_pts - shell_pts.mean(axis=0)
    shell = relax(shell_pts, "dzugutov")[1]
    shell = shell - shell.mean(axis=0)
    # place shell centroid at a comfortable distance from the core centroid
    dist = 1.15 * math.sqrt(shell_n) + 0.5
    direction = _EDGE_MID.copy()
    direction = direction @ _rotmat(rng)
    direction /= np.linalg.norm(direction)
    shell = shell + c + dist * direction
    return np.vstack([core, shell])

# ---------------- basin hopping ----------------
def _basin_hop(pos0, rng, budget):
    cur = pos0.copy()
    cur_e = float("inf")
    T = 0.15
    n = len(cur)
    while budget() > 0.4:
        try:
            e, p = relax(cur, "dzugutov")
        except Exception:
            break
        if e < cur_e:
            cur_e = e
            cur = p
        scale = min(2.0, 0.15 + 0.12 * math.sqrt(n) * (1.0 + 0.05 * rng.random()))
        pert = cur + scale * rng.normal(size=cur.shape)
        if rng.random() < 0.5:
            pert = pert @ _rotmat(rng)
        pert = pert - pert.mean(axis=0)
        # accept downhill or Metropolis
        try:
            e2, p2 = relax(pert, "dzugutov")
        except Exception:
            break
        if e2 < cur_e or rng.random() < math.exp(-(e2 - cur_e) / T):
            cur = p2
            cur_e = e2
    return cur_e, cur

# ---------------- per-instance driver ----------------
def _solve(instance):
    n = int(instance["n"])
    budget = _remaining
    best_e = float("inf")
    best_pos = None

    def upd(e, p):
        nonlocal best_e, best_pos
        if e < best_e:
            best_e = e
            best_pos = p.copy()

    # seed portfolio
    seeds = []
    seeds.append(_icosahedron_positions(np.random.default_rng(n * 7 + 1), n))
    seeds.append(_random_cluster(np.random.default_rng(n * 11 + 2), n))
    seeds.append(_random_cluster(np.random.default_rng(n * 13 + 3), n))
    if n >= 20:
        seeds.append(_assemble(np.random.default_rng(n * 17 + 4), n))
    seeds.append(_assemble(np.random.default_rng(n * 19 + 5), n))

    # relax each seed, keep the best
    for s in seeds:
        if budget() <= 0.6:
            break
        try:
            e, p = relax(s, "dzugutov")
            upd(e, p)
        except Exception:
            pass

    # basin hopping on the best structure
    if best_pos is not None and budget() > 1.0:
        rng = np.random.default_rng(n * 23 + 6)
        e, p = _basin_hop(best_pos, rng, budget)
        upd(e, p)

    # a couple of independent basin-hopping restarts from other seeds
    for i, s in enumerate(seeds[:2]):
        if budget() <= 1.0:
            break
        rng = np.random.default_rng(n * 29 + 7 + i)
        e, p = _basin_hop(s, rng, budget)
        upd(e, p)

    if best_pos is None:
        best_pos = _random_cluster(np.random.default_rng(n), n)
    return best_pos

# ---------------- entry point ----------------
def construct(instance):
    """Return Cartesian coordinates (n,3) of a low-energy Dzugutov cluster."""
    n = int(instance["n"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_solve, instance)
        try:
            return fut.result(timeout=max(0.5, _remaining()))
        except Exception:
            # graceful fallback: a valid random compact cluster
            rng = np.random.default_rng(n)
            return _random_cluster(rng, n)
# EVOLVE-BLOCK-END
