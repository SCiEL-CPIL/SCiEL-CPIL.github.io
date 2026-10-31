"""
LJ38 cluster global minimum search.

Approach:
1. Identify the core 19 atoms of the FCC truncated-octahedron (center + 12 nearest + 6 next-nearest).
2. Identify the outer 24 candidate atoms at the next shell distance.
3. Since the global minimum (38 atoms) is a symmetric truncated octahedron, it corresponds to a specific subset of 19 atoms from the 24 outer candidates.
4. We perform multiple local optimizations (L-BFGS) starting from different random subsets of 19 outer atoms.
5. We also include the deterministic "38 closest" subset and an icosahedral starting point.
6. The best energy configuration is returned.
"""
import time
import numpy as np
from scipy.optimize import minimize, basinhopping

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms,
    scaled so nearest-neighbor distance = 2^(1/6) (LJ equilibrium).
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    """
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    if 0 < d2 <= 6:
                        pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    
    # Core: d2 <= 4 (12 atoms at d2=2, 6 atoms at d2=4)
    core_pts = pts[pts[:, 3] <= 4]
    # Outer: d2 == 6 (24 atoms at d2=6)
    outer_pts = pts[pts[:, 3] == 6]
    
    # Add center
    center = np.array([[0.0, 0.0, 0.0, 0.0]])
    core_pts = np.vstack([center, core_pts])
    
    # Scale: FCC lattice with integer coords has NN distance sqrt(2).
    # We want NN distance = 2^(1/6), so scale factor = 2^(1/6) / sqrt(2)
    scale = (2.0 ** (1.0 / 6.0)) / np.sqrt(2.0)
    core_pts = core_pts[:, :3] * scale
    outer_pts = outer_pts[:, :3] * scale
    
    return core_pts, outer_pts

def _make_icosahedral_structure():
    """
    Icosahedral starting geometry for 38 atoms:
    1 center + 12 (icosahedron) + 12 (dodecahedron-like) + 13 (partial outer shell)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms
    
    # 12 dodecahedron vertices (dual to icosahedron)
    dodeca = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=np.float64)
    dodeca_norms = np.linalg.norm(dodeca, axis=1, keepdims=True)
    dodeca /= dodeca_norms
    
    # Build 38 atoms: 1 + 12 + 12 + 13
    pts = [np.zeros(3)]
    pts.extend(ico * 1.0)
    pts.extend(dodeca[:12] * 1.5)
    # Add 13 more from dodecahedron (remaining 8) plus some icosahedron directions
    pts.extend(dodeca[12:] * 1.8)
    pts.extend(ico[:1] * 2.2)  # just 1 more to reach 38
    
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    # Scale so average NN distance ~ 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

def _relax(pos, max_iter=500, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def _basinhopping_search(pos, max_time=10.0):
    """
    Perform a basin hopping search starting from pos.
    Returns (positions, energy).
    """
    t_start = time.time()
    
    def obj(x):
        p = x.reshape(-1, 3)
        e, _ = _lj_energy_and_grad(p)
        return e

    def grad(x):
        p = x.reshape(-1, 3)
        _, g = _lj_energy_and_grad(p)
        return g.ravel()

    # We wrap the call to basinhopping to monitor time
    # basinhopping doesn't have a direct time limit, so we limit niter and maxfun
    # and check time after each iteration if possible? No, it's a black box.
    # We'll set conservative limits.
    try:
        bh = basinhopping(
            obj,
            pos.ravel(),
            minimizer_kwargs={'jac': True, 'maxiter': 50, 'gtol': 1e-8},
            niter=20, # Limit number of steps
            stepsize=0.5,
            maxfun=500, # Limit function evaluations
            T=1.0,
            disp=False
        )
        p_out = bh.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin (evaluator kills at 150s)
    
    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    
    # 1. Deterministic "38 closest" - use all 19 core + 19 outer (closest to center)
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    starts.append(det_start)
    
    # 2. Symmetry-preserving perturbations of the FCC structure
    # The global min is a truncated octahedron with specific symmetry.
    # Apply small random perturbations that preserve the overall shape.
    for i in range(30):
        if time.time() > deadline - 40.0:
            break
        # Perturb outer atoms radially and angularly
        perturbed = det_start.copy()
        # Perturb the 19 outer atoms (indices 19-37)
        for j in range(19, 38):
            # Small random displacement
            disp = rng.normal(0, 0.1, size=3)
            perturbed[j] += disp
        starts.append(perturbed)
    
    # 3. Try different subsets of outer atoms with symmetry considerations
    # The 24 outer atoms form a cuboctahedron-like shell.
    # Try selecting 19 that form a more symmetric subset.
    for i in range(20):
        if time.time() > deadline - 40.0:
            break
        # Random subset but with bias toward symmetric selections
        if i % 3 == 0:
            # Take atoms from opposite faces preferentially
            idx = list(range(0, 24, 2))[:19]
        elif i % 3 == 1:
            idx = list(range(1, 24, 2))[:19]
        else:
            idx = list(rng.choice(24, size=19, replace=False))
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        starts.append(start_pos)
    
    # 4. Icosahedral start
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf

    # Phase 1: Local relaxation of all starts
    for start in starts:
        if time.time() > deadline - 25.0:
            break
        try:
            p, e = _relax(start, max_iter=300, gtol=1e-10)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Phase 2: Basin Hopping from the best local minimum found
    if time.time() < deadline - 15.0:
        try:
            p_bh, e_bh = _basinhopping_search(best_pos)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            pass

    # Phase 3: Try basin hopping from second-best and third-best
    if time.time() < deadline - 10.0:
        # Re-run relaxations to get sorted list
        relaxed = []
        for start in starts[:10]:
            if time.time() > deadline - 12.0:
                break
            try:
                p, e = _relax(start, max_iter=200, gtol=1e-10)
                relaxed.append((e, p))
            except Exception:
                continue
        relaxed.sort(key=lambda x: x[0])
        for e, p in relaxed[1:3]:  # Try 2nd and 3rd best
            if time.time() > deadline - 10.0:
                break
            try:
                p_bh, e_bh = _basinhopping_search(p)
                if e_bh < best_energy:
                    best_energy = e_bh
                    best_pos = p_bh
            except Exception:
                continue

    # Final polish
    if time.time() < deadline - 3.0:
        try:
            p, e = _relax(best_pos, max_iter=5000, gtol=1e-13, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    # Test
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
