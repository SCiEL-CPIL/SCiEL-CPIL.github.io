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
from scipy.optimize import minimize

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
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    """
    core = []
    # Center
    core.append([0.0, 0.0, 0.0])
    
    # 12 atoms: permutations of (1, 1, 0)
    for i in range(3):
        for j in range(3):
            if i != j:
                # Generate permutations of (1, 1, 0)
                pass
    
    # Simpler: generate all FCC points up to dist^2 = 6
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    if d2 <= 6:
                        pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    
    # Separate by distance
    core_pts = pts[pts[:, 3] <= 4]
    outer_pts = pts[pts[:, 3] == 6]
    
    # core_pts should have 1 + 12 + 6 = 19 atoms
    # outer_pts should have 24 atoms
    return core_pts[:, :3], outer_pts[:, :3]

def _make_icosahedral_structure():
    """
    Icosahedral starting geometry.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

def _relax(pos, max_iter=500, gtol=1e-10):
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
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _optimize_scale(pos, n_iter=20):
    """Optimize the overall scale of the cluster to minimize energy."""
    best_pos = pos.copy()
    best_e, _ = _lj_energy_and_grad(best_pos)
    for _ in range(n_iter):
        for delta in [1.01, 0.99, 1.005, 0.995, 1.02, 0.98]:
            trial = best_pos * delta
            e, _ = _lj_energy_and_grad(trial)
            if e < best_e:
                best_e = e
                best_pos = trial
    return best_pos, best_e

def _make_truncated_octahedron_start():
    """
    Create a proper truncated octahedron (FCC) starting configuration for 38 atoms.
    The truncated octahedron has 24 vertices (outer) and 14 internal atoms for 38 total.
    Actually, the standard FCC truncated octahedron for N=38 is:
    - 1 center atom
    - 12 at (±1,±1,0) and permutations [distance sqrt(2)]
    - 6 at (±2,0,0) and permutations [distance 2]
    - 19 more atoms... 
    
    Let me use the known coordinates for the FCC 38-atom cluster.
    The global minimum is a FCC truncated octahedron.
    """
    # Generate FCC lattice points
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([float(i), float(j), float(k)])
    
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    pts -= center
    
    # Sort by distance from center
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    # Take the first 38
    pts = pts[:N].copy()
    
    # Scale so nearest neighbor distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / min_d
        pts *= scale
    
    return pts

def _basin_hop(pos, max_steps=50, step_size=0.3, temperature=1.0, rng=None):
    """
    Basin hopping: perturb, relax, accept/reject with Metropolis criterion.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_pos, current_e = _relax(pos, max_iter=200)
    best_pos, best_e = current_pos.copy(), current_e
    
    for step in range(max_steps):
        # Perturb
        noise = rng.normal(0, step_size, size=current_pos.shape)
        trial_pos = current_pos + noise
        trial_pos, trial_e = _relax(trial_pos, max_iter=200)
        
        # Metropolis accept/reject
        if trial_e < current_e:
            accept = True
        else:
            delta_e = trial_e - current_e
            accept = rng.random() < np.exp(-delta_e / temperature)
        
        if accept:
            current_pos, current_e = trial_pos, trial_e
            if current_e < best_e:
                best_pos, best_e = current_pos.copy(), current_e
        
        # Gradually increase temperature to escape deep local minima
        temperature *= 1.05
    
    return best_pos, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    rng = np.random.default_rng(42)
    
    best_pos = None
    best_energy = np.inf

    # 1. Start with the proper FCC truncated octahedron
    try:
        to_start = _make_truncated_octahedron_start()
        to_start, to_e = _optimize_scale(to_start, n_iter=10)
        to_start, to_e = _relax(to_start, max_iter=500, gtol=1e-12)
        if to_e < best_energy:
            best_energy = to_e
            best_pos = to_start.copy()
    except Exception:
        pass

    # 2. Basin hopping from the truncated octahedron start
    if best_pos is not None and time.time() < deadline - 60.0:
        try:
            bh_pos, bh_e = _basin_hop(best_pos, max_steps=30, step_size=0.2, temperature=0.5, rng=rng)
            if bh_e < best_energy:
                best_energy = bh_e
                best_pos = bh_pos.copy()
        except Exception:
            pass

    # 3. Additional random FCC-based starts with basin hopping
    core, outer = _generate_fcc_core_and_outer()
    num_extra_starts = 15
    for i in range(num_extra_starts):
        if time.time() > deadline - 40.0:
            break
        try:
            idx = rng.choice(len(outer), size=19, replace=False)
            start_pos = np.vstack([core, outer[idx]])
            start_pos -= np.mean(start_pos, axis=0)
            
            # Scale
            diff = start_pos[:, np.newaxis, :] - start_pos[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(r2, np.inf)
            nn = np.sqrt(np.min(r2, axis=1))
            avg_nn = np.mean(nn)
            if avg_nn > 1e-12:
                scale = (2.0 ** (1.0 / 6.0)) / avg_nn
                start_pos *= scale
            
            # Quick relax first
            p, e = _relax(start_pos, max_iter=100)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
            
            # Basin hop from this start if time permits
            if time.time() < deadline - 30.0 and e < best_energy + 5.0:
                bh_pos, bh_e = _basin_hop(p, max_steps=10, step_size=0.15, temperature=0.3, rng=rng)
                if bh_e < best_energy:
                    best_energy = bh_e
                    best_pos = bh_pos.copy()
        except Exception:
            continue

    # 4. Icosahedral start with basin hopping
    if time.time() < deadline - 30.0:
        try:
            ico = _make_icosahedral_structure()
            ico, ico_e = _relax(ico, max_iter=200)
            if ico_e < best_energy:
                best_energy = ico_e
                best_pos = ico.copy()
            
            if time.time() < deadline - 20.0:
                bh_pos, bh_e = _basin_hop(ico, max_steps=20, step_size=0.25, temperature=0.5, rng=rng)
                if bh_e < best_energy:
                    best_energy = bh_e
                    best_pos = bh_pos.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        det_start = _make_truncated_octahedron_start()
        best_pos, best_energy = _relax(det_start, max_iter=500)

    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=5000, gtol=1e-14)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # Optimize scale one more time
    if time.time() < deadline - 5.0:
        try:
            best_pos, best_energy = _optimize_scale(best_pos, n_iter=15)
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
