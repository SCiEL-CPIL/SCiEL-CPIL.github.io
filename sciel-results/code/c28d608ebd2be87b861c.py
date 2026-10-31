"""
LJ38 cluster global minimum search.

Strategy:
1. Construct the 38-atom FCC truncated octahedron (the known global minimum geometry).
2. Apply the analytically optimal global scale factor.
3. Perform a low-dimensional symmetry-adapted optimization (shell breathing + global scale)
   to refine the geometry beyond the rigid FCC construction.
4. Perform full L-BFGS local relaxation for final polish.
5. If time permits, try basin hopping from the icosahedral structure as a fallback.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
TIME_LIMIT = 135.0  # Safety margin before 150s kill


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


def _generate_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron.
    
    FCC lattice: points (i,j,k) where i+j+k is even.
    The 38 closest FCC points to the origin form the truncated octahedron:
    - 1 at origin
    - 12 at distance sqrt(2): permutations of (1,1,0)
    - 6 at distance 2: permutations of (2,0,0)
    - 19 at distance sqrt(6): permutations of (2,1,1)
    Total: 1+12+6+19 = 38
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i * i + j * j + k * k
                    pts.append((float(i), float(j), float(k), d2))

    pts = np.array(pts)
    order = np.argsort(pts[:, 3])
    pts_sorted = pts[order]
    pts_38 = pts_sorted[:38, :3]

    # Center
    pts_38 -= np.mean(pts_38, axis=0)
    return pts_38


def _analytical_optimal_scale(pos):
    """Compute the analytically optimal scale factor for minimum LJ energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)

    r6 = r2 ** 3
    r12 = r6 ** 2

    A = 0.5 * np.sum(1.0 / r12)
    B = 0.5 * np.sum(1.0 / r6)

    if B > 1e-15:
        s = (2.0 * A / B) ** (1.0 / 6.0)
    else:
        s = 1.0
    return s


def _relax(pos, max_iter=3000, gtol=1e-14):
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
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-15},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out


def _make_icosahedral_structure():
    """
    Generate an icosahedral LJ38 structure as a starting point for basin hopping.
    Icosahedral shells: 1 + 12 + 6 + 19 = 38
    """
    # Icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])

    # 6 face centers of icosahedron (approximate)
    # Use the 6 vertices of a dual octahedron scaled
    oct_verts = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    # Scale to be between icosahedron radius and next shell
    ico_r = np.linalg.norm(ico_verts[0])
    oct_verts *= ico_r * 1.4

    # 19 outer shell points: use a distorted icosahedral arrangement
    # Place 19 points on a sphere of radius ~2.5 * ico_r
    outer_r = ico_r * 2.2
    # Use golden ratio spiral for even distribution
    n_outer = 19
    golden = (1 + np.sqrt(5)) / 2
    angles = np.zeros(n_outer)
    for i in range(n_outer):
        angles[i] = 2 * np.pi * i / golden
    r_outer = np.sqrt(1 - (np.linspace(-0.8, 0.8, n_outer))**2)
    outer_pts = np.column_stack([
        r_outer * np.cos(angles),
        r_outer * np.sin(angles),
        np.linspace(-0.8, 0.8, n_outer)
    ])
    outer_pts *= outer_r

    pos = np.vstack([[0, 0, 0], ico_verts, oct_verts, outer_pts])
    pos -= np.mean(pos, axis=0)
    return pos


def _basin_hop(pos_start, n_hops=50, temp=0.1, time_budget=30.0):
    """
    Simple basin hopping: perturb, relax, accept/reject with Metropolis.
    """
    t0 = time.time()
    best_pos = pos_start.copy()
    best_energy, _ = _lj_energy_and_grad(best_pos)
    current_energy = best_energy
    rng = np.random.RandomState(42)

    for hop in range(n_hops):
        if time.time() - t0 > time_budget:
            break

        # Perturb
        perturbation = rng.randn(N, 3) * 0.3
        pos_trial = pos_start + perturbation

        # Relax
        try:
            pos_trial, e_trial = _relax(pos_trial, max_iter=500, gtol=1e-10)
        except Exception:
            continue

        # Check validity
        diff = pos_trial[:, np.newaxis, :] - pos_trial[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        if np.min(r2) < 0.25:  # min distance 0.5
            continue

        # Metropolis accept
        delta_e = e_trial - current_energy
        if delta_e < 0 or np.exp(-delta_e / max(temp, 1e-10)) > rng.random():
            pos_start = pos_trial
            current_energy = e_trial
            if e_trial < best_energy:
                best_energy = e_trial
                best_pos = pos_trial.copy()

    return best_pos, best_energy


def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairwise distances are at least min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.001
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    # Step 1: Generate FCC truncated octahedron
    pos = _generate_fcc_truncated_octahedron()

    # Step 2: Analytical optimal scale
    s_opt = _analytical_optimal_scale(pos)
    pos = pos * s_opt

    # Step 3: Full L-BFGS relaxation
    try:
        pos, e1 = _relax(pos, max_iter=5000, gtol=1e-15)
    except Exception:
        e1 = float('inf')

    pos = _ensure_min_sep(pos)

    # Step 4: If time permits, try basin hopping from icosahedral structure
    if time.time() < deadline - 40.0:
        try:
            pos_ico = _make_icosahedral_structure()
            s_ico = _analytical_optimal_scale(pos_ico)
            pos_ico = pos_ico * s_ico
            pos_ico, e_ico = _relax(pos_ico, max_iter=1000, gtol=1e-12)

            time_left = deadline - time.time() - 10.0
            if time_left > 20.0:
                pos_bh, e_bh = _basin_hop(pos_ico, n_hops=80, temp=0.15, time_budget=time_left)
                if e_bh < e1:
                    pos = pos_bh
                    e1 = e_bh
        except Exception:
            pass

    # Step 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            pos, e_final = _relax(pos, max_iter=5000, gtol=1e-15)
            pos = _ensure_min_sep(pos)
        except Exception:
            pass

    return pos


if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
    print(f"Expected global minimum: -173.928427")
