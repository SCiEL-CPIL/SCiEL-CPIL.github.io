import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
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

def _generate_fcc_points(max_c=4, d2_max=20.0):
    """Generate FCC lattice points using proper primitive vectors."""
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= d2_max:
                    pts.append([p[0], p[1], p[2], d2])
    
    if not pts:
        return np.array([])
    
    pts = np.array(pts)
    # Remove duplicates
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    return pts

def _make_truncated_octahedron_start():
    """
    Create a starting geometry closely resembling the FCC truncated octahedron.
    Takes the 38 closest FCC lattice points to the origin.
    """
    pts = _generate_fcc_points(max_c=4, d2_max=20.0)
    if len(pts) < N:
        # Fallback to smaller search
        pts = _generate_fcc_points(max_c=3, d2_max=12.0)
    
    if len(pts) < N:
        return None
    
    # Sort by distance from origin and take N closest
    pts = pts[np.argsort(pts[:, 3])]
    selected = pts[:N, :3].copy()
    selected -= np.mean(selected, axis=0)
    
    # Scale so nearest-neighbor distance is ~2^(1/6)
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn_dists = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn_dists)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
    
    return selected

def _make_icosahedral_structure():
    """Icosahedral starting geometry (known local minimum trap)."""
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
    # Add more atoms to reach 38
    while len(pts) < N:
        # Add atoms at slightly perturbed icosahedral positions
        idx = len(pts) % 12
        base = ico[idx]
        jitter = 0.15 * np.array([1, 0, 0])
        pts.append(base * 2.0 + jitter)
    
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

def _optimize_scale(pos):
    """Optimize global scale to minimize energy (1D optimization)."""
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-8})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure minimum separation between atoms."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def _basin_hop(pos, rng, max_iters=30, T_start=0.3, T_end=0.02):
    """
    Basin hopping: perturb, relax, accept/reject by Metropolis criterion.
    Helps escape local minima like the icosahedral funnel.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy

    for i in range(max_iters):
        # Perturbation with annealed noise
        T_curr = T_start * (T_end / T_start) ** (i / max_iters)
        noise_scale = T_curr
        perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)
        
        # Relax
        relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-9)
        e_relaxed, _ = _lj_energy_and_grad(relaxed)
        
        # Metropolis acceptance
        if e_relaxed < current_energy:
            accept = True
        else:
            delta_e = e_relaxed - current_energy
            if delta_e > 50 * T_curr:
                accept = False
            else:
                accept = rng.random() < np.exp(-delta_e / T_curr)
        
        if accept:
            current_pos = relaxed.copy()
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin (evaluator kills at 150s)
    
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_energy = np.inf
    
    def try_configuration(pos, label=""):
        nonlocal best_pos, best_energy
        if time.time() > deadline - 15.0:
            return
        try:
            # Quick scale optimization
            pos_scaled, e_scaled = _optimize_scale(pos)
            # Relax
            p, e = _relax(pos_scaled, max_iter=500)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    # 1. Start with truncated octahedron (closest to known global minimum)
    to_start = _make_truncated_octahedron_start()
    if to_start is not None:
        try_configuration(to_start, "truncated_octahedron")
    
    # 2. Basin hopping from truncated octahedron
    if time.time() < deadline - 40.0 and to_start is not None:
        try:
            bh_pos, bh_energy = _basin_hop(to_start, rng, max_iters=20, T_start=0.25, T_end=0.02)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass
    
    # 3. Icosahedral structure (to explore that basin, then escape)
    ico_start = _make_icosahedral_structure()
    if time.time() < deadline - 35.0:
        try_configuration(ico_start, "icosahedral")
        # Basin hopping from icosahedral to escape the funnel
        if time.time() < deadline - 25.0:
            try:
                bh_pos, bh_energy = _basin_hop(ico_start, rng, max_iters=15, T_start=0.35, T_end=0.03)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos.copy()
            except Exception:
                pass
    
    # 4. Random FCC-based starts with different subsets
    if time.time() < deadline - 20.0:
        pts = _generate_fcc_points(max_c=4, d2_max=20.0)
        if len(pts) >= N + 10:
            # Try several random selections of N points from the FCC lattice
            num_random = 15
            for i in range(num_random):
                if time.time() > deadline - 15.0:
                    break
                idx = rng.choice(len(pts), size=N, replace=False)
                start_pos = pts[idx, :3].copy()
                start_pos -= np.mean(start_pos, axis=0)
                try_configuration(start_pos, f"fcc_random_{i}")
    
    # 5. Final polish with tight tolerance
    if best_pos is not None and time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=5000, gtol=1e-14)
            p, e = _optimize_scale(p)
            p, e_final = _relax(p, max_iter=2000, gtol=1e-14)
            if e_final < best_energy:
                best_energy = e_final
                best_pos = p
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        if to_start is not None:
            best_pos = to_start
        else:
            # Last resort: simple cubic-like arrangement
            best_pos = np.zeros((N, 3))
            for i in range(N):
                best_pos[i] = [i % 5, (i // 5) % 4, i // 20]
            best_pos -= np.mean(best_pos, axis=0)
            best_pos *= 1.2
    
    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
