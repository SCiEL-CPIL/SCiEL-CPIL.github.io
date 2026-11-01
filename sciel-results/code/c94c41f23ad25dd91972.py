"""Lennard-Jones 55-atom cluster: Mackay icosahedron with radial optimization."""
N = 55
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Vectorized LJ energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_grad(pos):
    """Vectorized LJ gradient."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    # coeff[i,j] = 4*(-12/r2*r6 + 6/r6) = 4*(-12/r8 + 6/r6)
    coeff = 4.0 * (-12.0 / (r2 * r6) + 6.0 / r6)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _get_mackay_directions():
    """
    Return 55 unit direction vectors for the Mackay icosahedron:
    - 1 center (direction irrelevant, radius=0)
    - 12 icosahedron vertex directions (shell 1)
    - 20 face-center directions (shell 2a)
    - 30 edge-midpoint directions (shell 2b)
    Total: 1 + 12 + 20 + 30 = 63... that's too many.
    
    Actually for 55 atoms: 1 + 12 + 42. The 42 second-shell atoms in the
    Mackay icosahedron are 20 face centers + 30 edge midpoints... but that's 50.
    
    Let me reconsider. The true Mackay icosahedron for n=2 has:
    - 1 center
    - 12 first shell (icosahedron vertices)
    - 42 second shell
    
    The 42 second-shell atoms: In the Mackay construction, the second shell
    consists of atoms at the vertices of a larger icosahedron (12) plus
    atoms at edge midpoints (30) = 42. But these are NOT the same as face centers.
    
    Actually, I believe the correct decomposition is:
    - 12 atoms along the 5-fold axes (same directions as shell 1, larger radius)
    - 30 atoms along the 2-fold axes (edge midpoints)
    Total second shell = 42.
    
    Let me verify: icosahedron has 6 2-fold axes (through edge midpoints) and
    10 3-fold axes (through face centers) and 6 5-fold axes (through vertices).
    Orbit sizes under full icosahedral group H (order 120):
    - 5-fold axis points: 12 (6 axes × 2 directions)
    - 3-fold axis points: 20 (10 axes × 2)
    - 2-fold axis points: 30 (15 axes × 2)
    
    So 42 = 12 + 30: 12 on 5-fold axes + 30 on 2-fold axes. This matches!
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, 1, -phi],
        [0, -1, phi],
        [0, -1, -phi],
        [1, phi, 0],
        [1, -phi, 0],
        [-1, phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1],
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges (pairs of vertices at minimum distance)
    edge_len = np.linalg.norm(verts_unit[0] - verts_unit[1])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - edge_len) < 1e-10:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for e in edges:
        em = (verts_unit[e[0]] + verts_unit[e[1]]) / 2.0
        em_norm = np.linalg.norm(em)
        edge_mids.append(em / em_norm)
    edge_mids = np.array(edge_mids)
    
    # Directions: center + 12 vertex dirs + 30 edge-mid dirs = 43 unique directions
    # But we need 55 atoms. The center is at origin (radius 0).
    # Shell 1: 12 atoms at radius r1 in vertex directions
    # Shell 2: 12 atoms at radius r2 in vertex directions + 30 atoms at radius r2 in edge-mid directions
    # Total: 1 + 12 + 12 + 30 = 55
    
    # Build direction array (55, 3)
    center_dir = np.zeros((1, 3))
    shell1_dirs = verts_unit  # (12, 3)
    shell2_vert_dirs = verts_unit  # (12, 3) - same directions as shell 1
    shell2_edge_dirs = edge_mids  # (30, 3)
    
    directions = np.vstack([center_dir, shell1_dirs, shell2_vert_dirs, shell2_edge_dirs])
    # (55, 3)
    return directions

def _construct_from_radii(radii):
    """Given 55 radii (one per atom direction), return (55,3) positions."""
    dirs = _get_mackay_directions()
    return dirs * radii[:, np.newaxis]

def _relax_lbfgs(pos, deadline):
    """Run L-BFGS relaxation with time guard."""
    try:
        result = minimize(
            _lj_energy, pos, jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 5000}
        )
        if result.x is not None:
            # Check min distance
            diff = result.x[:, np.newaxis, :] - result.x[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(r2, np.inf)
            min_d = np.sqrt(np.min(r2))
            if min_d >= 0.5:
                return result.x
    except Exception:
        pass
    return pos

def _polish(pos, deadline):
    """Tight polish with very small tolerances."""
    try:
        result = minimize(
            _lj_energy, pos, jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxfun': 10000}
        )
        if result.x is not None:
            diff = result.x[:, np.newaxis, :] - result.x[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(r2, np.inf)
            min_d = np.sqrt(np.min(r2))
            if min_d >= 0.5:
                return result.x
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    # Initial radii: center=0, shell1=r1, shell2=r2
    r1 = 2.0 ** (1.0 / 6.0)  # ~1.1225
    r2 = 2.0 ** (1.0 / 6.0) * 2.0  # ~2.245
    
    radii = np.zeros(55)
    radii[1:13] = r1    # shell 1: 12 atoms
    radii[13:25] = r2   # shell 2 vertex-type: 12 atoms
    radii[25:55] = r2   # shell 2 edge-type: 30 atoms
    
    pos = _construct_from_radii(radii)
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Step 1: L-BFGS in Cartesian space
    if time.time() < deadline - 30:
        pos = _relax_lbfgs(pos, deadline)
        e = _lj_energy(pos)
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
    
    # Step 2: Optimize radii with Nelder-Mead (reduces to 55D, more robust)
    if time.time() < deadline - 20:
        def radial_obj(r):
            p = _construct_from_radii(np.maximum(r, 0.1))
            return _lj_energy(p)
        
        try:
            nm_result = minimize(
                radial_obj, radii,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if nm_result.x is not None:
                r_opt = np.maximum(nm_result.x, 0.1)
                pos_nm = _construct_from_radii(r_opt)
                # Check min distance
                diff = pos_nm[:, np.newaxis, :] - pos_nm[np.newaxis, :, :]
                r2_arr = np.sum(diff ** 2, axis=2)
                np.fill_diagonal(r2_arr, np.inf)
                min_d = np.sqrt(np.min(r2_arr))
                if min_d >= 0.5:
                    e_nm = _lj_energy(pos_nm)
                    if e_nm < best_energy:
                        best_energy = e_nm
                        best_pos = pos_nm.copy()
                        # Polish in Cartesian
                        if time.time() < deadline - 10:
                            pos_nm = _relax_lbfgs(pos_nm, deadline)
                            e_pol = _lj_energy(pos_nm)
                            if e_pol < best_energy:
                                best_energy = e_pol
                                best_pos = pos_nm.copy()
        except Exception:
            pass
    
    # Step 3: Final polish
    if time.time() < deadline - 5:
        pos_final = _polish(best_pos, deadline)
        e_final = _lj_energy(pos_final)
        if e_final < best_energy:
            best_energy = e_final
            best_pos = pos_final.copy()
    
    return best_pos

# EVOLVE-BLOCK-END
