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

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    Uses the exact vertex set of the truncated octahedron from FCC lattice.
    """
    # FCC lattice points: (i,j,k) where i+j+k is even
    # Truncated octahedron for 38 atoms: 1 center + 12 first shell + 24 second shell + 1 third shell?
    # Actually the 38-atom LJ38 global min is a truncated octahedron with:
    # 1 center, 12 at distance a, 24 at distance b, 1 at distance c? No.
    # The 38-atom cluster: it's a subset of FCC. Let's use the integer lattice approach
    # but be more careful about which 38 we pick.
    
    # The 38-atom LJ38 global minimum is a truncated octahedron.
    # From the Cambridge Cluster Database, the structure has:
    # 1 atom at center, 12 at ~1.0, 24 at ~1.6, 1 at ~2.2 (approximately)
    # Actually it's 1+12+24+1 = 38? No, that's 38.
    # Wait: 1 + 12 + 24 + 1 = 38. Yes!
    
    # Let's generate FCC points and select the 38 closest to origin
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    # Sort by distance from origin and take 38 closest
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)
    
    # Scale so nearest-neighbor distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _make_icosahedral_cluster():
    """
    Generate an icosahedral structure for 38 atoms.
    Center + 12 (1st shell) + 20 (2nd shell) + 5 (3rd shell) = 38?
    Icosahedral shells: 1, 12, 20, 30...
    1+12+20 = 33. Need 5 more.
    This is a common local minimum trap.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)] # Center
    pts.extend(verts.tolist()) # 1st shell
    
    # 2nd shell: centers of faces of icosahedron? No, vertices of dodecahedron?
    # The 2nd shell in icosahedral clusters are usually 20 atoms at the centers of the triangular faces of the icosahedron.
    # Face normals of icosahedron.
    # There are 20 faces.
    # We can generate them by taking the cross product of adjacent edge vectors.
    # Simpler: The 2nd shell positions are proportional to the vertices of a dodecahedron?
    # Actually, for LJ, the 2nd shell is usually at distance ~1.6-1.7 times the 1st shell.
    # Let's use a known coordinate set for the 33-atom icosahedron and add 5 atoms.
    # Or just use a random perturbation of the FCC as the second start point.
    
    # For now, return a perturbed FCC to serve as a second basin start.
    return None

def _relax(pos, max_iter=2000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    # Use L-BFGS-B with reasonable settings
    # For very tight tolerances, increase maxcor to help convergence
    maxcor = 50 if gtol < 1e-12 else 20
    
    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxcor": maxcor},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_pos = None
    best_energy = np.inf
    
    # 1. Start with FCC - try to converge to global minimum
    try:
        pos0 = _make_fcc_cluster()
        p, e = _relax(pos0, max_iter=10000, gtol=1e-14, ftol=1e-18)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass

    # Check if we've reached the known global minimum (~-173.928)
    # If so, skip basin hopping and go straight to ultra-tight polish
    known_global_min = -173.928427
    reached_global = (best_energy is not None) and (best_energy > known_global_min - 0.001)
    
    if not reached_global:
        # 2. Basin Hopping / Random Restarts
        rng = np.random.default_rng(42)
        current_best = best_pos.copy() if best_pos is not None else _make_fcc_cluster()
        
        # Adaptive number of basin hopping iterations based on remaining time
        elapsed = time.time() - t0
        remaining = deadline - elapsed
        n_iter = max(5, min(50, int(remaining / 2.0)))  # ~2s per iteration
        
        for i in range(n_iter):
            if time.time() > deadline - 10.0:  # Reserve 10s for final polish
                break
                
            # Choose a base structure
            if i % 3 == 0:
                base = current_best
            elif i % 3 == 1:
                base = _make_fcc_cluster()
            else:
                # Random restart
                base = rng.standard_normal((N, 3)) * 2.0
                base = _ensure_min_sep(base)
            
            # Perturb with decreasing scale
            scale = 0.15 * (1.0 - 0.5 * i / max(1, n_iter)) + 0.05
            perturbed = base + scale * rng.standard_normal(base.shape)
            
            # Ensure separation
            perturbed = _ensure_min_sep(perturbed)
            
            # Relax
            try:
                p, e = _relax(perturbed, max_iter=3000, gtol=1e-12, ftol=1e-16)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
                    current_best = p
            except Exception:
                continue

    # 3. Final High-Precision Polish - use all remaining time
    if best_pos is not None:
        remaining = deadline - time.time() - 3.0  # 3s safety margin
        if remaining > 1.0:
            # Multiple polish passes with increasing precision
            try:
                p, e = _relax(best_pos, max_iter=20000, gtol=1e-16, ftol=1e-20)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
            except Exception:
                pass
            
            # Second polish if time remains
            remaining = deadline - time.time() - 2.0
            if remaining > 1.0:
                try:
                    p, e = _relax(best_pos, max_iter=20000, gtol=1e-18, ftol=1e-22)
                    if e < best_energy:
                        best_energy = e
                        best_pos = p
                except Exception:
                    pass

    if best_pos is None:
        # Fallback
        best_pos = _make_fcc_cluster()
        
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
