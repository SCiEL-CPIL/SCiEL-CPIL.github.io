"""LJ38 global minimum search: Explicit FCC cluster construction with
Basin Hopping optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE_S = 120.0
MARGIN = 15.0
R0 = 2.0 ** (1.0 / 6.0)

def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _relax(pos, maxiter=20000, deadline=None, gtol=1e-10, ftol=1e-16):
    """L-BFGS-B relaxation with optional deadline and precision control."""
    pos = np.array(pos, dtype=float).copy()

    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
                                'maxls': 50})
        return res.x.reshape(-1, 3), res.fun
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]

def _make_fcc_38():
    """Generate the 38 nearest neighbors in an FCC lattice, scaled to LJ units."""
    # FCC lattice points: (i, j, k) where i+j+k is even
    # We want the 38 points closest to the origin.
    points = []
    max_coord = 10
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    points.append((r2, i, j, k))
    
    points.sort(key=lambda x: x[0])
    
    # Take the first 38
    selected = points[:N]
    
    # Scale: The nearest neighbor distance in this integer lattice is sqrt(2).
    # We want the nearest neighbor distance to be R0.
    # So scale factor is R0 / sqrt(2).
    scale = R0 / np.sqrt(2.0)
    
    coords = np.array([[i, j, k] for (_, i, j, k) in selected], dtype=float)
    coords *= scale
    
    return coords

def _make_icosahedral_38():
    """Construct an icosahedral arrangement for 38 atoms."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * R0

    pos_list = [np.zeros(3)]
    pos_list.extend(list(verts))

    # Add 25 more atoms on a second shell (approximate)
    n_outer = N - len(pos_list)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        radius = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden
        v = 2.0 * R0 * np.array([radius * np.cos(theta),
                                  radius * np.sin(theta),
                                  z])
        pos_list.append(v)

    pos = np.array(pos_list[:N])
    pos -= pos.mean(axis=0)
    return pos

def _basin_hop(start_pos, deadline, rng, n_hops=50, step_size=0.5):
    """Perform Basin Hopping optimization."""
    current_pos, current_e = _relax(start_pos, deadline=deadline)
    best_pos, best_e = current_pos.copy(), current_e
    
    for i in range(n_hops):
        if time.time() > deadline:
            break
        
        # Perturb
        noise = rng.randn(N, 3) * step_size
        new_pos = current_pos + noise
        
        # Relax
        new_pos, new_e = _relax(new_pos, deadline=deadline)
        
        # Metropolis criterion
        if new_e < current_e:
            current_pos, current_e = new_pos, new_e
        else:
            # Accept with probability exp(-(new_e - current_e) / T)
            # T is effectively 1.0 in reduced units for LJ
            if np.random.rand() < np.exp(-(new_e - current_e)):
                current_pos, current_e = new_pos, new_e
        
        if current_e < best_e:
            best_pos, best_e = current_pos.copy(), current_e
            
        # Reduce step size slightly
        step_size *= 0.95
        
    return best_pos, best_e

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN
    
    best_pos = None
    best_e = np.inf
    
    rng = np.random.RandomState(42)
    
    # 1. Start with FCC Truncated Octahedron
    try:
        pos_fcc = _make_fcc_38()
        pos_b, e_b = _basin_hop(pos_fcc, deadline, rng, n_hops=30, step_size=0.3)
        if e_b < best_e:
            best_e = e_b
            best_pos = pos_b.copy()
    except Exception:
        pass
    
    # 2. Start with Icosahedral
    if time.time() < deadline - 10.0:
        try:
            pos_ico = _make_icosahedral_38()
            pos_b, e_b = _basin_hop(pos_ico, deadline, rng, n_hops=30, step_size=0.3)
            if e_b < best_e:
                best_e = e_b
                best_pos = pos_b.copy()
        except Exception:
            pass
            
    # 3. Random starts if time allows
    if time.time() < deadline - 10.0 and best_pos is not None:
        for _ in range(3):
            if time.time() > deadline:
                break
            pos_rand = rng.randn(N, 3) * 3.0
            pos_b, e_b = _basin_hop(pos_rand, deadline, rng, n_hops=10, step_size=0.5)
            if e_b < best_e:
                best_e = e_b
                best_pos = pos_b.copy()

    if best_pos is None:
        # Fallback
        return _make_fcc_38()

    # Final polish
    if time.time() < deadline - 2.0:
        pos_r, e_r = _relax(best_pos, deadline=deadline, gtol=1e-12)
        if e_r < best_e:
            best_pos = pos_r
            best_e = e_r

    # Ensure min distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
