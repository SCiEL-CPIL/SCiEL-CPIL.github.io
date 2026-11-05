"""Dzugutov cluster global minima search using continuous annealing on softness parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

# Global time tracker
_start_time = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 20.0

def _time_left():
    global _start_time
    if _start_time is None:
        return TIME_LIMIT
    return (TIME_LIMIT - SAFETY_MARGIN) - (time.time() - _start_time)

# Pre-computed pair indices cache
_pair_cache = {}

def _get_pair_indices(n):
    """Get or create pre-computed pair indices for n atoms."""
    if n not in _pair_cache:
        iu = np.triu_indices(n, k=1)
        _pair_cache[n] = iu
    return _pair_cache[n]

def _dzugutov_energy(positions, c_val=None, d_val=None):
    """Compute Dzugutov potential energy with optional annealed softness parameters."""
    if c_val is None:
        c_val = c
    if d_val is None:
        d_val = d
    n = len(positions)
    if n < 2:
        return 0.0
    
    iu = _get_pair_indices(n)
    diff = positions[iu[0]] - positions[iu[1]]
    r2_pair = np.sum(diff ** 2, axis=1)
    
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    
    r = np.sqrt(r2_pair[valid])
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c_val / (r1 - a)))
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        energy += np.sum(B * np.exp(d_val / (r2_ - b)))
    
    return float(energy)

def _dzugutov_gradient(positions, c_val=None, d_val=None):
    """Compute gradient with optional annealed softness parameters."""
    if c_val is None:
        c_val = c
    if d_val is None:
        d_val = d
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    
    iu = _get_pair_indices(n)
    i_idx = iu[0]
    j_idx = iu[1]
    diff = positions[i_idx] - positions[j_idx]
    r2_pair = np.sum(diff ** 2, axis=1)
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad
    
    i_idx_v = i_idx[valid]
    j_idx_v = j_idx[valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[valid]
    
    dV_dr = np.zeros_like(r)
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        exp_term = np.exp(c_val / (r1 - a))
        dV_dr[mask1] = A * (-m * r1 ** (-m - 1)) * exp_term + A * (r1 ** (-m) - B) * exp_term * (-c_val / (r1 - a) ** 2)
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        dV_dr[mask2] = B * np.exp(d_val / (r2_ - b)) * (-d_val / (r2_ - b) ** 2)
    
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    
    np.add.at(grad, i_idx_v, forces)
    np.add.at(grad, j_idx_v, -forces)
    
    return grad

def _get_icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _get_icosahedron_edges(verts):
    """Get the 30 edge midpoint directions of an icosahedron given its vertices."""
    n_verts = len(verts)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    # Find edges: pairs of vertices separated by the minimum distance
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
    
    edge_dirs = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edge_dirs.append(mid)
    
    return np.array(edge_dirs)

def _get_icosahedron_face_centers(verts):
    """Get the 20 face center directions of an icosahedron given its 12 vertices."""
    n_verts = len(verts)
    verts_unit = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
    
    # Find edges first
    min_dist = float('inf')
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist - 1e-6:
                min_dist = d
    
    # Build adjacency
    adj = {i: [] for i in range(n_verts)}
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - min_dist) < 1e-6:
                adj[i].append(j)
                adj[j].append(i)
    
    # Find faces: triples of mutually connected vertices
    face_dirs = []
    seen = set()
    for i in range(n_verts):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k <= j or k == i:
                    continue
                if k in adj[i]:
                    # Check if this face has been seen
                    face_key = tuple(sorted([i, j, k]))
                    if face_key not in seen:
                        seen.add(face_key)
                        # Compute face center as average of vertices, then normalize
                        center = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_dirs.append(center)
    
    return np.array(face_dirs)

def _build_mackay_55(r1, r2v, r2e):
    """
    Build Mackay 55 icosahedron.
    r1: radius of shell 1 (12 atoms)
    r2v: radius of shell 2 vertices (12 atoms, same directions as shell 1)
    r2e: radius of shell 2 edges (30 atoms, edge midpoint directions of shell 1)
    """
    points = [np.zeros(3)]  # Center
    
    # Shell 1: 12 vertices
    v1 = _get_icosahedron_vertices(r1)
    points.extend([v for v in v1])
    
    # Shell 2: 12 vertices + 30 edge centers
    v1_unit = v1 / r1
    v2 = v1_unit * r2v
    points.extend([v for v in v2])
    
    # Edge directions from unit icosahedron
    e_dirs = _get_icosahedron_edges(_get_icosahedron_vertices(1.0))
    e2 = e_dirs * r2e
    points.extend([e for e in e2])
    
    return np.array(points)

def _build_mackay_55_with_faces(r1, r2v, r2e):
    """
    Build Mackay 55 icosahedron and return face center directions for adding extra atoms.
    """
    pos = _build_mackay_55(r1, r2v, r2e)
    
    # Get face centers from the outer shell vertices (shell 2 vertices)
    v1 = _get_icosahedron_vertices(1.0)
    face_dirs = _get_icosahedron_face_centers(v1)
    
    return pos, face_dirs

def _relax_structure(pos, max_iter=200, ftol=1e-8, gtol=1e-6):
    """Relax a structure using L-BFGS-B with gradient."""
    n = len(pos)
    x0 = pos.flatten()
    
    def fun(x):
        p = x.reshape(n, 3)
        return _dzugutov_energy(p)
        
    def jac(x):
        p = x.reshape(n, 3)
        return _dzugutov_gradient(p).flatten()

    res = minimize(fun, x0, jac=jac, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
    return res.x.reshape(n, 3), res.fun

def _polish_structure(pos, max_iter=500):
    """Final high-precision polish pass with tighter tolerances."""
    n = len(pos)
    x0 = pos.flatten()
    
    def fun(x):
        p = x.reshape(n, 3)
        return _dzugutov_energy(p)
        
    def jac(x):
        p = x.reshape(n, 3)
        return _dzugutov_gradient(p).flatten()

    res = minimize(fun, x0, jac=jac, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
    return res.x.reshape(n, 3), res.fun

def _anneal_relax(pos, n_steps=60, max_iter_per_step=40):
    """
    Continuous annealing: gradually reduce softness parameters c and d from
    large values (smooth landscape) to their physical values (true potential).
    This allows a single trajectory to flow into the global basin.
    """
    n = len(pos)
    x = pos.flatten()
    
    # Annealing schedule: c goes from c*8 to c, d goes from d*8 to d
    # Use quadratic annealing for smoother early transition
    c_start = c * 8.0
    d_start = d * 8.0
    
    best_energy = float('inf')
    best_x = x.copy()
    
    for step in range(n_steps):
        # Quadratic annealing: t^2 for slower initial cooling
        t = step / max(n_steps - 1, 1)
        alpha = t * t
        c_cur = c_start + (c - c_start) * alpha
        d_cur = d_start + (d - d_start) * alpha
        
        def fun(x, c_cur=c_cur, d_cur=d_cur):
            p = x.reshape(n, 3)
            return _dzugutov_energy(p, c_val=c_cur, d_val=d_cur)
            
        def jac(x, c_cur=c_cur, d_cur=d_cur):
            p = x.reshape(n, 3)
            return _dzugutov_gradient(p, c_val=c_cur, d_val=d_cur).flatten()
        
        try:
            res = minimize(fun, x, jac=jac, method='L-BFGS-B',
                          options={'maxiter': max_iter_per_step, 'ftol': 1e-7, 'gtol': 1e-5})
            x = res.x
            # Track best with true potential
            e_true = _dzugutov_energy(x.reshape(n, 3))
            if e_true < best_energy:
                best_energy = e_true
                best_x = x.copy()
        except Exception:
            pass
        
        if _time_left() < 10:
            break
    
    # Final polish with true potential
    try:
        res = minimize(lambda x: _dzugutov_energy(x.reshape(n, 3)),
                      best_x,
                      jac=lambda x: _dzugutov_gradient(x.reshape(n, 3)).flatten(),
                      method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        e_final = _dzugutov_energy(res.x.reshape(n, 3))
        if e_final < best_energy:
            best_energy = e_final
            best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x.reshape(n, 3), best_energy

def _perturb_and_relax(pos, n_perturb=3, sigma=0.15, max_iter=150):
    """
    Apply small random perturbations to escape local minima, then relax.
    Returns the best structure found.
    """
    n = len(pos)
    best_energy = _dzugutov_energy(pos)
    best_x = pos.flatten().copy()
    
    rng = np.random.default_rng(42)
    
    for p in range(n_perturb):
        if _time_left() < 15:
            break
        
        # Apply Gaussian perturbation
        x_pert = best_x + rng.normal(0, sigma, size=(n, 3)).flatten()
        
        try:
            res = minimize(lambda x: _dzugutov_energy(x.reshape(n, 3)),
                          x_pert,
                          jac=lambda x: _dzugutov_gradient(x.reshape(n, 3)).flatten(),
                          method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            e = _dzugutov_energy(res.x.reshape(n, 3))
            if e < best_energy:
                best_energy = e
                best_x = res.x.copy()
        except Exception:
            continue
    
    return best_x.reshape(n, 3), best_energy

def construct(instance):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    n = instance["n"]
    potential = instance.get("potential", "dzugutov")
    
    # Check time
    if _time_left() < 5:
        # Emergency: return a simple cluster
        pos = np.random.randn(n, 3) * 0.5
        return pos

    best_energy = float('inf')
    best_pos = None
    
    # Strategy: 
    # 1. If n <= 55, use subsets of Mackay 55.
    # 2. If n > 55, use Mackay 55 + extra atoms at face centers.
    
    # Define radii guesses
    # Optimal bond length for Dzugutov is approx 1.5-1.8?
    # Let's try a few sets of radii.
    # Standard Mackay geometry ratios: 
    # If r1 = 1.0, r2v = sqrt(1+phi) ~ 1.7? r2e = sqrt(1+phi/2)?
    # Let's just use a range.
    radii_sets = [
        (1.5, 2.5, 2.5),
        (1.6, 2.6, 2.6),
        (1.7, 2.8, 2.8),
        (1.5, 2.4, 2.4),
        (1.8, 3.0, 3.0),
        (1.55, 2.55, 2.55),
        (1.65, 2.65, 2.65),
    ]
    
    # Generate candidates
    candidates = []
    
    if n <= 55:
        # Generate Mackay 55 for each radius set
        mackay_55s = []
        for r1, r2v, r2e in radii_sets:
            if _time_left() < 15:
                break
            m55 = _build_mackay_55(r1, r2v, r2e)
            # Relax the 55-atom cluster first to get optimal geometry
            try:
                pos55, e55 = _relax_structure(m55, max_iter=100)
                mackay_55s.append((pos55, e55))
            except Exception:
                pass
        
        # For each relaxed 55-atom cluster, create n-atom subsets
        for pos55, e55 in mackay_55s:
            if n == 55:
                candidates.append(pos55)
            else:
                # Create subset by removing outermost atoms
                # Calculate distances from centroid
                com = np.mean(pos55, axis=0)
                dists = np.linalg.norm(pos55 - com, axis=1)
                # Indices of atoms, sorted by distance (largest first)
                idx_sorted = np.argsort(dists)[::-1]
                # Remove (55-n) atoms
                remove_idx = idx_sorted[:55-n]
                keep_mask = np.ones(55, dtype=bool)
                keep_mask[remove_idx] = False
                pos_subset = pos55[keep_mask]
                candidates.append(pos_subset)
                
    else:
        # n > 55
        # Start with Mackay 55 and add extra atoms at face centers
        for r1, r2v, r2e in radii_sets[:5]:
            if _time_left() < 15:
                break
            m55, face_dirs = _build_mackay_55_with_faces(r1, r2v, r2e)
            try:
                pos55, e55 = _relax_structure(m55, max_iter=100)
                # Add extra atoms at face centers
                extra_n = n - 55
                if extra_n > 0:
                    # Place extra atoms at face center directions with appropriate radius
                    max_r = np.max(np.linalg.norm(pos55, axis=1))
                    # Use face center directions, cycling if needed
                    face_pos = []
                    for i in range(extra_n):
                        direction = face_dirs[i % len(face_dirs)]
                        # Place at radius slightly larger than max shell
                        radius = max_r + 0.3 + 0.1 * (i // len(face_dirs))
                        face_pos.append(direction * radius)
                    extra_pos = np.array(face_pos)
                    pos_n = np.vstack([pos55, extra_pos])
                    candidates.append(pos_n)
            except Exception:
                pass

    # Fallback if no candidates generated
    if not candidates:
        pos = np.random.randn(n, 3) * 1.0
        candidates.append(pos)

    # Relax candidates
    # Prioritize candidates with lower initial energy
    scored_candidates = []
    for pos in candidates:
        try:
            e = _dzugutov_energy(pos)
            # Check minimum distance constraint
            if len(pos) > 1:
                iu = _get_pair_indices(len(pos))
                diff = pos[iu[0]] - pos[iu[1]]
                r2 = np.sum(diff ** 2, axis=1)
                min_r2 = np.min(r2)
                if min_r2 < 0.25:  # 0.5^2
                    e = float('inf')  # Invalid
            scored_candidates.append((e, pos))
        except Exception:
            continue
    
    # Sort by energy
    scored_candidates.sort(key=lambda x: x[0])
    
    # Use annealing-based relaxation for the best candidates
    max_relaxations = 3
    for e_init, pos in scored_candidates[:max_relaxations]:
        if _time_left() < 20:
            break
        if e_init == float('inf'):
            continue
        try:
            # Annealing relaxation: smoother landscape helps find global basin
            pos_annealed, e_annealed = _anneal_relax(pos, n_steps=50, max_iter_per_step=35)
            if e_annealed < best_energy:
                best_energy = e_annealed
                best_pos = pos_annealed
        except Exception:
            continue
    
    # Perturbation restarts on the best structure found so far
    if best_pos is not None and _time_left() > 25:
        try:
            pos_pert, e_pert = _perturb_and_relax(best_pos, n_perturb=3, sigma=0.12, max_iter=200)
            if e_pert < best_energy:
                best_energy = e_pert
                best_pos = pos_pert
        except Exception:
            pass
    
    # Also try standard relaxation on the best candidate as backup
    if scored_candidates and scored_candidates[0][0] != float('inf'):
        if _time_left() > 20:
            try:
                pos_std, e_std = _relax_structure(scored_candidates[0][1], max_iter=300)
                if e_std < best_energy:
                    best_energy = e_std
                    best_pos = pos_std
            except Exception:
                pass

    # Final polish pass on the best structure
    if best_pos is not None and _time_left() > 10:
        try:
            pos_polished, e_polished = _polish_structure(best_pos, max_iter=500)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pos = pos_polished
        except Exception:
            pass

    if best_pos is None:
        # Return the best initial candidate
        if scored_candidates:
            best_pos = scored_candidates[0][1]
        else:
            best_pos = np.random.randn(n, 3) * 0.5
            
    return best_pos

# EVOLVE-BLOCK-END
