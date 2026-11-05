"""Dzugutov cluster global minima search using explicit Mackay icosahedron construction."""
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
SAFETY_MARGIN = 15.0

def _time_left():
    global _start_time
    if _start_time is None:
        return TIME_LIMIT
    return (TIME_LIMIT - SAFETY_MARGIN) - (time.time() - _start_time)

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Only consider upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    
    r = np.sqrt(r2_pair[valid])
    
    # Region 1: r < a
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c / (r1 - a)))
    
    # Region 2: a <= r < b
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        energy += np.sum(B * np.exp(d / (r2_ - b)))
    
    return float(energy)

def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    
    # Compute all pairwise vectors and distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad
    
    i_idx = iu[0][valid]
    j_idx = iu[1][valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[i_idx, j_idx]  # positions[i] - positions[j]
    
    # Compute dV/dr for each pair
    dV_dr = np.zeros_like(r)
    
    # Region 1: r < a
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        exp_term = np.exp(c / (r1 - a))
        dV_dr[mask1] = A * (-m * r1 ** (-m - 1)) * exp_term + A * (r1 ** (-m) - B) * exp_term * (-c / (r1 - a) ** 2)
    
    # Region 2: a <= r < b
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        dV_dr[mask2] = B * np.exp(d / (r2_ - b)) * (-d / (r2_ - b) ** 2)
    
    # Force factor: -dV_dr / r
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    
    # Accumulate gradients
    np.add.at(grad, i_idx, forces)
    np.add.at(grad, j_idx, -forces)
    
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
                diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                r2 = np.sum(diff ** 2, axis=2)
                iu = np.triu_indices(len(pos), k=1)
                min_r2 = np.min(r2[iu])
                if min_r2 < 0.25:  # 0.5^2
                    e = float('inf')  # Invalid
            scored_candidates.append((e, pos))
        except Exception:
            continue
    
    # Sort by energy
    scored_candidates.sort(key=lambda x: x[0])
    
    # Relax the best few
    max_relaxations = 6
    for e_init, pos in scored_candidates[:max_relaxations]:
        if _time_left() < 10:
            break
        if e_init == float('inf'):
            continue
        try:
            pos_rel, e_rel = _relax_structure(pos, max_iter=150)
            if e_rel < best_energy:
                best_energy = e_rel
                best_pos = pos_rel
        except Exception:
            continue

    # Final polish pass on the best structure
    if best_pos is not None and _time_left() > 5:
        try:
            pos_polished, e_polished = _polish_structure(best_pos, max_iter=300)
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
