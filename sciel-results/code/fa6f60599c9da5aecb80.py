"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

# EVOLVE-BLOCK-START

def _pairwise_sq_dists(points):
    """Compute squared pairwise distances as flat array of length N*(N-1)/2."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    sq_dists = _pairwise_sq_dists(points)
    d_min_sq = np.min(sq_dists)
    d_max_sq = np.max(sq_dists)
    if d_min_sq < 1e-24:
        return 1e10
    return d_max_sq / d_min_sq

def _soft_max_log_sq(x, beta=50.0):
    """
    Soft max of log(squared distances) using log-sum-exp.
    Returns log(sum(exp(beta * log_sq_dists))) / beta which approximates max(log_sq_dists).
    """
    # x is array of squared distances
    if np.any(x < 1e-30):
        return -np.inf
    log_x = np.log(x)
    m = np.max(log_x)
    if m < -300:
        return m
    shifted = beta * (log_x - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    log_sum = m + np.log(np.sum(np.exp(shifted))) / beta
    return log_sum

def _soft_min_log_sq(x, beta=50.0):
    """
    Soft min of log(squared distances) using log-sum-exp.
    Returns -log(sum(exp(-beta * log_sq_dists))) / beta which approximates min(log_sq_dists).
    """
    if np.any(x < 1e-30):
        return -np.inf
    log_x = np.log(x)
    m = np.min(log_x)
    if m < -300:
        return m
    shifted = -beta * (log_x - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = m - np.log(np.sum(np.exp(shifted))) / beta
    return log_sum

def _objective_log(params, beta=50.0):
    """
    Smooth objective in log-space: 2*(soft_max_log - soft_min_log)
    This is proportional to log(objective) where objective = (max/min)^2
    Minimizing this is equivalent to minimizing the true objective.
    """
    points = params.reshape(N, D)
    
    # Center to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = _pairwise_sq_dists(points)
    
    if np.any(sq_dists < 1e-20):
        return 1e10
    
    # Normalize scale: set mean squared distance to 1 for numerical stability
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-20:
        return 1e10
    
    # Scale factor: after normalization, distances are scaled by 1/sqrt(mean_sq)
    # log(d_normalized) = log(d) - 0.5*log(mean_sq)
    # So log_max_norm = log_max - 0.5*log(mean_sq), log_min_norm = log_min - 0.5*log(mean_sq)
    # The difference log_max_norm - log_min_norm = log_max - log_min (scale cancels!)
    # So we can compute directly without re-normalizing
    
    log_max = _soft_max_log_sq(sq_dists, beta)
    log_min = _soft_min_log_sq(sq_dists, beta)
    
    if log_max < -300 or log_min < -300:
        return 1e10
    
    # The objective is (max/min)^2 = exp(2*(log_max - log_min))
    # We minimize 2*(log_max - log_min) which is log(objective)
    obj = 2.0 * (log_max - log_min)
    
    if obj > 100:  # Cap to avoid overflow issues
        return 1e10
    
    return obj

def _objective_true(params):
    """True objective: (max / min)^2."""
    points = params.reshape(N, D)
    return _score(points)

def _tetrahedron_initial():
    """
    Generate initial configuration based on regular tetrahedron vertices
    plus midpoints of edges, then add additional points.
    4 vertices + 6 midpoints + 4 face centers = 14 points.
    """
    a = 1.0
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ], dtype=float)
    
    verts -= np.mean(verts, axis=0)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    verts /= edge_len
    
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    midpoints = []
    for i, j in edges:
        midpoints.append((verts[i] + verts[j]) / 2)
    midpoints = np.array(midpoints)
    
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    points = np.vstack([verts, midpoints, face_centers])
    points -= np.mean(points, axis=0)
    
    return points

def _cuboctahedron_initial():
    """
    Cuboctahedron has 12 vertices. Add 2 more points (e.g., at ±z poles) for 14 total.
    Cuboctahedron vertices: all permutations of (±1, ±1, 0)
    """
    # 12 vertices of cuboctahedron: permutations of (±1, ±1, 0)
    verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            verts.append([x, y, 0])
            verts.append([x, 0, y])
            verts.append([0, x, y])
    verts = np.array(verts, dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Add 2 poles
    p1 = np.array([0, 0, 1.5])
    p2 = np.array([0, 0, -1.5])
    
    points = np.vstack([verts, p1, p2])
    points -= np.mean(points, axis=0)
    
    return points

def _octahedron_plus8_initial(rng):
    """
    Octahedron has 6 vertices. Add 8 more points at face centers.
    6 + 8 = 14 points.
    """
    # 6 vertices of octahedron
    verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 8 face centers (each face is a triangle with 3 vertices from the 6)
    # Faces of octahedron: 8 triangular faces
    faces = [
        (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5),
        (2, 3, 4), (2, 3, 5), (4, 5, 0), (4, 5, 1)
    ]
    # Actually let me use the correct octahedron faces
    # Vertices: +x, -x, +y, -y, +z, -z
    # Faces: (+x,+y,+z), (+x,+y,-z), (+x,-y,+z), (+x,-y,-z),
    #         (-x,+y,+z), (-x,+y,-z), (-x,-y,+z), (-x,-y,-z)
    face_centers = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            for sz in [1, -1]:
                fc = np.array([sx, sy, sz]) / np.sqrt(3)
                face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    points = np.vstack([verts, face_centers])
    points -= np.mean(points, axis=0)
    
    # Random rotation
    rot = Rotation.from_quat(rng.normal(size=4))
    points = rot.apply(points)
    
    return points

def _icosahedron_plus2_initial(rng):
    """Icosahedron (12 vertices) + 2 additional points."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points at opposite poles or random positions
    p1 = rng.normal(size=D)
    p1 = p1 / np.linalg.norm(p1)
    p2 = -p1  # Opposite point for symmetry
    
    full_pts = np.vstack([ico, p1, p2])
    full_pts -= np.mean(full_pts, axis=0)
    
    # Random rotation
    rot = Rotation.from_quat(rng.normal(size=4))
    full_pts = rot.apply(full_pts)
    
    return full_pts

def _random_spherical_initial(rng):
    """Random points on sphere."""
    pts = rng.normal(size=(N, D))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts -= np.mean(pts, axis=0)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _fibonacci_sphere_initial(rng):
    """Fibonacci sphere distribution."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(N):
        z = 1 - 2 * (i + 0.5) / N
        radius = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    pts = np.array(points)
    pts -= np.mean(pts, axis=0)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _two_rings_initial(rng):
    """Two rings of 7 points each at different z-heights."""
    z_offset = rng.uniform(0.3, 0.6)
    ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi)
    ring2_angles = ring1_angles + np.pi/7
    
    ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.full(7, z_offset)])
    ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, -z_offset)])
    pts = np.vstack([ring1, ring2])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts -= np.mean(pts, axis=0)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _polish(points, deadline, beta=200.0):
    """Final high-precision polish pass with multiple optimizer stages."""
    if time.time() > deadline - 5.0:
        return points
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Stage 1: L-BFGS-B with moderate beta
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective_log,
                best_points.flatten(),
                args=(100.0,),
                method='L-BFGS-B',
                options={
                    'maxiter': 3000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            
            if result.fun < 50:
                candidate = result.x.reshape(N, D)
                candidate -= np.mean(candidate, axis=0)
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B with high beta
    if time.time() < deadline - 8.0:
        try:
            result = minimize(
                _objective_log,
                best_points.flatten(),
                args=(300.0,),
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxls': 50
                }
            )
            
            if result.fun < 50:
                candidate = result.x.reshape(N, D)
                candidate -= np.mean(candidate, axis=0)
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead with very tight tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective_log,
                best_points.flatten(),
                args=(500.0,),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result.fun < 50:
                candidate = result.x.reshape(N, D)
                candidate -= np.mean(candidate, axis=0)
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    # Stage 4: Final L-BFGS-B polish
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                _objective_log,
                best_points.flatten(),
                args=(800.0,),
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-14,
                    'maxls': 50
                }
            )
            
            if result.fun < 50:
                candidate = result.x.reshape(N, D)
                candidate -= np.mean(candidate, axis=0)
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    return best_points

def construct():
    """Construct 14 points in 3D to minimize squared max/min distance ratio."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = np.inf
    
    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        sq_dists = _pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-18:
            return
        score = _score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Tetrahedron-based (primary structural guess)
    candidates.append(_tetrahedron_initial())
    
    # Perturbed tetrahedron
    for trial in range(3):
        p = _tetrahedron_initial()
        p = p + rng.normal(size=(N, D)) * 0.05
        p -= np.mean(p, axis=0)
        candidates.append(p)
    
    # 2. Icosahedron + 2
    for trial in range(3):
        candidates.append(_icosahedron_plus2_initial(rng))
    
    # 3. Cuboctahedron + 2 poles
    candidates.append(_cuboctahedron_initial())
    
    # 4. Octahedron + 8 face centers
    for trial in range(2):
        candidates.append(_octahedron_plus8_initial(rng))
    
    # 5. Fibonacci sphere
    for trial in range(2):
        candidates.append(_fibonacci_sphere_initial(rng))
    
    # 6. Two rings
    for trial in range(3):
        candidates.append(_two_rings_initial(rng))
    
    # 7. Random spherical
    for trial in range(3):
        candidates.append(_random_spherical_initial(rng))
    
    # Evaluate all initial candidates
    for c in candidates:
        update_best(c)
    
    # Progressive beta annealing optimization
    beta_schedule = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0]
    
    n_candidates = len(candidates)
    
    # Phase 1: Coarse optimization with low beta
    for i in range(n_candidates):
        if time.time() > deadline - 35.0:
            break
        x0 = candidates[i].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[0],),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[i] = pts
        except Exception:
            pass
    
    # Phase 2: Medium optimization
    for i in range(n_candidates):
        if time.time() > deadline - 25.0:
            break
        x0 = candidates[i].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[1],),
                method='L-BFGS-B',
                options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[i] = pts
        except Exception:
            pass
    
    # Phase 3: Fine optimization
    for i in range(n_candidates):
        if time.time() > deadline - 15.0:
            break
        x0 = candidates[i].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[2],),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[i] = pts
        except Exception:
            pass
    
    # Phase 4: Very fine optimization on best candidates
    # Sort candidates by score and optimize top ones
    scored = []
    for i in range(n_candidates):
        pts = candidates[i]
        s = _score(pts)
        scored.append((s, i))
    scored.sort()
    
    top_k = min(6, len(scored))
    for rank in range(top_k):
        if time.time() > deadline - 10.0:
            break
        _, idx = scored[rank]
        x0 = candidates[idx].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[3],),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[idx] = pts
        except Exception:
            pass
    
    # Phase 5: Ultra-fine on top 3
    top_3 = min(3, len(scored))
    for rank in range(top_3):
        if time.time() > deadline - 6.0:
            break
        _, idx = scored[rank]
        x0 = candidates[idx].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[4],),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[idx] = pts
        except Exception:
            pass
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, deadline, beta=500.0)
    
    # Ensure we have a valid result
    if best_points is None:
        best_points = _tetrahedron_initial()
    
    # Final validation
    sq_dists = _pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-16:
        best_points = _tetrahedron_initial()
    
    return best_points

# EVOLVE-BLOCK-END
