"""Logarithmic energy minimization for N points on the unit sphere.
Cluster-based approach: 10 clusters of 10 points with fixed intra-cluster
positions (icosahedral vertices), optimizing only inter-cluster SO(3) rotations.
This reduces dimensionality from 300 to 30, enabling more robust global search."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 100
N_CLUSTERS = 10
PER_CLUSTER = 10

# Icosahedral vertices (12 vertices of regular icosahedron)
# We need 10 points per cluster, so we'll use a subset or a different design
# Actually, for 10 points, let's use a decagonal arrangement or just take
# 10 of the 12 icosahedral vertices
_PHI = (1 + math.sqrt(5)) / 2

def _get_cluster_template():
    """Return 10 unit vectors forming a well-distributed set on the sphere.
    We use a modified icosahedral subset: take all 12 icosahedral vertices
    and remove 2 that are closest to each other, or use a different approach.
    
    Better: use 10 points from a Fibonacci spiral on the sphere, which gives
    good distribution."""
    i = np.arange(PER_CLUSTER)
    z = 1.0 - 2.0 * (i + 0.5) / PER_CLUSTER
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


_CLUSTER_TEMPLATE = _get_cluster_template()


def _exp_to_rot(vec):
    """Convert exponential coordinates (axis-angle) to 3x3 rotation matrix.
    vec is a 3-vector representing rotation axis * angle."""
    angle = np.linalg.norm(vec)
    if angle < 1e-15:
        return np.eye(3)
    axis = vec / angle
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rot_to_exp(R):
    """Convert 3x3 rotation matrix to exponential coordinates (axis-angle)."""
    # Extract rotation angle
    tr = np.trace(R)
    cos_angle = (tr - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    if angle < 1e-15:
        return np.zeros(3)
    elif angle > np.pi - 1e-15:
        # Handle the case where angle is close to pi
        # Use eigenvector of R corresponding to eigenvalue 1
        w, v = np.linalg.eig(R)
        idx = np.argmin(np.abs(w - 1.0))
        axis = np.real(v[:, idx])
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    else:
        # Standard case: extract axis from antisymmetric part
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        axis = axis / (2.0 * np.sin(angle))
        return axis * angle


def _clusters_to_points(rot_params):
    """Convert rotation parameters (30 values = 10 clusters × 3 exp coords)
    to 100 points on the unit sphere."""
    n_clusters = len(rot_params) // 3
    points = np.zeros((N, 3))
    for c in range(n_clusters):
        exp_vec = rot_params[c * 3:(c + 1) * 3]
        R = _exp_to_rot(exp_vec)
        cluster_pts = _CLUSTER_TEMPLATE @ R.T
        points[c * PER_CLUSTER:(c + 1) * PER_CLUSTER] = cluster_pts
    return points


def _points_to_clusters(pts):
    """Convert 100 points to 30 rotation parameters (for each cluster,
    find the best-fitting rotation to the template)."""
    rot_params = np.zeros(N_CLUSTERS * 3)
    for c in range(N_CLUSTERS):
        cluster_pts = pts[c * PER_CLUSTER:(c + 1) * PER_CLUSTER]
        # Find rotation R that minimizes ||cluster_pts - template @ R.T||^2
        # This is a Procrustes problem
        # SVD of (template.T @ cluster_pts)
        M = _CLUSTER_TEMPLATE.T @ cluster_pts
        U, S, Vt = np.linalg.svd(M)
        R = Vt.T @ U.T
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        rot_params[c * 3:(c + 1) * 3] = _rot_to_exp(R)
    return rot_params


def _energy_from_params(rot_params):
    """Compute energy from rotation parameters."""
    pts = _clusters_to_points(rot_params)
    n = pts.shape[0]
    sq = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.clip(dist_sq, 1e-18, None)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad_from_params(rot_params):
    """Compute energy and gradient with respect to rotation parameters.
    Uses finite differences for the gradient since analytical derivation
    through SO(3) is complex."""
    eps = 1e-7
    n_params = len(rot_params)
    grad = np.zeros(n_params)
    
    base_energy = _energy_from_params(rot_params)
    
    for i in range(n_params):
        rot_plus = rot_params.copy()
        rot_plus[i] += eps
        e_plus = _energy_from_params(rot_plus)
        
        rot_minus = rot_params.copy()
        rot_minus[i] -= eps
        e_minus = _energy_from_params(rot_minus)
        
        grad[i] = (e_plus - e_minus) / (2 * eps)
    
    return base_energy, grad


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    n = pts.shape[0]
    sq = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.clip(dist_sq, 1e-18, None)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _optimize_clusters(rot_params, max_iter=5000, ftol=1e-12, gtol=1e-8):
    """Optimize rotation parameters using L-BFGS-B."""
    def objective_and_grad(flat):
        e, g = _energy_grad_from_params(flat)
        return e, g
    
    try:
        result = minimize(
            objective_and_grad,
            rot_params.copy(),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 50,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
            }
        )
        return result.x
    except Exception:
        return rot_params


def _optimize_full(pts, max_iter=5000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from a starting point (full 300-dim)."""
    n = pts.shape[0]
    
    def _energy_grad(pts):
        n = pts.shape[0]
        sq = np.einsum('ij,ij->i', pts, pts)
        dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.clip(dist_sq, 1e-18, None)
        dist = np.sqrt(dist_sq)
        
        iu = np.triu_indices(n, k=1)
        energy = float(-np.sum(np.log(dist[iu])))
        
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        row_sums = W.sum(axis=1, keepdims=True)
        grad_E = row_sums * pts - W @ pts
        dot = np.sum(grad_E * pts, axis=1, keepdims=True)
        grad = grad_E - dot * pts
        
        return energy, grad
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 20,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Cluster-based optimization (30-dim)
    # Generate diverse initial rotation parameters
    initial_rotations = []
    
    # Identity rotations (all clusters aligned)
    initial_rotations.append(np.zeros(N_CLUSTERS * 3))
    
    # Random rotations
    for s in range(20):
        rng_s = np.random.default_rng(100 + s)
        rot_params = rng_s.standard_normal(N_CLUSTERS * 3) * 0.5
        initial_rotations.append(rot_params)
    
    # Rotations with some structure
    for s in range(10):
        rng_s = np.random.default_rng(200 + s)
        rot_params = np.zeros(N_CLUSTERS * 3)
        for c in range(N_CLUSTERS):
            # Small random rotations
            rot_params[c * 3:(c + 1) * 3] = rng_s.standard_normal(3) * (0.1 + 0.1 * c)
        initial_rotations.append(rot_params)
    
    # Optimize each cluster initialization
    for init_rot in initial_rotations:
        if time.time() > deadline - 60.0:
            break
        
        optimized_rot = _optimize_clusters(init_rot, max_iter=3000, ftol=1e-10, gtol=1e-6)
        pts = _clusters_to_points(optimized_rot)
        e = _precise_energy(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 2: Full 300-dim optimization from best cluster result
    if best_pts is not None and time.time() < deadline - 20.0:
        refined = _optimize_full(best_pts, max_iter=5000, ftol=1e-14, gtol=1e-10)
        e = _precise_energy(refined)
        if e < best_energy:
            best_energy = e
            best_pts = refined
    
    # Phase 3: Also try full optimization from standard initializations
    # Fibonacci spiral
    fib_pts = _fibonacci_spiral(N, offset=0.5)
    if time.time() < deadline - 15.0:
        optimized_fib = _optimize_full(fib_pts, max_iter=3000, ftol=1e-12, gtol=1e-8)
        e = _precise_energy(optimized_fib)
        if e < best_energy:
            best_energy = e
            best_pts = optimized_fib
    
    # Random points
    if time.time() < deadline - 10.0:
        for s in range(3):
            rand_pts = _random_sphere(np.random.default_rng(500 + s))
            optimized_rand = _optimize_full(rand_pts, max_iter=2000, ftol=1e-12, gtol=1e-8)
            e = _precise_energy(optimized_rand)
            if e < best_energy:
                best_energy = e
                best_pts = optimized_rand
            if time.time() > deadline - 5.0:
                break
    
    # Final polish with tighter tolerance
    if best_pts is not None and time.time() < deadline - 3.0:
        polished = _optimize_full(best_pts, max_iter=5000, ftol=1e-15, gtol=1e-12)
        e = _precise_energy(polished)
        if e < best_energy:
            best_energy = e
            best_pts = polished
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    if not _valid(best_pts):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
