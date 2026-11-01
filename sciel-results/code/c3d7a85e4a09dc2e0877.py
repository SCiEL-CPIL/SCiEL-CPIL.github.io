"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (since d/dx_i [-log|x_i-x_j|] = -(x_i-x_j)/|x_i-x_j|^2, but energy is -log so gradient is (x_i-x_j)/|x_i-x_j|^2... 
    wait, let me re-derive:
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j| = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    So gradient is -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    """
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = -sum_j diff[i,j] / dist_sq[i,j]
    # Note: we sum over all j != i, but the factor of 2 issue:
    # E = sum_{i<j} -log(d_ij)
    # dE/dx_i = sum_{j>i} -1/d_ij * (x_i - x_j)/d_ij + sum_{j<i} -1/d_ij * (x_i - x_j)/d_ij
    #         = -sum_{j != i} (x_i - x_j) / d_ij^2
    # So:
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _energy_only(pts):
    """Compute energy only (faster, no gradient)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_init(n, dims=3):
    """Halton sequence initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        pts = []
        for b in [2, 3, 5]:
            pts.append(halton(i, b))
        # Map [0,1]^3 to sphere using spherical coordinates
        u, v, w = pts[0], pts[1], pts[2]
        # Use a mapping that distributes on sphere
        theta = 2 * np.pi * u
        cos_phi = 2 * v - 1
        sin_phi = np.sqrt(max(0, 1 - cos_phi**2))
        p = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        points.append(p)
    return np.array(points)


def _lbfgs_minimize(pts, max_iter=5000, tol=1e-12, deadline=None):
    """Custom L-BFGS with sphere projection."""
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # L-BFGS memory
    m = 10
    s_list = []  # step directions
    y_list = []  # gradient differences
    W_list = []  # rho values
    
    g = None
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if g is None:
            e, g = _energy_and_grad(pts)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
        
        # Check convergence
        if np.linalg.norm(g) < tol:
            break
        
        # L-BFGS two-loop recursion to compute direction
        # r = -g
        r = -g.copy()
        alphas = np.zeros(m)
        betas = np.zeros(m)
        
        # Two-loop recursion
        q = r.copy()
        for i in range(len(s_list) - 1, -1, -1):
            alphas[i] = W_list[i] * np.dot(s_list[i], q)
            q -= alphas[i] * y_list[i]
        
        # Initial Hessian approximation
        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
        else:
            gamma = 1.0
        r = gamma * q
        
        for i in range(len(s_list)):
            betas[i] = W_list[i] * np.dot(y_list[i], r)
            r += (alphas[i] - betas[i]) * s_list[i]
        
        # Line search (backtracking)
        step = r
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-14:
            break
        step = step / step_norm * min(1.0, step_norm)
        
        alpha = 1.0
        for ls in range(30):
            new_pts = _project_to_sphere(pts + alpha * step)
            new_e = _energy_only(new_pts)
            if new_e < e - 1e-4 * alpha * np.dot(g, step):
                break
            alpha *= 0.5
        
        # Update
        new_g = None
        new_pts = _project_to_sphere(pts + alpha * step)
        new_e, new_g = _energy_and_grad(new_pts)
        
        if new_e < best_energy:
            best_energy = new_e
            best_pts = new_pts.copy()
        
        # Update L-BFGS memory
        s = alpha * step
        y = new_g - g
        sy = np.dot(s, y)
        if sy > 1e-12:
            rho = 1.0 / sy
            s_list.append(s)
            y_list.append(y)
            W_list.append(rho)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                W_list.pop(0)
        
        pts = new_pts
        e = new_e
        g = new_g
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 120s limit, leave 5s buffer
    
    # Try multiple initializations and pick the best
    inits = []
    inits.append(_fibonacci_init(N))
    inits.append(_halton_init(N))
    
    # Add a few random inits
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        inits.append(pts)
    
    best_pts = None
    best_energy = np.inf
    
    for init in inits:
        if time.time() > deadline:
            break
        pts, energy = _lbfgs_minimize(init, max_iter=3000, tol=1e-12, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Final polish with tighter tolerance
    if best_pts is not None and time.time() < deadline - 5.0:
        pts, energy = _lbfgs_minimize(best_pts, max_iter=5000, tol=1e-14, deadline=deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_init(N)
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to fibonacci
        best_pts = _fibonacci_init(N)
    
    return best_pts
# EVOLVE-BLOCK-END
