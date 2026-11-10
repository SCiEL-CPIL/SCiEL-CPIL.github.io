"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 100

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _energy_and_grad_spherical(params):
    """
    Compute logarithmic energy and gradients with respect to spherical coordinates.
    params: (N, 2) array of [theta, phi]
    theta in [0, pi], phi in [0, 2*pi]
    """
    n = len(params)
    theta = params[:, 0]
    phi = params[:, 1]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Avoid numerical issues at poles
    sin_theta_safe = np.maximum(sin_theta, 1e-15)

    pts = np.column_stack([
        sin_theta_safe * cos_phi,
        sin_theta_safe * sin_phi,
        cos_theta
    ])

    # Compute pairwise dot products
    c = pts @ pts.T

    # Mask diagonal
    idx = np.triu_indices(n, k=1)
    c_ij = c[idx]

    # Energy E = -sum log(sqrt(2 - 2c)) = -0.5 sum log(2 - 2c)
    val = 2.0 - 2.0 * c_ij
    val = np.maximum(val, 1e-15)

    energy = -0.5 * np.sum(np.log(val))

    # Gradients
    cos_dphi = np.cos(phi[:, np.newaxis] - phi[np.newaxis, :])
    sin_dphi = np.sin(phi[:, np.newaxis] - phi[np.newaxis, :])

    # dc/dtheta_i
    dc_dti = (cos_theta[:, np.newaxis] * sin_theta[np.newaxis, :] * cos_dphi
              - sin_theta[:, np.newaxis] * cos_theta[np.newaxis, :])

    # dc/dphi_i
    dc_dphi_i = -sin_theta[:, np.newaxis] * sin_theta[np.newaxis, :] * sin_dphi

    denom = 1.0 - c
    mask = ~np.eye(n, dtype=bool)

    with np.errstate(divide='ignore', invalid='ignore'):
        grad_theta_mat = np.where(mask, dc_dti / np.where(denom > 1e-15, denom, 1.0), 0.0)
        grad_phi_mat = np.where(mask, dc_dphi_i / np.where(denom > 1e-15, denom, 1.0), 0.0)

    grad_theta = np.sum(grad_theta_mat, axis=1)
    grad_phi = np.sum(grad_phi_mat, axis=1)

    grads = np.column_stack([grad_theta, grad_phi])

    return energy, grads


def _spherical_to_cartesian(params):
    """Convert spherical coordinates to Cartesian on unit sphere."""
    theta = params[:, 0]
    phi = params[:, 1]
    st = np.sin(theta)
    ct = np.cos(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    pts = np.column_stack([st * cp, st * sp, ct])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _cartesian_to_spherical(pts):
    """Convert Cartesian points to spherical coordinates."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.column_stack([theta, phi])


def _make_fibonacci_sphere_params():
    """Generate Fibonacci spiral points in spherical coordinates."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    theta = np.arccos(z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.column_stack([theta, phi])


def _random_sphere_params(rng):
    """Generate random spherical coordinates."""
    theta = rng.uniform(0, np.pi, N)
    phi = rng.uniform(0, 2 * np.pi, N)
    return np.column_stack([theta, phi])


def _spectral_init(rng, n_points=100):
    """
    Generate initial points using spectral method on a Delaunay triangulation.
    This uses the graph Laplacian of a random point cloud's convex hull to
    find a more uniform distribution.
    """
    # Generate random points on sphere
    pts = rng.standard_normal((n_points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    try:
        # Create Delaunay triangulation of the point cloud
        # For points on a sphere, we use 3D convex hull
        hull = ConvexHull(pts)
        
        # Build adjacency matrix from hull facets
        n = n_points
        adj = np.zeros((n, n))
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    adj[simplex[i], simplex[j]] = 1
                    adj[simplex[j], simplex[i]] = 1
        
        # Compute graph Laplacian L = D - A
        degree = np.sum(adj, axis=1)
        L = np.diag(degree) - adj
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Use eigenvectors corresponding to small eigenvalues (excluding trivial ones)
        # Skip the first few (constant and linear modes)
        start_idx = 3
        if start_idx >= n:
            start_idx = 1
        
        # Take next k eigenvectors where k gives us 3 coordinates
        # We want to embed in 3D, so take eigenvectors 3, 4, 5 (indices)
        # But we need to be careful: for n=100, indices 3,4,5 should exist
        if start_idx + 2 < n:
            v1 = eigvecs[:, start_idx]
            v2 = eigvecs[:, start_idx + 1]
            v3 = eigvecs[:, start_idx + 2]
        else:
            v1 = eigvecs[:, 1]
            v2 = eigvecs[:, 2]
            v3 = eigvecs[:, 3]
        
        # Normalize eigenvectors
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        v3 /= np.linalg.norm(v3)
        
        # Create new points from eigenvector components
        new_pts = np.column_stack([v1, v2, v3])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-15)
        
        # Convert to spherical
        return _cartesian_to_spherical(new_pts)
        
    except Exception:
        # Fallback to Fibonacci if spectral method fails
        return _make_fibonacci_sphere_params()


def _adam_spherical(params0, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, max_iters=3000):
    """Adam optimizer in spherical coordinates."""
    n = len(params0)
    params = params0.copy()
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 0

    best_energy = _energy_and_grad_spherical(params)[0]
    best_params = params.copy()

    for it in range(max_iters):
        if _remaining_time() < 15.0:
            break

        energy, grad = _energy_and_grad_spherical(params)

        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = -lr * m_hat / (np.sqrt(v_hat) + eps)

        # Update parameters
        params = params + update

        # Clip theta to valid range
        params[:, 0] = np.clip(params[:, 0], 1e-6, np.pi - 1e-6)
        # Wrap phi
        params[:, 1] = params[:, 1] % (2 * np.pi)

    return best_params, best_energy


def _lbfgs_spherical(params0, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """L-BFGS-B optimization in spherical coordinates."""
    if _remaining_time() < 15.0:
        return None, float('inf')

    try:
        res = minimize(
            _energy_and_grad_spherical,
            params0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
        )
        return res.x, res.fun
    except Exception:
        return None, float('inf')


def _perturb_params(params, scale, rng):
    """Perturb spherical coordinates by given scale."""
    perturbed = params.copy()
    noise = rng.standard_normal((N, 2)) * scale
    perturbed[:, 0] += noise[:, 0]
    perturbed[:, 1] += noise[:, 1]
    perturbed[:, 0] = np.clip(perturbed[:, 0], 1e-6, np.pi - 1e-6)
    perturbed[:, 1] = perturbed[:, 1] % (2 * np.pi)
    return perturbed


def _high_precision_polish(params, maxiter=5000):
    """
    High-precision polish using very tight tolerances.
    This is the final refinement step to squeeze out last bits of energy.
    """
    if _remaining_time() < 10.0:
        return params, float('inf')
    
    try:
        # Use L-BFGS with very tight tolerances
        res = minimize(
            _energy_and_grad_spherical,
            params,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': 1e-16,
                'gtol': 1e-14,
                'maxls': 100
            }
        )
        if res.fun < float('inf'):
            return res.x, res.fun
    except Exception:
        pass
    
    # Fallback: Adam with very small learning rate
    if _remaining_time() > 5.0:
        try:
            params_adam, energy_adam = _adam_spherical(params, lr=0.0001, max_iters=2000)
            return params_adam, energy_adam
        except Exception:
            pass
    
    return params, float('inf')


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    global _START_TIME
    _START_TIME = time.time()

    rng = np.random.default_rng(42)

    best_params = None
    best_energy = float('inf')
    best_pts = None

    def _update_best(params, energy):
        nonlocal best_params, best_energy, best_pts
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()
            best_pts = _spherical_to_cartesian(params)

    # Phase 1: Multiple initializations
    # 1a: Fibonacci spiral
    fib_params = _make_fibonacci_sphere_params()
    params_opt, energy_opt = _lbfgs_spherical(fib_params, maxiter=1500, ftol=1e-13, gtol=1e-10)
    if params_opt is not None and energy_opt < best_energy:
        _update_best(params_opt, energy_opt)

    # 1b: Spectral initialization from Delaunay triangulation
    if _remaining_time() > 60.0:
        spectral_params = _spectral_init(rng)
        params_opt2, energy_opt2 = _lbfgs_spherical(spectral_params, maxiter=1500, ftol=1e-13, gtol=1e-10)
        if params_opt2 is not None and energy_opt2 < best_energy:
            _update_best(params_opt2, energy_opt2)

    # 1c: Another random spectral init
    if _remaining_time() > 50.0:
        spectral_params2 = _spectral_init(rng)
        params_opt3, energy_opt3 = _lbfgs_spherical(spectral_params2, maxiter=1000, ftol=1e-12, gtol=1e-10)
        if params_opt3 is not None and energy_opt3 < best_energy:
            _update_best(params_opt3, energy_opt3)

    # Phase 2: Adam refinement from best solution
    if best_params is not None and _remaining_time() > 40.0:
        params_adam, energy_adam = _adam_spherical(best_params, lr=0.003, max_iters=2000)
        if energy_adam < best_energy:
            _update_best(params_adam, energy_adam)

    # Phase 3: Multi-scale perturbations of best solution
    if best_params is not None and _remaining_time() > 30.0:
        scales = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
        for scale in scales:
            if _remaining_time() < 20.0:
                break
            for attempt in range(2):
                if _remaining_time() < 15.0:
                    break
                perturbed = _perturb_params(best_params, scale, rng)
                # Quick Adam pass
                params_a, energy_a = _adam_spherical(perturbed, lr=0.002, max_iters=600)
                if energy_a < best_energy:
                    _update_best(params_a, energy_a)
                # Follow with L-BFGS polish
                if _remaining_time() > 15.0:
                    params_l, energy_l = _lbfgs_spherical(params_a, maxiter=500, ftol=1e-13, gtol=1e-10)
                    if params_l is not None and energy_l < best_energy:
                        _update_best(params_l, energy_l)

    # Phase 4: Random restarts with spectral init
    if _remaining_time() > 25.0:
        for r in range(5):
            if _remaining_time() < 15.0:
                break
            # Alternate between random and spectral
            if r % 2 == 0:
                rand_params = _random_sphere_params(rng)
            else:
                rand_params = _spectral_init(rng)
            
            params_a, energy_a = _adam_spherical(rand_params, lr=0.005, max_iters=1000)
            if energy_a < best_energy:
                _update_best(params_a, energy_a)
            if _remaining_time() > 15.0:
                params_l, energy_l = _lbfgs_spherical(params_a, maxiter=600, ftol=1e-14, gtol=1e-10)
                if params_l is not None and energy_l < best_energy:
                    _update_best(params_l, energy_l)

    # Phase 5: Final high-precision polish
    if best_params is not None and _remaining_time() > 10.0:
        # Tighter L-BFGS
        params_final, energy_final = _high_precision_polish(best_params, maxiter=5000)
        if params_final is not None and energy_final < best_energy:
            _update_best(params_final, energy_final)

        # One more Adam pass with very small lr
        if _remaining_time() > 8.0:
            params_adam_final, energy_adam_final = _adam_spherical(
                best_params, lr=0.0003, max_iters=1500
            )
            if energy_adam_final < best_energy:
                _update_best(params_adam_final, energy_adam_final)

            # Final L-BFGS polish
            if _remaining_time() > 5.0:
                params_last, energy_last = _lbfgs_spherical(
                    best_params, maxiter=3000, ftol=1e-15, gtol=1e-13
                )
                if params_last is not None and energy_last < best_energy:
                    _update_best(params_last, energy_last)

    # Fallback if nothing worked
    if best_pts is None:
        fib_params = _make_fibonacci_sphere_params()
        best_pts = _spherical_to_cartesian(fib_params)

    # Final validation and correction
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)

    # Check for close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    if min_dist < 1e-9:
        for _ in range(200):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            rand_dir = rng.standard_normal(3)
            rand_dir /= np.linalg.norm(rand_dir)
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-15)

    return best_pts

# EVOLVE-BLOCK-END
