import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 130.0  # Safety margin from 150s limit

def _compute_radius(centers):
    """
    Compute the exact minimum radius for a set of centers.
    centers: (N, 2) array.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check boundaries
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(dx, dy)
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances using cdist for speed
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return min(min_boundary, min_pairwise)

def _soft_min(z, beta):
    """
    Compute the soft minimum of a vector z.
    """
    z_min = np.min(z)
    z_shifted = z - z_min
    terms = np.exp(-beta * z_shifted)
    log_sum = np.log(np.sum(terms))
    return z_min - (1.0 / beta) * log_sum

def _objective(x, beta):
    """
    Objective function to minimize: -soft_min(distances).
    """
    centers = x.reshape(N, 2)
    
    # Distances to boundary
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(dx, dy)
    
    # Pairwise distances using cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pairwise_dists = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([pairwise_dists.flatten(), boundary_dists])
    
    # Soft min
    r_soft = _soft_min(all_dists, beta)
    
    return -r_soft

def _objective_grad(x, beta):
    """
    Gradient of the objective function.
    """
    centers = x.reshape(N, 2)
    
    # Distances to boundary
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(dx, dy)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    pairwise_dists = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([pairwise_dists.flatten(), boundary_dists])
    
    # Soft min and its gradient
    z_min = np.min(all_dists)
    z_shifted = all_dists - z_min
    weights = np.exp(-beta * z_shifted)
    weights /= np.sum(weights)
    
    # Gradient of soft_min with respect to all_dists
    grad_all_dists = -weights
    
    # Gradient with respect to centers
    grad_centers = np.zeros_like(centers)
    
    # Pairwise distances gradient
    # For each pair (i, j), the distance is |c_i - c_j| / 2
    # d|c_i - c_j| / dc_i = (c_i - c_j) / |c_i - c_j|
    # So d(dist/2)/dc_i = (c_i - c_j) / (2 * |c_i - c_j|)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                # Index in pairwise_dists for (i, j)
                idx = i * N + j
                w = grad_all_dists[idx]
                # Gradient contribution to center i
                grad_centers[i] += w * d / (2 * dist)
                # Gradient contribution to center j
                grad_centers[j] -= w * d / (2 * dist)
    
    # Boundary distances gradient
    # boundary_dists[k] = min(0.5 - |x_k|, 0.5 - |y_k|)
    for k in range(N):
        w = grad_all_dists[-(N - k) + N * (N - 1)]  # Index in all_dists
        # Actually, boundary_dists is appended after pairwise_dists
        # pairwise_dists has N*N elements (with diagonal set to inf, but we use all)
        # Wait, cdist gives N*N, we fill diagonal with inf, but for gradient we need to be careful
        # Let's re-index: all_dists = [pairwise_dists.flatten(), boundary_dists]
        # pairwise_dists has N*N elements
        # boundary_dists has N elements
        # So index for boundary_dists[k] is N*N + k
        
        idx_boundary = N * N + k
        
        # Gradient of boundary_dists[k] with respect to centers[k]
        # boundary_dists[k] = min(0.5 - |x_k|, 0.5 - |y_k|)
        # If 0.5 - |x_k| <= 0.5 - |y_k|, i.e., |x_k| >= |y_k|, then grad_x = -sign(x_k), grad_y = 0
        # Otherwise, grad_x = 0, grad_y = -sign(y_k)
        
        if abs(centers[k, 0]) >= abs(centers[k, 1]):
            grad_centers[k, 0] += w * (-np.sign(centers[k, 0]))
        else:
            grad_centers[k, 1] += w * (-np.sign(centers[k, 1]))
    
    # The objective is -soft_min, so gradient is -grad_soft_min
    # grad_soft_min = d(soft_min)/dx = grad_all_dists @ (d all_dists / dx)
    # We computed grad_all_dists = -weights
    # And we computed d all_dists / dx contributions
    # So grad_objective = - (grad_all_dists @ d all_dists / dx)
    # But we accumulated grad_centers as sum of w * d(dist)/dc, where w = grad_all_dists[idx]
    # So grad_centers is actually grad_all_dists @ (d all_dists / dx)
    # Therefore, grad_objective = -grad_centers
    
    return -grad_centers.flatten()

def _hexagonal_init(jitter=0.02):
    """
    Generate a jittered hexagonal lattice initialization.
    """
    s = 0.14
    dx = s
    dy = s * np.sqrt(3) / 2
    
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        points = points[indices]
    elif len(points) < N:
        missing = N - len(points)
        random_points = np.random.uniform(-0.4, 0.4, size=(missing, 2))
        points = np.vstack([points, random_points])
        
    points += np.random.normal(0, jitter, size=points.shape)
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    return points

def _random_init():
    """
    Random initialization with repulsion relaxation.
    """
    points = np.random.uniform(-0.45, 0.45, size=(N, 2))
    for _ in range(50):
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 0.15:
                    f = d / (dist + 1e-6) * (0.15 - dist)
                    forces[i] += f
                    forces[j] -= f
        points += 0.1 * forces
        points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
        points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    return points

def construct():
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    # Generate multiple initializations
    inits = []
    inits.append(_hexagonal_init(jitter=0.02))
    inits.append(_hexagonal_init(jitter=0.05))
    inits.append(_random_init())
    
    # Add a perturbed grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    grid += np.random.normal(0, 0.03, size=grid.shape)
    inits.append(grid)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for i_init, init_centers in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        x_current = init_centers.flatten()
        
        # Annealing schedule: start with low beta, increase gradually
        beta_schedule = [10.0, 20.0, 40.0, 80.0, 160.0, 300.0]
        
        for beta in beta_schedule:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            try:
                res = opt.minimize(
                    _objective, 
                    x_current, 
                    jac=_objective_grad,
                    method='SLSQP', 
                    bounds=bounds, 
                    args=(beta,),
                    options={'maxiter': 100, 'ftol': 1e-12, 'eps': 1e-8}
                )
                
                x_current = res.x
                
                # Calculate actual radius
                r = _compute_radius(x_current.reshape(N, 2))
                if r > best_radius:
                    best_radius = r
                    best_centers = x_current.reshape(N, 2)
                    
            except Exception:
                # If SLSQP fails, try L-BFGS-B as fallback
                try:
                    res = opt.minimize(
                        _objective, 
                        x_current, 
                        method='L-BFGS-B', 
                        bounds=bounds, 
                        args=(beta,),
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    x_current = res.x
                    r = _compute_radius(x_current.reshape(N, 2))
                    if r > best_radius:
                        best_radius = r
                        best_centers = x_current.reshape(N, 2)
                except Exception:
                    continue
        
        # Final polish with very high beta
        if time.time() - start_time < DEADLINE - 3:
            try:
                res_final = opt.minimize(
                    _objective, 
                    x_current, 
                    jac=_objective_grad,
                    method='SLSQP', 
                    bounds=bounds, 
                    args=(500.0,),
                    options={'maxiter': 50, 'ftol': 1e-14, 'eps': 1e-10}
                )
                x_final = res_final.x
                r_final = _compute_radius(x_final.reshape(N, 2))
                if r_final > best_radius:
                    best_radius = r_final
                    best_centers = x_final.reshape(N, 2)
            except Exception:
                pass

    # Fallback if no valid center found
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers

if __name__ == "__main__":
    c = construct()
    print("Radius:", _compute_radius(c))
