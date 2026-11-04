"""Improved packer: BFGS optimization on Log-Sum-Exp smooth minimum objective."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100


def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)


def _lse_energy_and_grad(centers, M):
    """
    Energy function using Log-Sum-Exp approximation of the minimum radius.
    Objective: Minimize f(x) = (1/M) * log( sum exp(-M * z_k) )
    where z_k are the radii (pairwise/2 and wall clearances).
    This approximates -min(z_k). Minimizing this maximizes min(z_k).
    """
    c = centers.reshape(-1, 3)
    
    # --- Pairwise Terms ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e-12) # Avoid div by 0, though we mask later
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # z_ij = dist / 2
    pair_z = pair_dists * 0.5
    
    # exp(-M * z)
    pair_terms = np.exp(-M * pair_z)
    
    # --- Wall Terms ---
    # For each point, distance to each of the 6 walls
    # Walls: x=0.5, x=-0.5, y=0.5, y=-0.5, z=0.5, z=-0.5
    # Clearance to wall is min(0.5 - x, 0.5 + x, ...)
    # We compute exp(-M * clearance) for each wall and sum them.
    # To be consistent with the "min" approximation, we sum exp(-M * d_wall) for all walls.
    # The LSE of these 6 terms approximates -min(clearance).
    
    # c is (N, 3)
    # x: c[:, 0], y: c[:, 1], z: c[:, 2]
    
    # Distances to walls:
    # 0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z
    wall_clearances = np.concatenate([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    
    wall_terms = np.exp(-M * wall_clearances)
    
    # --- Sum and Log ---
    total_sum = np.sum(pair_terms) + np.sum(wall_terms)
    energy = (1.0 / M) * np.log(total_sum)
    
    # --- Gradient ---
    # grad = - (1/S) * sum ( exp(-M*z_k) * grad_z_k )
    # Note: The 1/M in energy cancels with -M in derivative of exp(-M*z)
    # d/dx [ (1/M) log(sum exp(-M z)) ] = (1/M) * (1/S) * sum( exp(-M z) * (-M grad_z) )
    # = - (1/S) * sum( exp(-M z) * grad_z )
    
    S = total_sum
    
    # Gradient of pairwise terms
    # For pair (i, j) with dist d_ij and weight w_ij = exp(-M * d_ij / 2)
    # grad_i contribution: w_ij * (1/S) * ( (c_i - c_j) / d_ij )  (since grad_z = 0.5 * unit_vec, and there is a - sign in formula)
    # Wait, let's re-verify signs.
    # f = (1/M) log S.
    # df/dx = (1/M S) dS/dx.
    # dS/dx = sum d(exp(-M z_k))/dx = sum exp(-M z_k) * (-M dz_k/dx).
    # df/dx = - (1/S) sum exp(-M z_k) dz_k/dx.
    
    # For pair (i, j):
    # z = d_ij / 2.
    # dz/dc_i = 0.5 * (c_i - c_j) / d_ij.
    # dz/dc_j = -0.5 * (c_i - c_j) / d_ij.
    
    # So contribution to grad_i from pair (i,j) is:
    # - (w_ij / S) * 0.5 * (c_i - c_j) / d_ij
    
    # Contribution to grad_j:
    # - (w_ij / S) * (-0.5 * (c_i - c_j) / d_ij) = + (w_ij / S) * 0.5 * (c_i - c_j) / d_ij
    
    # Let's build the gradient array
    grad = np.zeros((N, 3))
    
    # We need to accumulate gradients for each point from all pairs
    # It is more efficient to compute the full matrix of weights and vectorize.
    
    # Recalculate full dist matrix and unit vectors for all pairs
    # We already have diff and dists.
    # unit_vecs[i, j] = (c_i - c_j) / d_ij
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # Weights matrix W[i, j] = exp(-M * d_ij / 2)
    # We only care about i < j, but the formula is symmetric.
    # Let's compute full W.
    W = np.exp(-M * dists * 0.5)
    np.fill_diagonal(W, 0.0)
    
    # Grad_pair_i = - (1/S) * sum_j W[i, j] * 0.5 * unit_vecs[i, j]
    # Note: unit_vecs[i, j] = (c_i - c_j)/d_ij.
    
    grad_pair = -0.5 * (W[:, :, np.newaxis] * unit_vecs).sum(axis=1) / S
    
    # Wall Gradient
    # z_wall_k = clearance to wall k.
    # For wall x=0.5: z = 0.5 - c_x. dz/dc_x = -1.
    # For wall x=-0.5: z = 0.5 + c_x. dz/dc_x = 1.
    # ...
    # Contribution to grad: - (w_wall / S) * dz/dc
    
    # Let's compute wall weights
    # wall_terms is (N, 6)
    # w_x_pos = exp(-M(0.5 - c_x)) -> dz/dc_x = -1
    # w_x_neg = exp(-M(0.5 + c_x)) -> dz/dc_x = 1
    # w_y_pos = exp(-M(0.5 - c_y)) -> dz/dc_y = -1
    # w_y_neg = exp(-M(0.5 + c_y)) -> dz/dc_y = 1
    # w_z_pos = exp(-M(0.5 - c_z)) -> dz/dc_z = -1
    # w_z_neg = exp(-M(0.5 + c_z)) -> dz/dc_z = 1
    
    # grad_wall_i = - (1/S) * ( w_x_pos * (-1) * e_x + w_x_neg * (1) * e_x + ... )
    # = (1/S) * ( (w_x_pos - w_x_neg) e_x + (w_y_pos - w_y_neg) e_y + (w_z_pos - w_z_neg) e_z )
    
    w_x_pos = wall_terms[:, 0]
    w_x_neg = wall_terms[:, 1]
    w_y_pos = wall_terms[:, 2]
    w_y_neg = wall_terms[:, 3]
    w_z_pos = wall_terms[:, 4]
    w_z_neg = wall_terms[:, 5]
    
    grad_wall = np.zeros((N, 3))
    grad_wall[:, 0] = (w_x_pos - w_x_neg) / S
    grad_wall[:, 1] = (w_y_pos - w_y_neg) / S
    grad_wall[:, 2] = (w_z_pos - w_z_neg) / S
    
    total_grad = (grad_pair + grad_wall).flatten()
    
    return energy, total_grad


def _fcc_init(n_points):
    """
    Initialize points in an FCC lattice structure clipped to the cube.
    FCC lattice points: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0) + integer shifts.
    We generate a large grid of FCC points and pick the first n_points that fit in [-0.5, 0.5].
    """
    points = []
    # FCC basis
    basis = [
        (0, 0, 0),
        (0, 0.5, 0.5),
        (0.5, 0, 0.5),
        (0.5, 0.5, 0)
    ]
    
    # Generate points in a range slightly larger than the cube
    # Scale factor 1.0 means unit cell size 1.
    # We need to fill [-0.5, 0.5].
    # Let's iterate over integer grid indices -2 to 2 to be safe.
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                for b in basis:
                    p = (i + b[0], j + b[1], k + b[2])
                    # Check if inside [-0.5, 0.5]
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    # If we don't have enough, we can add random points or scale the lattice
    # For N=100, this should be enough.
    # Let's count: 
    # i,j,k in -2..2 (5^3=125 cells). 4 points per cell = 500 points.
    # Many will be outside. 
    # In [-0.5, 0.5], we have 1 unit cube.
    # FCC density is ~0.74.
    # Volume 1. ~74 spheres of radius 1/(2*sqrt(2)) ~ 0.35? No.
    # In a unit cube, FCC with lattice constant a=1 has 4 atoms per unit cell.
    # If we place FCC points with lattice constant 1, they are at (0,0,0) etc.
    # Distance between (0,0,0) and (0.5, 0.5, 0) is sqrt(0.5) ~ 0.707.
    # Radius would be ~0.35.
    # But we only have 1 unit cube.
    # The points (0,0,0) is a corner. (0.5, 0.5, 0) is face center.
    # We need to shift the lattice to center it?
    # The task is to pack 100 spheres.
    # A simple cubic grid of 5x5x5 has 125 points.
    # FCC is denser.
    
    # Let's just use the generated points.
    if len(points) < n_points:
        # Fallback to grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:n_points].copy()
        
    return np.array(points[:n_points])


def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin under 120s

    # ---- Seed: FCC initialization ----
    pts = _fcc_init(N)
    
    # Ensure points are strictly inside
    pts = np.clip(pts, -0.49, 0.49)
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)

    # Strategy: Run BFGS with LSE objective, increasing sharpness M
    # Small M: smooth, global-ish. Large M: sharp, local polish.
    
    M_values = [10, 50, 100, 200]
    
    for M in M_values:
        if time.time() - t0 > DEADLINE:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.01, size=(N, 3))
        init_pts = np.clip(pts + noise, -0.49, 0.49)
        
        def objective(x):
            e, g = _lse_energy_and_grad(x, M)
            return e, g
            
        try:
            result = minimize(
                objective,
                init_pts.flatten(),
                method='BFGS',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-9, 'disp': False}
            )
            opt_pts = result.x.reshape(-1, 3)
            
            # Validate
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
                    pts = best_pts.copy() # Update seed for next M
        except Exception:
            pass

    # If time permits, try a few random restarts with the best M
    if time.time() - t0 < DEADLINE - 10.0:
        for i in range(5):
            if time.time() - t0 > DEADLINE:
                break
            noise = rng.normal(0, 0.05, size=(N, 3))
            init_pts = np.clip(pts + noise, -0.49, 0.49)
            
            def objective_rand(x):
                e, g = _lse_energy_and_grad(x, 200)
                return e, g
                
            try:
                result = minimize(
                    objective_rand,
                    init_pts.flatten(),
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 500, 'gtol': 1e-9, 'disp': False}
                )
                opt_pts = result.x.reshape(-1, 3)
                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                    r_opt = _score(opt_pts)
                    if r_opt > best_r:
                        best_r = r_opt
                        best_pts = opt_pts.copy()
            except Exception:
                pass

    # Final safety
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
        
    return best_pts
