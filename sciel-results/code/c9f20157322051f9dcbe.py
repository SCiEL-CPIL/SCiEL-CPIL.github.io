"""Physics-based relaxation for packing 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Pack 50 equal circles in a unit circle using physics-based relaxation.
    Circles are treated as charged particles with strong repulsion,
    constrained to stay inside the unit disk via a soft wall potential.
    """
    np.random.seed(42)
    
    # Initial positions: random points inside the unit disk
    # Use a spiral or random initialization
    # Try multiple initializations and pick the best
    
    def energy_and_grad(x):
        """
        Compute the negative of the objective (we want to maximize min distance,
        so we minimize the negative). Actually, let's formulate as minimizing
        a smooth approximation to the constraint violations.
        
        We want to maximize r = min(min_{i<j} |c_i - c_j|/2, min_i(1-|c_i|))
        
        Instead, let's use a physics-based approach:
        - Repulsive force between circles: ~1/d^4 (strong)
        - Wall repulsion: ~1/(1-|c|)^4 when close to wall
        
        We minimize the total potential energy.
        """
        x = x.reshape(N, 2)
        
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Avoid division by zero on diagonal
        dist_diag = np.where(np.eye(N, dtype=bool), 1e10, dist)
        
        # Repulsive potential: sum over i<j of 1/d^4
        # Only count each pair once
        iu = np.triu_indices(N, k=1)
        pair_dist = dist[iu]
        pair_dist = np.maximum(pair_dist, 1e-8)  # avoid singularity
        repulsion = np.sum(1.0 / pair_dist**4)
        
        # Wall potential: sum over i of 1/(1-|c_i|)^4 for |c_i| > 0.5
        r = np.sqrt(np.sum(x**2, axis=1))
        wall_margin = np.maximum(1.0 - r, 1e-8)
        # Only apply strong wall force when close to boundary
        wall = np.sum(1.0 / wall_margin**4)
        
        # Total energy
        E = repulsion + wall
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Gradient of repulsion
        # d/dx_i [1/d_ij^4] = -4/d_ij^5 * (x_i - x_j)/d_ij ... let's be careful
        # Actually: d(1/d^4)/dx_i where d = |x_i - x_j|
        # = d(1/d^4)/dd * dd/dx_i
        # = -4/d^5 * (x_i - x_j)/d
        # = -4(x_i - x_j)/d^6
        
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = dist[i, j]
            if d < 1e-8:
                continue
            coeff = -4.0 / d**6
            diff_ij = x[i] - x[j]
            grad[i] += coeff * diff_ij
            grad[j] -= coeff * diff_ij
        
        # Gradient of wall potential
        # d(1/(1-r)^4)/dx_i where r = |x_i|
        # = d(1/(1-r)^4)/dr * dr/dx_i
        # = 4/(1-r)^5 * x_i/r
        for i in range(N):
            ri = r[i]
            if ri < 1e-8:
                continue
            margin = 1.0 - ri
            if margin < 1e-8:
                margin = 1e-8
            coeff = 4.0 / margin**5 * 1.0 / ri
            grad[i] += coeff * x[i]
        
        return E, grad.ravel()
    
    def score(x):
        """Compute the actual score: the largest common radius."""
        x = x.reshape(N, 2)
        # Check all points are inside
        r = np.sqrt(np.sum(x**2, axis=1))
        if np.any(r > 1.0):
            return 0.0
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e10)
        min_pair_dist = np.min(dist)
        
        # Wall distance
        min_wall_dist = np.min(1.0 - r)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def try_initialization(init_x):
        """Run optimization from a given initial position."""
        init_x = init_x.reshape(-1)
        
        result = minimize(
            energy_and_grad,
            init_x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-8}
        )
        
        return result.x
    
    # Multiple initializations
    best_score = -1
    best_x = None
    
    # Initialization 1: Random points in disk
    for attempt in range(5):
        # Random initialization
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.8
        init_x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        x_opt = try_initialization(init_x)
        s = score(x_opt)
        if s > best_score:
            best_score = s
            best_x = x_opt
    
    # Initialization 2: Ring-based
    for attempt in range(3):
        # Place points on concentric rings
        x_init = np.zeros((N, 2))
        # 1 at center
        x_init[0] = [0, 0]
        # 6 on first ring
        if N >= 7:
            ring1_r = 0.3
            for i in range(1, min(7, N)):
                angle = 2 * np.pi * (i - 1) / 6
                x_init[i] = [ring1_r * np.cos(angle), ring1_r * np.sin(angle)]
        
        # 12 on second ring
        if N >= 13:
            ring2_r = 0.6
            for i in range(7, min(13, N)):
                angle = 2 * np.pi * (i - 7) / 12
                x_init[i] = [ring2_r * np.cos(angle), ring2_r * np.sin(angle)]
        
        # 12 on third ring
        if N >= 25:
            ring3_r = 0.85
            for i in range(13, min(25, N)):
                angle = 2 * np.pi * (i - 13) / 12
                x_init[i] = [ring3_r * np.cos(angle), ring3_r * np.sin(angle)]
        
        # Fill remaining on outer ring
        if N >= 26:
            ring4_r = 0.95
            remaining = N - 25
            for i in range(25, N):
                angle = 2 * np.pi * (i - 25) / max(remaining, 1)
                x_init[i] = [ring4_r * np.cos(angle), ring4_r * np.sin(angle)]
        
        x_opt = try_initialization(x_init)
        s = score(x_opt)
        if s > best_score:
            best_score = s
            best_x = x_opt
    
    # Initialization 3: Hexagonal lattice
    for spacing in [0.18, 0.19, 0.20]:
        x_init = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                px = (i + 0.5 * (j % 2)) * spacing
                py = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(px**2 + py**2) <= 1.0 - spacing / 2.0:
                    x_init.append([px, py])
        
        x_init = np.array(x_init)
        if len(x_init) >= N:
            # Take the first N (closest to center)
            r = np.sqrt(np.sum(x_init**2, axis=1))
            idx = np.argsort(r)[:N]
            x_init = x_init[idx]
            
            x_opt = try_initialization(x_init)
            s = score(x_opt)
            if s > best_score:
                best_score = s
                best_x = x_opt
    
    # Final polish: use the best solution and do a final local optimization
    if best_x is not None:
        # Do a few more iterations from the best solution
        result = minimize(
            energy_and_grad,
            best_x.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        x_final = result.x.reshape(N, 2)
        s_final = score(x_final)
        if s_final > best_score:
            best_x = x_final
            best_score = s_final
    
    # Ensure all points are strictly inside
    r = np.sqrt(np.sum(best_x**2, axis=1))
    for i in range(N):
        if r[i] >= 1.0:
            best_x[i] *= 0.999
    
    return best_x

# EVOLVE-BLOCK-END
