import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Objective function: Minimize the ratio of max to min distance.
    # Using log of ratio to handle scale invariance better, or directly the ratio.
    # The task asks for (max_dist / min_dist)^2.
    # Let's minimize f(x) = (max_d / min_d). Squaring is monotonic for positive values.
    # To avoid division by zero, we add a small epsilon or handle it in the objective.
    
    def objective(pts):
        # pts: (N, 2)
        # Calculate pairwise distances
        # diff[i, j, k] = pts[i, k] - pts[j, k]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        d_flat = dists[mask]
        
        d_min = d_flat.min()
        d_max = d_flat.max()
        
        if d_min < 1e-9:
            return 1e10
        return (d_max / d_min) ** 2

    # Physics-based relaxation strategy
    # 1. Initialize points in a random or grid-like fashion
    # 2. Apply forces:
    #    - Repulsion between all pairs: F_rep ~ 1/r^2 (or similar)
    #    - Attraction to center: F_att ~ -k * r
    #    This tends to form a compact cluster.
    # 3. Iterate until convergence or time limit.
    # 4. Final polish with scipy.optimize.minimize if time permits.

    def relax_points(initial_pts):
        pts = initial_pts.copy()
        # Parameters
        repulsion_strength = 1.0
        attraction_strength = 0.05
        damping = 0.9
        dt = 0.05
        
        for i in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Calculate forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion
            for i_idx in range(N):
                for j_idx in range(i_idx + 1, N):
                    diff = pts[i_idx] - pts[j_idx]
                    dist_sq = np.dot(diff, diff)
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-6:
                        # Random nudge if too close
                        force = np.random.randn(2) * 10.0
                    else:
                        force = (repulsion_strength / dist_sq) * (diff / dist)
                    
                    forces[i_idx] += force
                    forces[j_idx] -= force
            
            # Attraction to center (mean)
            center = np.mean(pts, axis=0)
            for i_idx in range(N):
                forces[i_idx] -= attraction_strength * (pts[i_idx] - center)
            
            # Update positions
            pts = pts + dt * forces
            # Optional: Damping is handled implicitly by dt, but let's keep it simple.
            
        return pts

    best_pts = None
    best_score = float('inf')
    
    # Try multiple restarts with different initializations
    # Initialization 1: Random within a circle
    # Initialization 2: Grid
    # Initialization 3: Hexagonal lattice
    
    initializations = []
    
    # 1. Random
    for _ in range(5):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.random.uniform(0, 2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
        
    # 2. Grid
    side = int(np.ceil(N ** 0.5))
    grid = list(np.meshgrid(range(side), range(side)))
    grid_pts = np.array([p for p in zip(*grid)][:N], dtype=float)
    # Center it
    grid_pts -= np.mean(grid_pts, axis=0)
    initializations.append(grid_pts)
    
    # 3. Hexagonal
    hex_pts = []
    r = 0.5
    # Generate hexagonal points
    for ring in range(0, 10):
        if len(hex_pts) >= N: break
        if ring == 0:
            hex_pts.append([0.0, 0.0])
        else:
            # 6 * ring points
            for k in range(6 * ring):
                if len(hex_pts) >= N: break
                angle = (k / (6.0 * ring)) * 2 * np.pi
                # Hexagonal distance is not circular, but this approximates
                # Better to use axial coordinates
                pass
            
    # Let's use a proper hexagonal generator
    def hex_coords(radius):
        coords = []
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                coords.append((q, r))
        return coords

    hex_list = hex_coords(4)
    hex_pts = []
    for q, r in hex_list[:N]:
        x = 1.5 * q
        y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
        hex_pts.append([x, y])
    hex_pts = np.array(hex_pts, dtype=float)
    hex_pts -= np.mean(hex_pts, axis=0)
    initializations.append(hex_pts)

    for init_pts in initializations:
        if time.time() > deadline - 20:
            break
            
        relaxed_pts = relax_points(init_pts)
        
        # Calculate score
        diff = relaxed_pts[:, np.newaxis, :] - relaxed_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(N, dtype=bool)
        d_flat = dists[mask]
        d_min = d_flat.min()
        d_max = d_flat.max()
        if d_min < 1e-9:
            score = 1e10
        else:
            score = (d_max / d_min) ** 2
            
        if score < best_score:
            best_score = score
            best_pts = relaxed_pts.copy()

    # Final polish with scipy if time allows
    if best_pts is not None and time.time() < deadline - 30:
        try:
            # Use Nelder-Mead or Powell for local refinement
            res = minimize(objective, best_pts, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if time.time() < deadline - 5:
                new_score = objective(res.x)
                if new_score < best_score:
                    best_pts = res.x
                    best_score = new_score
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
